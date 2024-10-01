import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision.models.detection.rpn import RPNHead
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.rpn import RegionProposalNetwork
import torch.nn.functional as F
import os
from Data_alignment import prepare_dataloader
from torchvision.models import resnet50, ResNet50_Weights
from roi_heads_copy import RoIHeads
from skimage.transform import resize

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from visualize_copy import display_instances
import numpy as np

# Define directories
csv_directory = r'F:\DLC_thermography-JLanden-2024-07-09\videos'
image_directory = r'F:\DLC_thermography-JLanden-2024-07-09\labeled-data'
segments_dirs = r'F:\Segmentation\training_segments'
save_to_dir = r'F:\Segmentation\results'

# Hyperparameters
num_epochs = 200
validate_every_n_epochs = 100
learning_rate = 0.0001
batch_size = 4  # Adjust as needed
val_split = 0.2  # 20% of the data for validation
num_classes = 3  # Set according to your dataset
height, width = 348, 464  # Replace with the actual size of your images
class_names = ['implanted_mouse', 'implanted_bat', 'implanted_rump',
               'extra_mouse', 'extra_bat', 'extra_rump']

# Scales & Aspect ratios:
# From Scales_AspectRatios_calc
anchor_generator = AnchorGenerator(
    sizes=((32, 64, 128), (64, 128, 256), (128, 256, 512),),
    aspect_ratios=((0.5, 1.0, 2.0), (0.5, 1.0, 2.0), (0.5, 1.0, 2.0),)
)


def move_to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, list):
        return [move_to_device(item, device) for item in obj]
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    else:
        return obj


def keypoints_to_heatmap(keypoints, H, W):
    batch_size, num_keypoints, _ = keypoints.size()
    heatmap = torch.zeros((batch_size, num_keypoints, H, W), device=keypoints.device)

    for i in range(batch_size):
        for j in range(num_keypoints):
            if keypoints[i, j, 2] > 0:  # Check if the keypoint is visible
                x = int((keypoints[i, j, 0] / W).clamp(0, W - 1))
                y = int((keypoints[i, j, 1] / H).clamp(0, H - 1))
                heatmap[i, j, y, x] = 1.0  # Set the corresponding heatmap location to 1

    return heatmap


def visualize_predictions(images, segments, predictions, epoch, batch_idx, phase, save_img_dir, class_names,
                          keypoints=None):
    print(f"Visualizing predictions for batch {batch_idx} in phase {phase}")

    # Handle tuple structure if predictions come as (list, dict)
    if isinstance(predictions, tuple):
        predictions = predictions[0]  # Extract the list of predictions from the tuple

    if not predictions:
        print("No predictions found for visualization.")
        return

    # Iterate through each image and its corresponding prediction
    for i, (image, segment, prediction) in enumerate(zip(images, segments, predictions)):
        # Convert tensors to numpy arrays
        image_np = image.permute(1, 2, 0).cpu().numpy()

        # Normalize the image to the 0-255 range if necessary
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        elif image_np.max() > 255.0:
            image_np = np.clip(image_np, 0, 255).astype(np.uint8)

        pred_boxes = prediction['boxes'].cpu().numpy()
        pred_masks = prediction['masks'].cpu().numpy()
        pred_class_ids = prediction['labels'].cpu().numpy()
        pred_scores = prediction['scores'].cpu().numpy()

        ## Squeeze and transpose masks to ensure correct shape
        pred_masks = pred_masks.squeeze(1)  # Change shape to (47, 28, 28)
        pred_masks_resized = np.zeros((image_np.shape[0], image_np.shape[1], pred_masks.shape[0]), dtype=np.float32)

        # Resize each mask to the image size
        for j in range(pred_masks.shape[0]):
            pred_masks_resized[:, :, j] = resize(pred_masks[j, :, :], (image_np.shape[0], image_np.shape[1]),
                                                 mode='constant', preserve_range=True, anti_aliasing=False)

        assert pred_boxes.shape[0] == pred_masks_resized.shape[-1] == pred_class_ids.shape[0]

        # Prepare to draw keypoints
        if keypoints is not None:
            kp = keypoints[i].cpu().numpy()
            # Extract the x, y coordinates and visibility directly
            kp_x = kp[:, 0]  # x coordinates
            kp_y = kp[:, 1]  # y coordinates
            kp_vis = kp[:, 2]  # visibility (likelihood)

            # Combine the coordinates and visibility into a single array
            kp_coords = np.stack([kp_x, kp_y, kp_vis], axis=-1)
        else:
            kp_coords = None

        # Create dir for saving
        os.makedirs(f"{save_img_dir}/{phase}/epoch{epoch}", exist_ok=True)
        save_plot_dir = os.path.join(f"{save_img_dir}/{phase}/epoch{epoch}/batch_{batch_idx}_image_{i}.png")

        # Display the image with predictions using Matterport's display_instances
        fig, ax = plt.subplots(1, figsize=(16, 16))
        display_instances(
            image=image_np,
            boxes=pred_boxes,
            masks=pred_masks_resized,
            class_ids=pred_class_ids,
            class_names=class_names,
            keypoints=kp_coords,
            scores=pred_scores,
            title=f"Epoch: {epoch}, Batch: {batch_idx}, Image: {i}",
            ax=ax,
            show_mask=True,  # Ensure masks are shown
            show_bbox=True,  # Ensure boxes are shown
            save_to_file=save_plot_dir
        )


class AttentionLayer(nn.Module):
    def __init__(self, image_feature_dim, keypoint_dim):
        super(AttentionLayer, self).__init__()
        self.image_fc = nn.Linear(image_feature_dim, image_feature_dim)
        self.keypoint_proj = nn.Conv2d(keypoint_dim, keypoint_dim, kernel_size=1)  # Project keypoint features
        self.keypoint_fc = nn.Linear(keypoint_dim, image_feature_dim)
        self.attention_fc = nn.Linear(image_feature_dim, 1)

    def forward(self, image_features, keypoint_features):
        batch_size = image_features.size(0)
        num_channels = image_features.size(1)
        H = image_features.size(2)
        W = image_features.size(3)
        spatial_dim = H * W

        # Generate a heatmap from keypoints
        keypoint_heatmap = keypoints_to_heatmap(keypoint_features, H, W)

        # Now you can interpolate the heatmap to match the image features, if necessary
        keypoint_heatmap = F.interpolate(keypoint_heatmap, size=(H, W), mode='bilinear', align_corners=False)

        # Project keypoint features to keypoint dimensions
        keypoint_features = self.keypoint_proj(keypoint_heatmap)

        # Reshape keypoint_features to [batch_size, 19, H, W]
        actual_keypoint_size = keypoint_features.numel()
        expected_size = batch_size * 19 * H * W

        if actual_keypoint_size != expected_size:
            raise ValueError(f"Incompatible keypoint feature size: {actual_keypoint_size} vs. expected {expected_size}")

        # Flatten the image features along the spatial dimensions
        image_features = image_features.view(batch_size, num_channels, -1)  # [batch_size, num_channels, H*W]
        image_features = image_features.permute(0, 2, 1).contiguous()  # [batch_size, H*W, num_channels]

        # Flatten the keypoint features across spatial dimensions to match the flattened image features
        keypoint_features = keypoint_features.view(batch_size, -1, spatial_dim).permute(0, 2,
                                                                                        1).contiguous()  # [batch_size, H*W, 27]
        # Apply linear layers
        img_proj = self.image_fc(image_features)  # [batch_size, H*W, image_feature_dim]
        kp_proj = self.keypoint_fc(keypoint_features)  # [batch_size, H*W, image_feature_dim]

        # Debugging: Check for NaN/Inf after linear projections
        if torch.isnan(img_proj).any():
            print("NaN detected in img_proj after image_fc")
        if torch.isinf(img_proj).any():
            print("Inf detected in img_proj after image_fc")
        if torch.isnan(kp_proj).any():
            print("NaN detected in kp_proj after keypoint_fc")
        if torch.isinf(kp_proj).any():
            print("Inf detected in kp_proj after keypoint_fc")

        # Combine features
        combined = torch.tanh(img_proj + kp_proj)

        # Compute attention scores
        attention_scores = torch.sigmoid(self.attention_fc(combined))  # [batch_size, H*W, 1]

        # Debugging: Check for NaN/Inf in attention_scores
        if torch.isnan(attention_scores).any():
            print("NaN detected in attention_scores")
        if torch.isinf(attention_scores).any():
            print("Inf detected in attention_scores")

        # Apply attention to the image features
        attended_features = image_features * attention_scores

        # Reshape back to original dimensions
        attended_features = attended_features.permute(0, 2, 1).contiguous()  # [batch_size, num_channels, H*W]
        attended_features = attended_features.view(batch_size, num_channels, H, W)  # [batch_size, num_channels, H, W]

        # Debugging: Final check before returning
        if torch.isnan(attended_features).any():
            print("NaN detected in attended_features")
        if torch.isinf(attended_features).any():
            print("Inf detected in attended_features")

        return attended_features


class CustomResNetBackbone(nn.Module):
    def __init__(self, feature_dim=2048, keypoint_dim=19, pretrained=True):
        super(CustomResNetBackbone, self).__init__()
        resnet = resnet50(weights="IMAGENET1K_V1")

        # Extract layers from the ResNet
        self.layer1 = nn.Sequential(*list(resnet.children())[:5])  # [B, 256, H1, W1]
        self.layer2 = nn.Sequential(*list(resnet.children())[5])  # [B, 512, H2, W2]
        self.layer3 = nn.Sequential(*list(resnet.children())[6])  # [B, 1024, H3, W3]
        self.layer4 = nn.Sequential(*list(resnet.children())[7])  # [B, 2048, H4, W4]

        # Attention layer to combine image and keypoint features
        self.attention_layer = AttentionLayer(feature_dim, keypoint_dim)

        # Reduce the number of channels in each feature map to match the RPN's expectation (e.g., 256 channels)
        self.conv1x1_layer2 = nn.Conv2d(512, 256, kernel_size=1)  # Reduce layer2 to 256 channels
        self.conv1x1_layer3 = nn.Conv2d(1024, 256, kernel_size=1)  # Reduce layer3 to 256 channels
        self.conv1x1_layer4 = nn.Conv2d(2048, 256, kernel_size=1)  # Reduce layer4 to 256 channels

        # Number of output channels for the last layer
        self.out_channels = 256  # Must match the output of the final layer

    def forward(self, x, keypoints):
        # Ensure `x` is a tensor, not a list
        if isinstance(x, list):
            x = torch.stack(x, dim=0)  # Convert list to tensor
        # If there is an unexpected dimension, squeeze it out
        if x.dim() == 5:
            x = torch.squeeze(x, dim=1)  # Remove the extra dimension

        c1 = self.layer1(x)  # [B, 256, H1, W1]
        c2 = self.layer2(c1)  # [B, 512, H2, W2]
        c3 = self.layer3(c2)  # [B, 1024, H3, W3]
        c4 = self.layer4(c3)  # [B, 2048, H4, W4]

        # Debugging: Check for NaN/Inf in feature maps after each layer
        if torch.isnan(c1).any():
            print("NaN detected in c1 after layer1")
        if torch.isinf(c1).any():
            print("Inf detected in c1 after layer1")
        if torch.isnan(c2).any():
            print("NaN detected in c2 after layer2")
        if torch.isinf(c2).any():
            print("Inf detected in c2 after layer2")
        if torch.isnan(c3).any():
            print("NaN detected in c3 after layer3")
        if torch.isinf(c3).any():
            print("Inf detected in c3 after layer3")
        if torch.isnan(c4).any():
            print("NaN detected in c4 after layer4")
        if torch.isinf(c4).any():
            print("Inf detected in c4 after layer4")

        # Apply attention mechanism on the last layer's output
        c4 = self.attention_layer(c4, keypoints)  # [B, 2048, H4, W4]

        # Reduce the number of channels to 256 for each feature map
        c2 = self.conv1x1_layer2(c2)  # [B, 256, H2, W2]
        c3 = self.conv1x1_layer3(c3)  # [B, 256, H3, W3]
        c4 = self.conv1x1_layer4(c4)  # [B, 256, H4, W4]

        # Debugging: Final check before returning
        if torch.isnan(c2).any():
            print("NaN detected in c2 after conv1x1_layer2")
        if torch.isinf(c2).any():
            print("Inf detected in c2 after conv1x1_layer2")
        if torch.isnan(c3).any():
            print("NaN detected in c3 after conv1x1_layer3")
        if torch.isinf(c3).any():
            print("Inf detected in c3 after conv1x1_layer3")
        if torch.isnan(c4).any():
            print("NaN detected in c4 after conv1x1_layer4")
        if torch.isinf(c4).any():
            print("Inf detected in c4 after conv1x1_layer4")

        return {'0': c2, '1': c3, '2': c4}  # Return a dictionary of feature maps with consistent channels


class CustomRPNHead(RPNHead):
    def __init__(self, in_channels, num_anchors):
        super(CustomRPNHead, self).__init__(in_channels, num_anchors)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1)

    def forward(self, x):
        logits = []
        bbox_reg = []
        epsilon = 1e-6  # Small epsilon to prevent numerical instability

        for feature in x:
            if torch.isnan(feature).any():
                print("NaN detected in feature maps before RPN")
            if torch.isinf(feature).any():
                print("Infinite values detected in feature maps before RPN")

            t = self.conv(feature)  # Apply conv layer
            logits_t = self.cls_logits(t)
            logits_t = logits_t.clamp(min=epsilon, max=1 - epsilon)  # Prevent NaN in logits

            logits.append(logits_t)  # Apply classification head
            bbox_reg.append(self.bbox_pred(t))  # Apply box regression head

        # print(f"Objectness logits before loss: {logits}")

        return logits, bbox_reg


class SkeletonGuidedMaskRCNN(MaskRCNN):
    def __init__(self, backbone, num_classes):
        super(SkeletonGuidedMaskRCNN, self).__init__(backbone, num_classes=num_classes)

        # Replace the RPN head with the custom RPN head
        self.rpn.head = CustomRPNHead(in_channels=backbone.out_channels,
                                      num_anchors=self.rpn.anchor_generator.num_anchors_per_location()[0])

    def forward(self, images, keypoints, targets=None):
        features = self.backbone(images, keypoints)
        if not isinstance(features, dict):
            # Wrap features in a dict with appropriate keys
            features = {'0': features}

        # Debug
        for feature in features.values():
            if torch.isnan(feature).any():
                print("NaN detected in feature maps before RPN")
            if torch.isinf(feature).any():
                print("Infinite values detected in feature maps before RPN")

        # Convert images to ImageList
        image_shapes = [img.shape[-2:] for img in images]
        image_list = ImageList(images, image_shapes)

        if self.training:
            # Generate proposals and image shapes
            proposals, proposal_losses = self.rpn(image_list, features, targets)
            result, detector_losses = self.roi_heads(features, proposals, image_shapes, targets)

            # Combine losses
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses
        else:
            # For inference, generate proposals without targets
            proposals, _ = self.rpn(image_list, features, targets=None)
            predictions = self.roi_heads(features, proposals, image_shapes)
            return predictions


# CREATE THE MODEL
# Initialize the custom backbone
backbone = CustomResNetBackbone(pretrained=ResNet50_Weights.DEFAULT)

# Initialize custom model
model = SkeletonGuidedMaskRCNN(
    backbone=backbone,
    num_classes=num_classes,
)

num_anchors = anchor_generator.num_anchors_per_location()[0]
rpn_head = CustomRPNHead(in_channels=backbone.out_channels, num_anchors=num_anchors)

# Integrate the RPN head and anchor generator into the RPN using the updated API
model.rpn = RegionProposalNetwork(
    anchor_generator=anchor_generator,
    head=rpn_head,
    fg_iou_thresh=0.7,
    bg_iou_thresh=0.3,
    batch_size_per_image=256,
    positive_fraction=0.5,
    pre_nms_top_n={'training': 2000, 'testing': 1000},
    post_nms_top_n={'training': 2000, 'testing': 1000},
    nms_thresh=0.7
)

# Replace model.roi_heads with CustomRoIHeads
model.roi_heads = RoIHeads(
    box_roi_pool=model.roi_heads.box_roi_pool,
    box_head=model.roi_heads.box_head,
    box_predictor=model.roi_heads.box_predictor,
    mask_roi_pool=model.roi_heads.mask_roi_pool,
    mask_head=model.roi_heads.mask_head,
    mask_predictor=model.roi_heads.mask_predictor,
    fg_iou_thresh=0.5,  # foreground IoU threshold
    bg_iou_thresh=0.5,  # background IoU threshold
    batch_size_per_image=512,  # typically used batch size per image
    positive_fraction=0.25,
    bbox_reg_weights=None,  # or some weights if needed
    score_thresh=0.05,  # threshold to filter detections
    nms_thresh=0.5,  # non-maximum suppression threshold
    detections_per_img=100  # max number of detections per image
)

# Move the model to the appropriate device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# Optimizer and learning rate scheduler
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Prepare DataLoaders for training and validation
train_loader, val_loader = prepare_dataloader(csv_directory, segments_dirs, image_directory,
                                              batch_size, val_split)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_idx, (images, segments, skeletons) in enumerate(train_loader):
        images = [img.to(device) for img in images]
        images = torch.stack(images, dim=0)
        keypoints = skeletons.to(device)

        # Ensure segments remain as lists when passed to the model
        segments = [{k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in segment.items()} for segment in
                    segments]

        # Forward pass with images, keypoints, and segmentation targets
        loss_dict = model(images, keypoints, segments)
        losses = sum(loss for loss in loss_dict.values())

        # Backpropagation
        optimizer.zero_grad()
        losses.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += losses.item()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}')

    # Validation at specified epoch intervals
    if (epoch + 1) % validate_every_n_epochs == 0:
        print(f"Running validation after epoch {epoch + 1}")
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (val_images, val_segments, val_skeletons) in enumerate(val_loader):
                val_images = [img.to(device) for img in val_images]
                val_images = torch.stack(val_images, dim=0)
                val_keypoints = val_skeletons.to(device)

                val_segments = [{k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in segment.items()}
                                for segment in val_segments]

                predictions = model(val_images, val_keypoints)

                # Visualize every batch during validation
                visualize_predictions(val_images, val_segments, predictions, epoch + 1, batch_idx, 'val_predictions',
                                      save_to_dir, class_names, keypoints=val_keypoints)

                val_loss_dict = model(val_images, val_keypoints, val_segments)
                # val_losses = sum(loss for loss in val_loss_dict.values())
                # val_loss += val_losses.item()

        print(f'Validation Loss: {val_loss / len(val_loader)}')

    # Save the model checkpoint after each epoch
    checkpoint_dir = os.path.join(save_to_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch + 1}.pth')
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model saved to {checkpoint_path}")
