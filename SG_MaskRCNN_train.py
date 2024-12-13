import os
import torch
import torch.nn as nn
import torchvision
import json
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from Data_alignment import prepare_dataloader
from utils import keypoints_to_heatmaps, visualize_prediction_images, filter_duplicate_classes

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Ignore

# Define directories
csv_directory = r'F:\DLC_thermography-JLanden-2024-07-09\videos' #Keypoints from DLC
image_directory = r'F:\DLC_thermography-JLanden-2024-07-09\labeled-data' # Images
segments_dirs = r'F:\Segmentation\training_segments' # Training segments from Darwin
save_to_dir = r'F:\Segmentation\results_12-05-24_BS8_solo' #Where output is saved

# Parameters
num_epochs = 500
validate_every_n_epochs = 100
batch_size = 8  # Adjust as needed
val_split = 0.2  # 20% of the data for validation
#mask_threshold = 0.5  # 0.5 is normal
mask_display_threshold = 0.8
save_weights = True

image_size = (348, 464)  # H, W
num_keypoints = 19
num_classes = 4  # Set this according to your dataset (# classes + background)
class_names = ['background', 'implanted_mouse', 'implanted_bat', 'implanted_rump' #,
               # , 'extra_mouse', 'extra_bat', 'extra_rump'
               ]

colors = [(1, 0, 0), (.1, .6, .1), (0, 0, 1)]  #R, G, B: Must match number of class names
# NMS = non-max suppression

# Attention Mechanism
class AttentionLayer(nn.Module):
    def __init__(self, img_channels, keypoint_channels, attention_channels):
        super(AttentionLayer, self).__init__()
        # Convolution layers for image and keypoints (input)
        self.img_conv1 = nn.Conv2d(img_channels, attention_channels, kernel_size=1)
        self.img_conv2 = nn.Conv2d(attention_channels, attention_channels, kernel_size=3, padding=1)
        self.img_conv3 = nn.Conv2d(attention_channels, attention_channels, kernel_size=3, padding=1)

        self.keypoint_conv1 = nn.Conv2d(keypoint_channels, attention_channels, kernel_size=1)
        self.keypoint_conv2 = nn.Conv2d(attention_channels, attention_channels, kernel_size=3, padding=1)
        self.keypoint_conv3 = nn.Conv2d(attention_channels, attention_channels, kernel_size=3, padding=1)

        # Activation function
        self.relu = nn.ReLU()

        # Convolution for the attention mechanism (combine image and keypoint features)
        self.attention_conv1 = nn.Conv2d(attention_channels, attention_channels, kernel_size=3, padding=1)
        self.attention_conv2 = nn.Conv2d(attention_channels, attention_channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, images, keypoints):
        # Process the image through multiple convolutions
        img_out = self.relu(self.img_conv1(images))
        img_out = self.relu(self.img_conv2(img_out))
        img_out = self.relu(self.img_conv3(img_out))

        # Process the keypoints through multiple convolutions
        keypoint_out = self.relu(self.keypoint_conv1(keypoints))
        keypoint_out = self.relu(self.keypoint_conv2(keypoint_out))
        keypoint_out = self.relu(self.keypoint_conv3(keypoint_out))

        # Combine image and keypoint features using element-wise multiplication
        combined = img_out * keypoint_out

        # Apply further attention convolutions
        attention_map = self.relu(self.attention_conv1(combined))
        attention_map = self.sigmoid(self.attention_conv2(attention_map))

        # Apply attention to image features
        enhanced_img_features = images * attention_map
        return enhanced_img_features


# Custom Mask R-CNN Model with Attention
class SkeletonGuidedMaskRCNN(nn.Module):
    def __init__(self, num_classes):
        super(SkeletonGuidedMaskRCNN, self).__init__()

        # Attention layer first to overlay images with keypoints
        self.attention = AttentionLayer(img_channels=3, keypoint_channels=num_keypoints, attention_channels=3)

        # Load a pre-trained Mask R-CNN backbone
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        backbone = nn.Sequential(*list(backbone.children())[:-2])  # Remove last fully connected layer

        self.backbone = backbone
        self.backbone.out_channels = 2048

        # Create an Anchor Generator for the FPN (feature pyramid network)
        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),) * 5)

        # Define RoI Align and FPN
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=4, sampling_ratio=2)

        # Define Mask R-CNN model with the modified backbone
        self.model = MaskRCNN(self.backbone, num_classes=num_classes, rpn_anchor_generator=anchor_generator,
                              box_roi_pool=roi_pooler, mask_roi_pool=roi_pooler)

    def forward(self, images, keypoints, targets=None, return_predictions=False):
        # Apply attention to images using keypoints
        enhanced_img_features = self.attention(images, keypoints)

        # Use enhanced image features in the Mask R-CNN model
        if self.training and not return_predictions:
            return self.model(enhanced_img_features, targets)
        else: # Validate
            return self.model(enhanced_img_features)


# Training script
# Initialize the custom loss
# loss_fn = CustomLoss()

########################
def save_predictions_to_json(predictions, save_to_dir, videoID, frameID):
    os.makedirs(f"{save_to_dir}/predictions/{videoID}", exist_ok=True)
    json_path = os.path.join(save_to_dir, f"predictions", videoID, f"predictions_{frameID}.txt")

    serializable_predictions = []
    for pred in predictions:
        serializable_predictions.append({
            'labels': pred['labels'].cpu().tolist(),  # Convert tensor to list
            'scores': pred['scores'].cpu().tolist(),  # Convert tensor to list
            'boxes': pred['boxes'].cpu().tolist(),  # Convert tensor to list
            'masks': pred['masks'].cpu().numpy().tolist()  # Convert tensor to nested lists
        })

    # Write the predictions to a file in JSON format
    with open(json_path, 'w') as f:
        json.dump(serializable_predictions, f, indent=4)
#####################


def train_model(train_loader, val_loader, num_classes, num_epochs=num_epochs):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = SkeletonGuidedMaskRCNN(num_classes=num_classes).to(device)
    #model = SkeletonGuidedMaskRCNN(num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch_idx, (images, segments, skeletons) in enumerate(train_loader):
            images = [image.to(device) for image in images]
            H, W = images[0].shape[1], images[0].shape[2]
            # Convert images to tensors
            images = torch.stack([image.to(device) for image in images]).to(device)

            keypoints = skeletons.to(device)
            kp_heatmaps = keypoints_to_heatmaps(keypoints, H, W)
            segments = [{k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in segment.items()} for
                        segment in segments]

            optimizer.zero_grad()
            outputs = model(images, kp_heatmaps, segments)

            # Use the losses from the model's output
            rcnn_losses = sum(loss for loss in outputs.values())
            total_loss = rcnn_losses

            total_loss.backward()
            optimizer.step()
            total_loss += total_loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader)}")

        if (epoch + 1) % validate_every_n_epochs == 0:
            model.eval()
            predictions_list = []
            for batch_idx, (images, segments, skeletons) in enumerate(val_loader):
                images = [image.to(device) for image in images]
                H, W = images[0].shape[1], images[0].shape[2]
                images = torch.stack([image.to(device) for image in images]).to(device)

                keypoints = skeletons.to(device)
                kp_heatmaps = keypoints_to_heatmaps(keypoints, H, W)
                segments = [{k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in segment.items()} for
                            segment in segments]

                with torch.no_grad():  # Disable gradient calculation
                    # Get predictions instead of losses
                    predictions = model(images, kp_heatmaps)
                    predictions = filter_duplicate_classes(predictions)
                    predictions_list.append(predictions)

                # Optionally visualize here
                visualize_prediction_images(
                    images=images,
                    predictions=predictions,
                    epoch=epoch + 1,
                    phase="validate",
                    batch_idx=batch_idx,
                    save_img_dir=save_to_dir,
                    class_names=class_names,
                    colors=colors,
                    keypoints=keypoints,
                    threshold=mask_display_threshold
                )

            else:
                continue

    # Save model weights
    if save_weights is True:
        model_save_path = os.path.join(save_to_dir, f'sg-mrcnn_epochs_{num_epochs}.pth')
        torch.save(model.state_dict(), model_save_path)
        print(f"Model weights saved to {model_save_path}")

    return model


if __name__ == "__main__":
    # Assuming train_loader and val_loader are already created with your data
    train_loader, val_loader = prepare_dataloader(csv_directory, segments_dirs, image_directory,
                                                  batch_size, val_split)  # Implement this function

    trained_model = train_model(train_loader, val_loader, num_classes)
