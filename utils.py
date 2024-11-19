import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.transform import resize
from visualize_copy import display_instances

########### Custom loss function to better incorporate keypoint info
class CustomLoss(nn.Module):
    def __init__(self, lambda_coord=1.0):
        super(CustomLoss, self).__init__()
        self.lambda_coord = lambda_coord
        self.l1_loss = nn.L1Loss()

    def forward(self, outputs, targets):
        # Extract predicted boxes and keypoints from outputs and targets
        pred_boxes = [output['boxes'] for output in outputs]
        true_keypoints = [target['keypoints'] for target in targets]

        # Calculate the center of the predicted boxes
        pred_centers = [(boxes[:, :2] + boxes[:, 2:]) / 2 for boxes in pred_boxes]

        # Calculate the L1 loss between predicted centers and true keypoints
        coord_loss = sum(self.l1_loss(pred_center, keypoints) for pred_center, keypoints in zip(pred_centers, true_keypoints))

        # Combine with other losses (e.g., classification, mask loss)
        total_loss = self.lambda_coord * coord_loss  # Add other losses as needed

        return total_loss

    def compute_keypoint_bbox_penalty(self, bboxes, keypoints):
        """
        Compute a penalty for keypoints that fall outside their respective bounding boxes.

        Parameters:
        - bboxes: Predicted bounding boxes (N x 4 tensor)
        - keypoints: Predicted keypoints (N x K x 2 tensor, where K is the number of keypoints)

        Returns:
        - Total keypoint-bounding box penalty
        """

        penalty = 0.0

        for i in range(len(bboxes)):
            bbox = bboxes[i]
            kpts = keypoints[i]

            # Extract the bounding box coordinates (x_min, y_min, x_max, y_max)
            x_min, y_min, x_max, y_max = bbox

            # Loop over each keypoint and check if it falls within the bounding box
            for kp in kpts:
                kp_x, kp_y = kp

                if kp_x < x_min:  # Keypoint is left of the box
                    penalty += (x_min - kp_x) ** 2
                if kp_x > x_max:  # Keypoint is right of the box
                    penalty += (kp_x - x_max) ** 2
                if kp_y < y_min:  # Keypoint is above the box
                    penalty += (y_min - kp_y) ** 2
                if kp_y > y_max:  # Keypoint is below the box
                    penalty += (kp_y - y_max) ** 2

        return penalty

########### Convert keypoints to heat maps
def gaussian_kernel(size, sigma):
    """Creates a 2D Gaussian kernel."""
    x = torch.arange(0, size, 1, dtype=torch.float32)
    y = torch.arange(0, size, 1, dtype=torch.float32)
    x_grid, y_grid = torch.meshgrid(x, y, indexing='ij')

    center = (size - 1) / 2
    kernel = torch.exp(-((x_grid - center) ** 2 + (y_grid - center) ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    return kernel


def keypoints_to_heatmaps(keypoints, H, W, kernel_size=7, sigma=2):
    """
    :param keypoints: Keypoints coming from dataloader, as a tensor
    :param H: Height of image
    :param W: Width of image
    :param kernel_size: Size of the Gaussian kernel
    :param sigma: Standard deviation of the Gaussian kernel
    :return: Smoothed heatmap for each body part, for every individual visible in the image
    """
    batch_size, num_keypoints, _ = keypoints.size()
    # Create a zeroed heatmap with dimensions [batch_size, num_keypoints, H, W]
    heatmap = torch.zeros((batch_size, num_keypoints, H, W), device=keypoints.device)

    # Create a Gaussian kernel
    kernel = gaussian_kernel(kernel_size, sigma).to(keypoints.device)
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, kernel_size, kernel_size]

    for i in range(batch_size):
        for j in range(num_keypoints):
            likelihood = keypoints[i, j, 2]
            if likelihood > 0:
                # Normalize keypoint coordinates to the heatmap dimensions
                x = int((keypoints[i, j, 0] * W).clamp(0, W - 1))
                y = int((keypoints[i, j, 1] * H).clamp(0, H - 1))
                # Set the heatmap pixel at (y, x) equal to likelihood that the kp is correct
                heatmap[i, j, y, x] = likelihood

                # Apply Gaussian smoothing
                heatmap[i, j] = F.conv2d(heatmap[i, j].unsqueeze(0).unsqueeze(0), kernel,
                                         padding=kernel_size // 2).squeeze()

    return heatmap

########### Filter each class by highest confidence so we only get one output for each class
def filter_duplicate_classes(predictions):
    """
    Filters predictions to retain only the highest confidence prediction per class in each image.
    Args:
        predictions (list of dict): List of predictions, each containing 'boxes', 'labels', 'scores', and 'masks'.
    Returns:
        filtered_predictions (list of dict): Filtered predictions with unique classes.
    """
    filtered_predictions = []

    for pred in predictions:
        unique_class_predictions = {}
        for idx, label in enumerate(pred['labels']):
            score = pred['scores'][idx]

            # Keep the prediction with the highest score for each class label
            if label.item() not in unique_class_predictions or score > unique_class_predictions[label.item()]['score']:
                unique_class_predictions[label.item()] = {
                    'box': pred['boxes'][idx],
                    'mask': pred['masks'][idx],
                    'score': score
                }

        # If unique_class_predictions is empty, skip stacking or handle accordingly
        if unique_class_predictions:
            filtered_predictions.append({
                'boxes': torch.stack([data['box'] for data in unique_class_predictions.values()]),
                'masks': torch.stack([data['mask'] for data in unique_class_predictions.values()]),
                'labels': torch.tensor(list(unique_class_predictions.keys()), device=pred['labels'].device),
                'scores': torch.tensor([data['score'] for data in unique_class_predictions.values()],
                                       device=pred['scores'].device)
            })
        else:
            # Add an empty prediction with the correct structure if needed
            filtered_predictions.append({
                'boxes': torch.empty((0, 4), device=pred['boxes'].device, dtype=pred['boxes'].dtype),
                'masks': torch.empty((0, *pred['masks'].shape[1:]), device=pred['masks'].device,
                                     dtype=pred['masks'].dtype),
                'labels': torch.empty((0,), device=pred['labels'].device, dtype=torch.int64),
                'scores': torch.empty((0,), device=pred['scores'].device, dtype=pred['scores'].dtype)
            })

    return filtered_predictions


################## Visualize final output
def visualize_prediction_images(images, segments, predictions, epoch, batch_idx, phase, save_img_dir, class_names,
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
        print(f"Boxes for image {i}: {pred_boxes}")
        pred_masks = prediction['masks'].cpu().numpy()
        pred_class_ids = prediction['labels'].cpu().numpy()
        print(f"Classes for image {i}: {pred_class_ids}")
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
