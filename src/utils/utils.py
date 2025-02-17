import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.transform import resize
from visualize_copy import display_instances
import imageio.v3 as iio

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
def visualize_prediction_images(images, predictions, epoch, batch_idx, phase, save_img_dir, class_names,
                                colors, keypoints=None, threshold=0.8):
    # Handle tuple structure if predictions come as (list, dict)
    if isinstance(predictions, tuple):
        predictions = predictions[0]  # Extract the list of predictions from the tuple

    if not predictions:
        print("No predictions found for visualization.")
        return

    # Iterate through each image and its corresponding prediction
    for i, (image, prediction) in enumerate(zip(images, predictions)):
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
        if phase == "validate":
            os.makedirs(f"{save_img_dir}/{phase}/epoch{epoch}", exist_ok=True)
            save_plot_dir = os.path.join(f"{save_img_dir}/{phase}/epoch{epoch}/batch_{batch_idx}_image_{i}.png")
        elif phase == "analysis":
            os.makedirs(f"{save_img_dir}/{phase}/{epoch}", exist_ok=True)
            save_plot_dir = os.path.join(f"{save_img_dir}/{phase}/{epoch}/{batch_idx}.png")
        else:
            print("Variable phase invalid. Should be either 'validate' or 'analysis'.")

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
            ax=ax,
            show_mask=True,  # Ensure masks are shown
            show_bbox=True,  # Ensure boxes are shown
            save_to_file=save_plot_dir,
            colors=colors,
            threshold=threshold
        )


############################################################
## Combine temperature data (from tiff files) with predictions from MRCNN

# Convert tiff temperature file to usable temperatures
def tiff2temp(array_path):
    array = iio.imread(array_path)
    array = array.astype(np.float64)

    # Convert from raw to kelvin
    array *= .04
    # Convert from kelvin to Celsius
    array += -273.15

    # Round to three digits
    array = np.round(array, decimals=4)

    # Save array as .txt file
    return array


# Extract temperatures from prediction coordinates
def save_temperature_stats(images, predictions, temperature_array, frameID, threshold, class_names):
    """
    Save temperature statistics (mean and SD) for each bounding box into a CSV file.

    Args:
        predictions (list): List of predictions containing 'boxes' and 'labels'.
        temperature_data (numpy.ndarray): Temperature data (H x W) matching the image dimensions.
    """
    results = []

    # Iterate through each image and its corresponding prediction
    for i, (image, prediction) in enumerate(zip(images, predictions)):
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

        ## Squeeze and transpose masks to ensure correct shape
        pred_masks = pred_masks.squeeze(1)  # Change shape to (47, 28, 28)
        pred_masks_resized = np.zeros((image_np.shape[0], image_np.shape[1], pred_masks.shape[0]), dtype=np.float32)

        # Resize each mask to the image size
        for j in range(pred_masks.shape[0]):
            pred_masks_resized[:, :, j] = resize(pred_masks[j, :, :], (image_np.shape[0], image_np.shape[1]),
                                                 mode='constant', preserve_range=True, anti_aliasing=False)

        assert pred_boxes.shape[0] == pred_masks_resized.shape[-1] == pred_class_ids.shape[0]

        # Number of instances
        N = pred_boxes.shape[0]
        if not N:
            print("\n*** No boxes to display *** \n")
        else:
            assert pred_boxes.shape[0] == pred_masks_resized.shape[-1] == pred_class_ids.shape[0]

        for i in range(N):
            class_id = pred_class_ids[i]
            label = class_names[class_id]

            # Bounding box
            if not np.any(pred_boxes[i]):
                # Skip this instance. Has no bbox.
                continue
            x_min, y_min, x_max, y_max = np.round(pred_boxes[i]).astype(int)

            # Crop the temperature region corresponding to the bounding box
            cropped_temp = temperature_array[y_min:y_max, x_min:x_max]

            if cropped_temp.size > 0:  # Ensure the cropped region is not empty
                mean_box_temp = np.mean(cropped_temp)
                std_box_temp = np.std(cropped_temp)
                min_box_temp = np.min(cropped_temp)
                max_box_temp = np.max(cropped_temp)
            else:
                mean_box_temp = np.nan
                std_box_temp = np.nan
                min_box_temp = np.nan
                max_box_temp = np.nan

            # Mask
            mask = pred_masks_resized[:, :, i]
            mask_temp = temperature_array[mask > threshold]

            if mask_temp.size > 0:  # Ensure the mask region is not empty
                mean_mask_temp = np.mean(mask_temp)
                std_mask_temp = np.std(mask_temp)
                min_mask_temp = np.min(mask_temp)
                max_mask_temp = np.max(mask_temp)
            else:
                print(f"Mask region empty for frame {frameID}")
                mean_mask_temp = np.nan
                std_mask_temp = np.nan
                min_mask_temp = np.nan
                max_mask_temp = np.nan

            # Append results
            results.append({
                'Frame': frameID,
                'Label': label,
                'Mean_box': mean_box_temp,
                'SD_box': std_box_temp,
                'Min_box': min_box_temp,
                'Max_box': max_box_temp,
                'Mean_mask': mean_mask_temp,
                'SD_mask': std_mask_temp,
                'Min_mask': min_mask_temp,
                'Max_mask': max_mask_temp
            })
        return results
