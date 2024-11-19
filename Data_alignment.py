import json
import pandas as pd
import os
import glob

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from torchvision import transforms
import hashlib
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap

name_mapping = {
    "Implanted_mouse": 1,
    "Implanted_BAT": 2,
    "Implanted_rump": 3 # ,
    # "Extra_mouse": 4,
    # "Extra_BAT": 5,
    # "Extra_rump": 6,
}

# Function to apply the mapping
def replace_names(name, name_mapping):
    # If the name exists in the mapping, replace it
    if name in name_mapping:
        return name_mapping[name]
    else:
        print(f"Warning: '{name}' not found in name_mapping.")

def prepare_dataloader(csv_directory, segments_dirs, image_directory, batch_size=4, val_split=0.2):
    """
    Prepare DataLoaders for training and validation by aligning segmentation and skeleton data, and loading images.

    Args:
        csv_directory (str): Directory containing CSV files with skeleton data.
        segments_dirs (str): Directory containing subdirectories of JSON files with segmentation data.
        image_directory (str): Directory containing images corresponding to the frames.
        batch_size (int): Number of samples per batch.
        val_split (float): Proportion of data to be used for validation.

    Returns:
        DataLoader, DataLoader: DataLoaders for the training and validation datasets.
    """

    def custom_collate(batch):
        # Assuming each item in the batch is a tuple (concatenated_image, target, skeleton_data)
        images = [item[0] for item in batch]  # List of concatenated tensors
        segments = [item[1] for item in batch]
        skeletons = [item[2] for item in batch]

        # Stack images into a single batch tensor
        batched_images = torch.stack(images, dim=0)  # Shape: [batch_size, channels, height, width]

        # Process segmentation data
        batched_segments = [
            {
                'boxes': seg['boxes'],  # Leave as a list of tensors
                'masks': seg['masks'],  # Leave as a list of tensors
                'labels': seg['labels']  # Leave as a list of tensors
            } for seg in segments
        ]

        # Stack skeleton keypoints into a single batch tensor
        batched_skeletons = torch.stack(skeletons, dim=0)  # Shape: [batch_size, 3, H, W]

        return batched_images, batched_segments, batched_skeletons

    def load_skeleton_data(csv_directory, video_file_name):
        # Identify the csv file we are looking for
        pattern = os.path.join(csv_directory, f"{video_file_name}*.csv")
        matching_files = glob.glob(pattern)

        # Load the CSV file
        keypoints_data = pd.read_csv(matching_files[0], header=None, skiprows=4)

        # Extract the first three rows as metadata
        individuals = pd.read_csv(matching_files[0], header=None, skiprows=1, nrows=1).iloc[0, 1:].values
        body_parts = pd.read_csv(matching_files[0], header=None, skiprows=2, nrows=1).iloc[0, 1:].values
        coords_types = pd.read_csv(matching_files[0], header=None, skiprows=3, nrows=1).iloc[0, 1:].values

        # Create a dictionary to hold the coordinates and likelihoods for each body part
        keypoints_dict = {}

        # Iterate through the individuals, body parts, and coordinate types
        for i, (individual, part, coord_type) in enumerate(zip(individuals, body_parts, coords_types)):
            if individual not in keypoints_dict:
                keypoints_dict[individual] = {}
            if part not in keypoints_dict[individual]:
                keypoints_dict[individual][part] = {"x": [], "y": [], "likelihood": []}

            # Populate the dictionary with the appropriate data
            keypoints_dict[individual][part][coord_type] = pd.to_numeric(keypoints_data.iloc[:, i + 1],
                                                                         errors='coerce').values

        return keypoints_dict

    def load_segmentation_data(json_dir):
        segmentation_data = {}
        for json_file in os.listdir(json_dir):
            if json_file.endswith('.json'):
                with open(os.path.join(json_dir, json_file), 'r') as f:
                    data = json.load(f)
                    segmentation_data[json_file] = data
        return segmentation_data

    def _create_mask_from_polygon(polygon, image_size):
        """
        Create a binary mask from polygon coordinates.

        Args:
            polygon (dict): A dictionary containing polygon paths.
            image_size (tuple): The size of the image (width, height).

        Returns:
            torch.Tensor: A binary mask where the polygon area is filled.
        """
        # Ensure image_size is a tuple with (width, height)
        if isinstance(image_size, tuple) and len(image_size) == 2:
            mask = Image.new('L', image_size, 0)  # Create a blank image with background 0
        else:
            raise ValueError("Image size must be a tuple containing width and height.")

        for path in polygon['paths']:
            coords = [(p['x'], p['y']) for p in path]
            ImageDraw.Draw(mask).polygon(coords, outline=1, fill=1)  # Fill the polygon with 1

        mask = np.array(mask)  # Convert to numpy array
        return torch.tensor(mask, dtype=torch.uint8)  # Convert to PyTorch tensor

    def summarize_mask(mask_tensor):
        """Summarize mask by returning its shape and a hash."""
        mask_hash = hashlib.md5(mask_tensor.numpy()).hexdigest()
        return {
            "shape": mask_tensor.shape,
            "hash": mask_hash
        }

    def extract_segmentation_data(segmentation_json):
        # Extract the correct image name from the 'source_files' key
        image_name = segmentation_json['item']['slots'][0]['source_files'][0]['file_name']
        annotations = segmentation_json['annotations']

        # Extract annotations
        unsummarized_data = []
        summarized_data = []
        for annotation in annotations:
            if 'bounding_box' in annotation and 'polygon' in annotation:
                name = annotation['name']
                # Apply the mapping
                class_id = replace_names(name, name_mapping)
                bounding_box = annotation['bounding_box']

                # Convert bounding box format from {"h": h, "w": w, "x": x, "y": y} to [x_min, y_min, x_max, y_max]
                x_min = bounding_box['x']
                y_min = bounding_box['y']
                x_max = x_min + bounding_box['w']
                y_max = y_min + bounding_box['h']
                converted_bounding_box = [x_min, y_min, x_max, y_max]

                # Generate mask from polygon and summarize it
                image_size = (
                segmentation_json['item']['slots'][0]['width'], segmentation_json['item']['slots'][0]['height'])
                mask = _create_mask_from_polygon(annotation['polygon'], image_size)
                mask_summary = summarize_mask(mask)

                # Store both summarized and full mask data
                summarized_data.append({
                    "name": name,
                    "bounding_box": converted_bounding_box,
                    "mask_summary": mask_summary
                })

                unsummarized_data.append({
                    "labels": class_id,
                    "bounding_box": converted_bounding_box,
                    "mask": mask  # Store the full mask tensor here
                })

        return image_name, summarized_data, unsummarized_data

    def align_data(segmentation_data, skeleton_data, imageName):
        aligned_segmentation = {}
        aligned_skeleton = {}
        ########################
        output_file = f'F:\\Segmentation\\results\\align_check\\aligned_data_{imageName}.txt'
        ########################
        # Open the file in write mode once at the beginning to clear it
        with open(output_file, 'w') as file:
            file.write("")

        for json_file, seg_data in segmentation_data.items():
            image_name, summarized_segmentation_data, unsummarized_segmentation_data = extract_segmentation_data(
                seg_data)
            frame_index = int(image_name.split('.')[0].replace('img', ''))

            # Add segmentation data
            aligned_segmentation[image_name] = unsummarized_segmentation_data

            # Add skeleton data
            skeleton_frame_data = {}
            for individual, parts in skeleton_data.items():
                skeleton_frame_data[individual] = {}
                for part, coords in parts.items():
                    skeleton_frame_data[individual][part] = {
                        "x": coords["x"][frame_index] if len(coords["x"]) > frame_index else None,
                        "y": coords["y"][frame_index] if len(coords["y"]) > frame_index else None,
                        "likelihood": coords["likelihood"][frame_index] if len(
                            coords["likelihood"]) > frame_index else None
                    }
            aligned_skeleton[image_name] = skeleton_frame_data

            # Write the summarized data to the .txt file for verification
            with open(output_file, 'a') as file:
                if image_name in aligned_segmentation:
                    file.write(f"Image: {image_name}\n")
                    file.write("Segmentation Data:\n")
                    for seg in summarized_segmentation_data:  # Use summarized data for writing
                        file.write(
                            f"name: {seg['name']}, bounding_box: {seg['bounding_box']}, mask_summary: {seg['mask_summary']}\n")
                    file.write("Skeleton Data:\n")
                    if image_name in aligned_skeleton:
                        for individual, parts in aligned_skeleton[image_name].items():
                            file.write(f"Individual: {individual}\n")
                            for part, coords in parts.items():
                                file.write(
                                    f"{part}: x = {coords['x']}, y = {coords['y']}, likelihood = {coords['likelihood']}\n")
                    file.write("-" * 50 + "\n")

        return aligned_segmentation, aligned_skeleton

    class MouseDataset(Dataset):
        def __init__(self, segment_data, skeleton_data, images_dir, transforms=None):
            self.original_image_width = 464
            self.original_image_height = 348
            self.segment_data = segment_data
            self.skeleton_data = skeleton_data
            self.images_dir = images_dir
            self.transforms = transforms

            # Ensure both segmentation and skeleton data have the same keys
            self.common_keys = list(set(self.segment_data.keys()) & set(self.skeleton_data.keys()))
            if not self.common_keys:
                print("No common keys between segmentation and skeleton data.")


            # Filter image files based on common keys
            self.image_files = [
                f for f in os.listdir(images_dir)
                if f.endswith('.png') and f in self.common_keys
            ]

            # Determine the number of keypoints dynamically
            self.num_keypoints = self._get_num_keypoints()

        def _get_num_keypoints(self):
            """Determine the number of keypoints from the skeleton data."""
            num_keypoints = 0
            for key in self.common_keys:
                skeleton_data = self.skeleton_data.get(key, {})
                for individual, parts in skeleton_data.items():
                    num_keypoints = max(num_keypoints, len(parts))
            return num_keypoints

        def __len__(self):
            return len(self.image_files)

        def __getitem__(self, idx):
            img_name = self.image_files[idx]
            img_path = os.path.join(self.images_dir, img_name)
            image = Image.open(img_path).convert("RGB")
            image = transforms.ToTensor()(image)

            if self.transforms:
                image = self.transforms(image)

            # Add a batch dimension to image
            if image.dim() == 3:  # Check if image is 3D
                image = image.unsqueeze(0)  # Add batch dimension: shape (1, C, H, W)

            # Segmentation
            segmentation = self.segment_data.get(img_name, [])
            if segmentation is None:
                raise ValueError(f"Missing segmentation data for image {img_name}")

            # Extract bounding boxes and masks
            boxes = torch.tensor([seg['bounding_box'] for seg in segmentation], dtype=torch.float32)
            masks = []  # Initialize an empty list for masks
            labels = []

            for seg in segmentation:
                if 'mask' in seg:  # Ensure that 'mask' exists and is a proper binary mask
                    if isinstance(seg['mask'], torch.Tensor):
                        mask = seg['mask'].clone().detach().float()
                    else:
                        mask = torch.tensor(seg['mask'], dtype=torch.float32)
                    masks.append(mask)
                else:
                    # Placeholder for missing mask data or incorrect mask loading
                    mask_shape = (1, *image.shape[1:])
                    mask = torch.zeros(mask_shape, dtype=torch.float32)
                    masks.append(mask)

                labels.append(seg['labels'])

            # Stack the masks into a tensor
            masks = torch.stack(masks) if masks else torch.zeros((0, *image.shape[1:]), dtype=torch.float32)
            # Stack labels into a tensor but leave possibility of empty image
            if labels:
                labels = torch.stack([torch.tensor(label, dtype=torch.int64) for label in labels])
            else:
                # Handle case where labels are empty, for instance by creating an empty tensor
                labels = torch.empty(0, dtype=torch.int64)

            target = {
                'boxes': boxes,
                'masks': masks,
                'labels': labels
            }

            # Check if the number of boxes matches the number of masks
            if boxes.shape[0] != masks.shape[0]:
                print(f"Mismatch in number of boxes and masks for image {img_name}!")

            # Handle cases where no annotations are present
            if len(target['boxes']) == 0:
                target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            if len(target['masks']) == 0:
                target['masks'] = torch.zeros((0, *image.shape[1:]), dtype=torch.float32)

            # Skeleton
            ##################################
            skeleton = self.skeleton_data.get(img_name, {})
            skeleton_data = {
                'keypoints': [],
                'individuals': []
            }
            max_keypoints = max([len(parts) for parts in skeleton.values()], default=0)

            for individual, parts in skeleton.items():
                keypoints = []
                for part, coords in parts.items():
                    x = coords.get('x', 0.0)  # Replace NaN with 0.0
                    y = coords.get('y', 0.0)  # Replace NaN with 0.0
                    likelihood = coords.get('likelihood', 0.0)  # Replace NaN with 0.0

                    keypoints.append([x, y, likelihood])

                keypoints_tensor = torch.tensor(keypoints, dtype=torch.float32)

                # Replace NaN with 0 in keypoints_tensor
                keypoints_tensor = torch.nan_to_num(keypoints_tensor, nan=0.0)

                skeleton_data['keypoints'].append(keypoints_tensor)
                skeleton_data['individuals'].append(individual)

            # Stack all keypoints into one tensor if there are multiple individuals
            if skeleton_data['keypoints']:
                skeleton_data['keypoints'] = torch.cat(skeleton_data['keypoints'], dim=0)
            else:
                skeleton_data['keypoints'] = torch.zeros((3, image.shape[2], image.shape[3]),
                                                         dtype=torch.float32)  # Default if no skeleton data

            if torch.isnan(skeleton_data['keypoints']).any():
                print("NaN detected in skeleton keypoints before sending to model")

            # Return image and keypoints separately
            return image.squeeze(0), target, skeleton_data['keypoints']

        def visualize_sample(self, idx):
            """
            Visualizes the image, bounding boxes, masks, and labels for a given index.
            Args:
                idx (int): Index of the sample to visualize.
            """
            image, target, _ = self[idx]
            fig, ax = plt.subplots(1, figsize=(12, 8))

            # Display the image
            ax.imshow(image.permute(1, 2, 0).cpu().numpy())

            # Define colormap for masks and labels
            colormap = plt.get_cmap('Set1')

            # Plot bounding boxes, masks, and labels
            for box, mask, label in zip(target['boxes'], target['masks'], target['labels']):
                # Plot bounding box
                x_min, y_min, x_max, y_max = box
                rect = patches.Rectangle(
                    (x_min, y_min), x_max - x_min, y_max - y_min,
                    linewidth=2, edgecolor='orange', facecolor='none'
                )
                ax.add_patch(rect)

                # Display label
                label_name = list(name_mapping.keys())[list(name_mapping.values()).index(label.item())]
                ax.text(
                    x_min, y_min - 5, label_name,
                    color='white', fontsize=12, bbox=dict(facecolor='orange', edgecolor='none', pad=2)
                )

                # Display mask
                mask = mask.squeeze(0).cpu().numpy()  # Remove channel dimension and convert to numpy
                ax.imshow(mask, alpha=0.3, cmap=colormap, interpolation='none')

            plt.axis('off')
            plt.show()

    segment_datasets = []
    skeleton_datasets = []
    for dir in os.listdir(segments_dirs):
        if dir.endswith('.v7'):
            continue
        else:
            json_dir = os.path.join(segments_dirs, dir)
            segmentation_data = load_segmentation_data(json_dir)
            skeleton_data = load_skeleton_data(csv_directory, dir)
            image_dir = os.path.join(image_directory, dir)
            if not segmentation_data or not skeleton_data:
                print(f"No valid data found in directory: {json_dir}")
                continue

            # Align data into segmented and skeleton
            aligned_segmentation_data, aligned_skeleton_data = align_data(segmentation_data, skeleton_data, dir)

            # Create the dataset and add it to the list
            segment_data = MouseDataset(aligned_segmentation_data, aligned_skeleton_data, image_dir)
            ##### Visualize ground truth to ensure it is being loaded correctly
            # segment_data.visualize_sample(0)
            #####
            segment_datasets.append(segment_data)
            skeleton_datasets.append(segment_data)  # Also adding to skeleton_datasets

    combined_segment = ConcatDataset(segment_datasets)

    # Split the dataset into training and validation
    train_size = int((1 - val_split) * len(combined_segment))
    val_size = len(combined_segment) - train_size
    train_dataset, val_dataset = random_split(combined_segment, [train_size, val_size])

    # Create DataLoader for training and validation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)

    return train_loader, val_loader
