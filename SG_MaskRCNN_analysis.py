import os
import torch
import json
from PIL import Image
from torchvision.transforms import ToTensor
from tqdm import tqdm
import pandas as pd


from Data_alignment import prepare_keypoints_for_analysis
from SG_MaskRCNN_train import SkeletonGuidedMaskRCNN, class_names, num_classes, colors
from utils import keypoints_to_heatmaps, visualize_prediction_images, filter_duplicate_classes, tiff2temp, \
    save_temperature_stats

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Ignore

# Define directories
csv_directory = r'F:\DLC_thermography-JLanden-2024-07-09\videos' # Keypoints from DLC
analysis_directory = r'F:\Segmentation\analyze_input' # Images to be analyzed
save_to_dir = r'F:\Segmentation\results_12-05-24_BS8_solo' # Where output is saved
temperature_folder = r'F:\Seq_results' # Folder where seq_process output is saved

# Path to saved weights
weights_path = os.path.join(save_to_dir, 'sg-mrcnn_epochs_500.pth')

visualize = False  # Whether you would like images to be saved or not (Default = True)
save_predictions_txt = False  # Whether you would like predictions to be saved to csv file (Default = True)
combine_predictions_temp = True # Whether you would like temp for each body part saved as csv
mask_display_threshold = 0.8

def load_model(weights_path, num_classes):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = SkeletonGuidedMaskRCNN(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()  # Set the model to evaluation mode
    print(f"Model loaded successfully from {weights_path}")
    return model

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

def combine_temperature_predictions(images, predictions, temperature_folder, videoID, frameID):
    'Load in temperature data from processed .seq file for alignment with predictions'

    #Load in temperature data
    tiff_frameID = f"frame_{frameID}.tiff"
    temperature_file = os.path.join(temperature_folder, videoID, "radiometric", tiff_frameID)
    temperature_array = tiff2temp(temperature_file)

    temp_stats = save_temperature_stats(images, predictions, temperature_array, frameID,
                                        threshold=mask_display_threshold, class_names=class_names)
    return temp_stats


def analyze_images(model, video_name, image_folder, keypoints_loader, save_to_dir, visualize=True, threshold=0.8):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    # Ensure output folder exists
    os.makedirs(save_to_dir, exist_ok=True)

    # Create an iterator for the keypoints loader
    keypoints_iter = iter(keypoints_loader)

    combined_temp_stats = []
    # Process each image in the folder
    for image_file in tqdm(os.listdir(image_folder)):
        if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Skipping image {image_file}")
            continue
        image_filename = image_file.split('_')[1].split('.')[0]

        # Load and preprocess image
        image_path = os.path.join(image_folder, image_file)
        image = Image.open(image_path).convert("RGB")
        image_tensor = ToTensor()(image).unsqueeze(0).to(device)

        # Load corresponding keypoints
        try:
            _, keypoints = next(keypoints_iter)
            kp_heatmaps = keypoints_to_heatmaps(keypoints, image_tensor.shape[-2], image_tensor.shape[-1]).to(device)
            # print(f"Kp_heatmap shape: {kp_heatmaps.shape}")
        except StopIteration:
            print(f"No more keypoints available for {image_file}. Skipping.")
            continue

        # Perform inference
        with torch.no_grad():
            model.eval()
            predictions = model(image_tensor, kp_heatmaps)
            predictions = filter_duplicate_classes(predictions)

        # Visualize and save results (implement your visualization function)
        if visualize is True:
            visualize_prediction_images(
                images=image_tensor,
                predictions=predictions,
                epoch=video_name,
                phase="analysis",
                batch_idx=image_filename,
                save_img_dir=save_to_dir,
                class_names=class_names,
                colors=colors,
                keypoints=keypoints,
                threshold=threshold
            )

        if save_predictions_txt is True:
            save_predictions_to_json(predictions, save_to_dir, videoID=video_name, frameID=image_filename)

        if combine_predictions_temp is True:
            temp_stats = combine_temperature_predictions(images=image_tensor,
                                                         predictions=predictions,
                                                         temperature_folder=temperature_folder,
                                                         videoID=video_name, frameID=image_filename)
            combined_temp_stats.extend(temp_stats)

    if combine_predictions_temp is True:
        # Subset temperature data based on predictions
        os.makedirs(f"{save_to_dir}/temp_combined/", exist_ok=True)
        output_csv = f"{save_to_dir}/temp_combined/{video_name}.csv"

        # Save all results to a single CSV file
        df = pd.DataFrame(combined_temp_stats)
        df.to_csv(output_csv, index=False)
        print(f"All temperature statistics saved to {output_csv}")



# Load keypoints
keypoints_loader = prepare_keypoints_for_analysis(csv_directory, analysis_directory)

# Load model
model = load_model(weights_path, num_classes=num_classes)

for image_folder in os.listdir(analysis_directory):
    print(f"Analyzing images in folder: {image_folder}")
    video_name = f"{image_folder}"
    image_path = os.path.join(analysis_directory, image_folder)
    analyze_images(model, video_name, image_path, keypoints_loader, save_to_dir, visualize, threshold=mask_display_threshold)


