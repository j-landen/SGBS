import deeplabcut as dlc
import os

project_name = "DLC_thermography"
username = "jlanden"
#list_videos = ['video1_path', 'video2_path', 'video3_path']
model_num = 1
config_path = 'F:\DLC_thermography_solo-JL-2024-11-19\config.yaml'
video_directory = r'F:\DLC_thermography_solo-JL-2024-11-19\videos'  # Replace with the directory containing your videos
videotype='.avi'

def train_and_evaluate(config_path, model_num, max_iterations):
    #dlc.create_multianimaltraining_dataset(config_path)
    #dlc.train_network(config_path, shuffle=model_num, maxiters=max_iterations, gputouse=0, allow_growth=True)
    dlc.evaluate_network(config_path, gputouse=0, plotting=True)
    dlc.extract_save_all_maps(config_path, shuffle=model_num,  gputouse=0)

def analyze_videos_in_directory(config_path, directory, videotype):
    # List all files in the directory
    video_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(videotype)]

    # Analyze each video using DeepLabCut
    if video_paths:
        dlc.analyze_videos(config_path, video_paths, save_as_csv=True, videotype=videotype)
        dlc.create_video_with_all_detections(config_path, video_paths, videotype=videotype)
    else:
        print(f"No videos with type {videotype} found in directory {directory}.")

## Create project
#dlc.create_new_project(project_name,username,
#              ['full path of video 1', 'full path of video 2'],
#              copy_videos=True, multianimal=True)

## IF NEEDED: Add videos in addition to those added above
#dlc.add_new_videos(config_path, ['full path of video 1', 'full path of video 2'], copy_videos=True/False)

## From each video, pull out 50 frames for annotation (# can be changed in config.yaml)
#dlc.extract_frames(config_path, mode='automatic', algo='kmeans', userfeedback=False, crop=False)

## Annotate frames of videos  ###########################################
#dlc.label_frames(config_path)

## Split annotations into training and evaluation datasets
#dlc.check_labels(config_path, visualizeindividuals=True)

## Train network based on annotations created above
## Evaluate & save results
train_and_evaluate(config_path, model_num, max_iterations=200000)

## Check results of evaluation & determine if more annotation is needed

## Analyze entire videos based on model that was created above
## Saves detection videos and a CSV file for combination with Mask R-CNN
analyze_videos_in_directory(config_path, video_directory, videotype)



### IF TRACKING IS NOT PERFORMING HOW YOU WOULD LIKE:
## Find outliers and relabel
#dlc.find_outliers_in_raw_data(config_path, pickle_path, video_path)
#dlc.label_frames(config_path)

## Merge new annotations and retrain new model (usually need less iterations)
#dlc.merge_datasets(config_path)
#train_and_evaluate(config_path, model_num, maxiters=100000)
