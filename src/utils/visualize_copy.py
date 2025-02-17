"""
Mask R-CNN Visualization Functions

Original Code:
- Copyright (c) 2017 Matterport, Inc.
- Written by Waleed Abdulla
- Licensed under the MIT License (see LICENSE for details)
- Source: https://github.com/matterport/Mask_RCNN

Modifications:
- Jason Landen, 2024
- I adjusted the display_instances function to support keypoint overlays
"""

import os
import sys
import random
import colorsys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches,  lines


# Root directory of the project
ROOT_DIR = os.path.abspath("../../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils


############################################################
#  Visualization
############################################################

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, alpha=0.5, threshold=0.8):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask >= threshold,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def display_instances(image, boxes, masks, class_ids, class_names,
                      keypoints=None, scores=None,
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None, save_to_file=None, threshold=0.8):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    keypoints: (optional) [num_instance, num_keypoints, 3] in (x, y, visibility) format.
    scores: (optional) confidence scores for each box
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    save_to_file: (optional) If provided, saves the figure to the specified file path.
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Generate random colors
    colors = colors or random_colors(N)

    # Show area outside image boundaries.  We want this off
    height, width = image.shape[:2]
    ax.set_ylim(height)
    ax.set_xlim(width)
    ax.axis('off')

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        x1, y1, x2, y2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                alpha=0.7, linestyle="dashed",
                                edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        ax.text(x1, y1 + 8, caption,
                color=color, size=11, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color, threshold)

        # Draw keypoints
        if keypoints is not None:
            for kp in keypoints:
                kp_x, kp_y, kp_vis = kp

                if kp_vis > 0:  # Only plot visible keypoints
                    ax.plot(kp_x, kp_y, 'o', color="gray", markersize=4)  # Keypoint as a circle
                    # ax.text(kp_x, kp_y, f"{kp_x:.1f},{kp_y:.1f}", fontsize=6, color='yellow')  # Keypoints coordinates

    ax.imshow(masked_image.astype(np.uint8))

    # Save the figure if save_to_file is provided
    if save_to_file:
        os.makedirs(os.path.dirname(save_to_file), exist_ok=True)
        plt.savefig(save_to_file, bbox_inches='tight', pad_inches=0)

    elif auto_show:
        plt.show()
    plt.close()
