import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

annotation_file = '/home/turing_lab/cse12210414/projects/RT-DETR/rtdetrv2_pytorch/dataset/kitti_coco/annotations/instances_train2017.json'
image_dir = '/home/turing_lab/cse12210414/projects/RT-DETR/rtdetrv2_pytorch/dataset/kitti_coco/train2017'

with open(annotation_file, 'r') as f:
    coco_data = json.load(f)

def get_image_data(image_id):
    for img in coco_data['images']:
        if int(img['id']) == image_id:
            return img
    return None

def display_bboxes(image_id):
    # Get image information
    image_info = get_image_data(image_id)
    if not image_info:
        print(f"Image ID {image_id} not found!")
        return

    # Load the image
    image_path = os.path.join(image_dir, image_info['file_name'])
    image = Image.open(image_path)

    # Create figure and axes
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)

    # Get annotations for the image
    annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]

    # Draw each bounding box
    for ann in annotations:
        bbox = ann['bbox']  # [x, y, width, height]
        category_id = ann['category_id']

        # Find category name
        category_name = next(cat['name'] for cat in coco_data['categories'] if cat['id'] == category_id)

        # Draw rectangle
        rect = patches.Rectangle(
            (bbox[0], bbox[1]), bbox[2], bbox[3],
            linewidth=2, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)

        # Add label
        plt.text(bbox[0], bbox[1] - 5, category_name, color='red', fontsize=10, weight='bold')

    plt.axis('off')
    plt.savefig(f'bbox_{image_id}.png')

display_bboxes(230)