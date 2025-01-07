import albumentations as A
import cv2
import os
import json

ORIGINAL_PATH = '/home/turing_lab/cse12210414/projects/RT-DETR/rtdetrv2_pytorch/dataset/kitti_coco'
AUGMENTED_PATH = '/home/turing_lab/cse12210414/projects/RT-DETR/rtdetrv2_pytorch/dataset/kitti_coco_strong'

light_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomCrop(width=1000, height=300),
    A.RandomBrightnessContrast(p=0.5),
], bbox_params=A.BboxParams(format='coco', min_area=1, min_visibility=0.5, label_fields=['class_labels']), p=1)

medium_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomCrop(height=300, width=900, p=1),
    A.GridDropout(ratio=0.3, unit_size_range=(100, 200), fill="inpaint_ns", p=1.0)
], bbox_params=A.BboxParams(format='coco', min_area=1, min_visibility=0.5, label_fields=['class_labels']),  p=1)

strong_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomCrop(height=300, width=800, p=1),
    A.MotionBlur(blur_limit=(3, 11), p=1),
], bbox_params=A.BboxParams(format='coco', min_area=1, min_visibility=0.5, label_fields=['class_labels']), p=1)

if __name__ == '__main__':
    train_annotations = os.path.join(ORIGINAL_PATH, 'annotations', 'instances_train2017.json')
    images_dir = os.path.join(ORIGINAL_PATH, 'train2017')
    
    with open(train_annotations, 'r') as f:
        train_data = json.load(f)
        
    images = train_data['images']
    annotations = train_data['annotations']
    categories = train_data['categories']
    
    augmented_image = []
    augmented_annotations = []
    
    augmented_image_id = 0
    augmented_annotations_id = 0
    for image in images:
        # Get image information
        image_id = image['id']
        file_name = image['file_name']
        image_path = os.path.join(images_dir, file_name)
        image = cv2.imread(image_path)
        
        # Get annotations for the image
        image_annotations = [annotation for annotation in annotations if annotation['image_id'] == image_id]
        bboxes = []
        class_labels = []
        for annotation in image_annotations:
            bbox = annotation['bbox']
            category_id = annotation['category_id']
            category = [category for category in categories if category['id'] == category_id][0]
            class_label = category['name']
            
            bboxes.append(bbox)
            class_labels.append(class_label)
            
        # Original image
        original_file_name = file_name.split('.')[0] + '_original.jpg'
        original_image_path = os.path.join(AUGMENTED_PATH, 'train2017', original_file_name)
        cv2.imwrite(original_image_path, image)
        for i, bbox in enumerate(bboxes):
            augmented_image.append({
                "id": augmented_image_id,
                "file_name": original_file_name,
                "height": image.shape[0],
                "width": image.shape[1]
            })
            augmented_annotations.append({
                "id": augmented_annotations_id,
                "image_id": augmented_image_id,
                "category_id": [category['id'] for category in categories if category['name'] == class_labels[i]][0],
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0
            })
            augmented_annotations_id += 1
        augmented_image_id += 1
        
        # # Apply slight transformation
        # slight = slight_transform(image=image, bboxes=bboxes, class_labels=class_labels)
        # slight_image = slight['image']
        # slight_bboxes = slight['bboxes']
        # slight_class_labels = slight['class_labels']
        # slight_file_name = file_name.split('.')[0] + '_slight.jpg'
        # slight_image_path = os.path.join(AUGMENTED_PATH, 'train2017', slight_file_name)
        # cv2.imwrite(slight_image_path, slight_image)
        # for i, bbox in enumerate(slight_bboxes):
        #     augmented_image.append({
        #         "id": augmented_image_id,
        #         "file_name": slight_file_name,
        #         "height": slight_image.shape[0],
        #         "width": slight_image.shape[1]
        #     })
        #     augmented_annotations.append({
        #         "id": augmented_annotations_id,
        #         "image_id": augmented_image_id,
        #         "category_id": [category['id'] for category in categories if category['name'] == slight_class_labels[i]][0],
        #         "bbox": bbox,
        #         "area": bbox[2] * bbox[3],
        #         "iscrowd": 0
        #     })
        #     augmented_annotations_id += 1
        # augmented_image_id += 1
        
        # Apply light transformation
        # light = light_transform(image=image, bboxes=bboxes, class_labels=class_labels)
        # light_image = light['image']
        # light_bboxes = light['bboxes']
        # light_class_labels = light['class_labels']
        # light_file_name = file_name.split('.')[0] + '_light.jpg'
        # light_image_path = os.path.join(AUGMENTED_PATH, 'train2017', light_file_name)
        # cv2.imwrite(light_image_path, light_image)
        # for i, bbox in enumerate(light_bboxes):
        #     augmented_image.append({
        #         "id": augmented_image_id,
        #         "file_name": light_file_name,
        #         "height": light_image.shape[0],
        #         "width": light_image.shape[1]
        #     })
        #     augmented_annotations.append({
        #         "id": augmented_annotations_id,
        #         "image_id": augmented_image_id,
        #         "category_id": [category['id'] for category in categories if category['name'] == light_class_labels[i]][0],
        #         "bbox": bbox,
        #         "area": bbox[2] * bbox[3],
        #         "iscrowd": 0
        #     })
        #     augmented_annotations_id += 1
        # augmented_image_id += 1
        
        # Apply medium transformation
        # medium = medium_transform(image=image, bboxes=bboxes, class_labels=class_labels)
        # medium_image = medium['image']
        # medium_bboxes = medium['bboxes']
        # medium_class_labels = medium['class_labels']
        # medium_file_name = file_name.split('.')[0] + '_medium.jpg'
        # medium_image_path = os.path.join(AUGMENTED_PATH, 'train2017', medium_file_name)
        # cv2.imwrite(medium_image_path, medium_image)
        # for i, bbox in enumerate(medium_bboxes):
        #     augmented_image.append({
        #         "id": augmented_image_id,
        #         "file_name": medium_file_name,
        #         "height": medium_image.shape[0],
        #         "width": medium_image.shape[1]
        #     })
        #     augmented_annotations.append({
        #         "id": augmented_annotations_id,
        #         "image_id": augmented_image_id,
        #         "category_id": [category['id'] for category in categories if category['name'] == medium_class_labels[i]][0],
        #         "bbox": bbox,
        #         "area": bbox[2] * bbox[3],
        #         "iscrowd": 0
        #     })
        #     augmented_annotations_id += 1
        # augmented_image_id += 1
        
        # Apply strong transformation
        strong = strong_transform(image=image, bboxes=bboxes, class_labels=class_labels)
        strong_image = strong['image']
        strong_bboxes = strong['bboxes']
        strong_class_labels = strong['class_labels']
        strong_file_name = file_name.split('.')[0] + '_strong.jpg'
        strong_image_path = os.path.join(AUGMENTED_PATH, 'train2017', strong_file_name)
        cv2.imwrite(strong_image_path, strong_image)
        for i, bbox in enumerate(strong_bboxes):
            augmented_image.append({
                "id": augmented_image_id,
                "file_name": strong_file_name,
                "height": strong_image.shape[0],
                "width": strong_image.shape[1]
            })
            augmented_annotations.append({
                "id": augmented_annotations_id,
                "image_id": augmented_image_id,
                "category_id": [category['id'] for category in categories if category['name'] == strong_class_labels[i]][0],
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0
            })
            augmented_annotations_id += 1
        augmented_image_id += 1
        
    augmented_data = {
        "images": augmented_image,
        "annotations": augmented_annotations,
        "categories": categories
    }
    
    with open(os.path.join(AUGMENTED_PATH, 'annotations', 'instances_train2017.json'), 'w') as f:
        json.dump(augmented_data, f)