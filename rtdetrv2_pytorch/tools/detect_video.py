import torch
from torch import nn
import cv2
import numpy as np
from torchvision import transforms

"""Copyright(c) 2024 lyuwenyu. All Rights Reserved.
"""

import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.core import YAMLConfig

dependencies = ['torch', 'torchvision',]
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def _load_checkpoint(path: str, map_location='cpu'):
    state = torch.load(path, map_location=map_location)
    return state


def _build_model(args, ):
    """main
    """
    cfg = YAMLConfig(args.config)

    if args.resume:
        checkpoint = _load_checkpoint(args.resume, map_location='cpu') 
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']

        # NOTE load train mode state
        cfg.model.load_state_dict(state)


    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    return Model()


CONFIG = {
    'rtdetrv2_r101vd': {
        'config': '/home/turing_lab/cse12210414/projects/RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r101vd_6x_coco.yml',
        'resume': '/home/turing_lab/cse12210414/projects/RT-DETR/rtdetrv2_pytorch/logs/best_random.pth',
    },
}

def rtdetrv2_r101vd(pretrained=True):
    args = type('Args', (), CONFIG['rtdetrv2_r101vd'])()
    args.resume = args.resume if pretrained else ''
    return _build_model(args, )


rtdetrv2_x = rtdetrv2_r101vd

# Define your model architecture (ensure it matches the saved model)
model = rtdetrv2_x(pretrained=True).to(device)

# Define preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load video
cap = cv2.VideoCapture('/home/turing_lab/cse12210414/projects/RT-DETR/rtdetrv2_pytorch/test.MP4')  # Replace with your video file
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Prepare video writer
out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

# Class labels (replace with your custom labels if needed)
COCO_LABELS = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
    'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]
KITTI_LABELS = [
    'car', 'pedestrian', 'cyclist'
]
frame_count = 0
color = [
    (0, 255, 0), (0, 0, 255), (255, 0, 0)
]
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    height_orig, width_orig = frame.shape[:2]
    new_height, new_width = 640, 640
    image = cv2.resize(frame, (new_width, new_height))
    image = transform(image).unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        predictions = model(image, torch.tensor([[width_orig, height_orig]]).to(device))

    (labels, boxes, scores) = predictions
    
    # Draw bounding boxes and labels
    for label, box, score in zip(labels[0], boxes[0], scores[0]):
        if score < 0.5 or label not in range(3):
            continue

        box = [int(b) for b in box]
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color[label], 2)
        cv2.putText(frame, f'{KITTI_LABELS[int(label)]}: {score:.2f}', (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color[label], 2)

    # Save the frame
    out.write(frame)
    cv2.imwrite(f"output_frame/4/output_{frame_count}.jpg", frame)
    frame_count += 1

cap.release()
out.release()
cv2.destroyAllWindows()
