from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

ANNFILE_PATH="./dataset/kitti_coco/annotations/instances_val2017.json"
ROOT_PATH="./dataset/kitti_coco/val2017"

transform = transforms.Compose([
    transforms.Resize((300, 800)),
    transforms.ToTensor()
])

def collate_fn(batch):
    images = []
    annotations = []
    for (image, annotation) in batch:
        images.append(image)
        annotations.append(annotation)
    images = torch.stack(images, 0)    
    return images, annotations

dataset = datasets.CocoDetection(root=ROOT_PATH, annFile=ANNFILE_PATH, transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=8)

mean = 0.0
std = 0.0
total_images = 0

for images, _ in tqdm(loader):
    print()
    batch_size = images.size(0)
    images = images.view(batch_size, images.size(1), -1)
    mean += images.mean(2).sum(0)
    std += images.std(2).sum(0)
    total_images += batch_size

mean /= total_images
std /= total_images

print(f"Mean: {mean}")
print(f"Std: {std}")
