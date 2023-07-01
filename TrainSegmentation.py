#-------------- train----------------
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from tqdm import tqdm
import json
from PIL import Image
import numpy as np

import SegmentationDataset
import SegmentationModule

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define hyperparameters
num_classes = 6
batch_size = 8
epochs = 50
learning_rate = 0.001
weight_decay = 0.0005

# Define transforms
transforms = Compose([ToTensor(),
                      Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# Load datasets
train_dataset = SegmentationDataset("train", transforms)
val_dataset = SegmentationDataset("val", transforms)

# Load data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Load masks coordinates from JSON file
with open("instances_default.json", "r") as f:
    masks_data = json.load(f)

# Create segmentation masks for each image
masks = {}
for mask_data in masks_data["annotations"]:
    image_id = mask_data["image_id"]
    category_id = mask_data["category_id"]
    segmentation = mask_data["segmentation"]
    if image_id not in masks:
        masks[image_id] = np.zeros((1080, 1920, num_classes), dtype=np.uint8)
    for segment in segmentation:
        polygon = np.array(segment).reshape((-1, 2))
        mask = Image.new("L", (1920, 1080), 0)
        ImageDraw.Draw(mask).polygon(polygon.tolist(), outline=1, fill=1)
        mask = np.array(mask)
        masks[image_id][:, :, category_id-1] = np.logical_or(masks[image_id][:, :, category_id-1], mask)

# Initialize model and move to device
model = SegmentationModule(num_classes=num_classes).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=255).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Train loop
for epoch in range(epochs):
    # Train
    model.train()
    running_loss = 0.0
    for inputs, targets in tqdm(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        image_ids = targets[:, 0].cpu().numpy().tolist()
        target_masks = [masks[image_id] for image_id in image_ids]
        target_masks = np.stack(target_masks, axis=0)
        target_masks = torch.from_numpy(target_masks).to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, target_masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_dataset)
    # Evaluate
    model.eval()
    running_loss = 0.0
    for inputs, targets in tqdm(val_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        running_loss += loss.item() * inputs.size(0)
    epoch_val_loss = running_loss / len(val_dataset)

    # Print epoch loss
    print("Epoch {} - Loss: {:.4f} - Val Loss: {:.4f}".format(epoch+1, epoch_loss, epoch_val_loss))

    # Save model checkpoint
    torch.save(model.state_dict(), "segmentation_module_defects_{}.pth".format(epoch+1))
