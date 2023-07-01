import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from my_dataset import MyDataset
import DetectionModule

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define transforms for input images and annotations
transform_img = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_ann = transforms.Compose([
    # Define annotation transforms if necessary
])

# Define dataset
train_dataset = MyDataset(data_path='path/to/train/data',
                          transform_img=transform_img,
                          transform_ann=transform_ann)

# Define data loader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Define model and optimizer
model = DetectionModule(num_classes=2).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define loss function
cls_criterion = nn.CrossEntropyLoss()
bbox_criterion = nn.SmoothL1Loss()

# Train the model
for epoch in range(10):
    for i, (images, targets) in enumerate(train_loader):
        images = images.to(device)
        targets = targets.to(device)
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Forward pass
        cls_preds, bbox_preds = model(images)
        
        # Compute loss
        cls_loss = cls_criterion(cls_preds, targets['labels'])
        bbox_loss = bbox_criterion(bbox_preds, targets['boxes'])
        loss = cls_loss + bbox_loss
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Print loss
        if i % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch+1, 10, i+1, len(train_loader), loss.item()))

# Save the trained model
torch.save(model.state_dict(), 'path/to/save/model.pth')
