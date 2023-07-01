import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import DetectionModule

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load image and apply transforms
img = Image.open('path/to/image.jpg')
img_draw = ImageDraw.Draw(img)
width, height = img.size

# Draw ROIs. The ROIs can be modified according to the size of the input frames
img_draw.rectangle([0, 0, width, height//2], outline="green")
img_draw.rectangle([0, height//2, width, height], outline="green")

# Apply transforms
transform_img = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
img_tensor = transform_img(img).unsqueeze(0)

# Load model
model = DetectionModule(num_classes=2)
model.load_state_dict(torch.load('path/to/model.pth'))
model.to(device)

# Make predictions
model.eval()
with torch.no_grad():
    cls_preds, bbox_preds = model(img_tensor.to(device))
    cls_probs = cls_preds.softmax(dim=1).cpu().numpy()[0, :, :, :]
    bbox_preds = bbox_preds.cpu().numpy()[0, :, :, :]

# Post-process predictions
# Define threshold for classification
cls_threshold = 0.5

# Filter out background class
plank_probs = cls_probs[:, 1]

# Apply threshold
plank_inds = plank_probs > cls_threshold

# Extract bounding boxes
bboxes = bbox_preds[plank_inds, :]

# Crop and save image for segmentation module input
for i, bbox in enumerate(bboxes):
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    cropped_img = img.crop((x1, y1, x2, y2))
    cropped_img.save(f'path/to/save/cropped_image_{i}.jpg')
