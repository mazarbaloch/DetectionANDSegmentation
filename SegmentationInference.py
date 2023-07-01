
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
import matplotlib.pyplot as plt

import SegmentationModule

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model checkpoint
model = SegmentationModule(num_classes=6)
model.load_state_dict(torch.load("segmentation_module_defects_50.pth", map_location=device))
model.to(device)
model.eval()

# Define colors for each defect type
colors = {
    0: (255, 0, 0),   # Knot in red
    1: (0, 255, 0),   # Branch in green
    2: (0, 0, 255),   # Area in blue
    3: (255, 255, 0), # Lines in yellow
    4: (255, 0, 255), # Edge in magenta
    5: (0, 255, 255)  # Stain in cyan
}

# Define transforms
transforms = Compose([ToTensor(),
                      Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# Load the image from DetectionModule output and preprocess image
image = Image.open("test_image.png").convert("RGB")
image_tensor = transforms(image).unsqueeze(0).to(device)

# Run inference
with torch.no_grad():
    output = model(image_tensor)
    prediction = output.argmax(1).squeeze().cpu().numpy()

# Create RGB image with segmentation mask for each defect type
segmentation_mask = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)
for defect_type in colors:
    segmentation_mask[prediction == defect_type] = colors[defect_type]

# Visualize segmentation mask
plt.imshow(segmentation_mask)
plt.show()

# Save segmentation mask as image
segmentation_image = Image.fromarray(segmentation_mask)
segmentation_image.save("segmentation_mask.png")
