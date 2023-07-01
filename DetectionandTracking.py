import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import DetectionModule
import time

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load image and apply transforms
img = Image.open('path/to/image.jpg')
img_draw = ImageDraw.Draw(img)
width, height = img.size

# Draw ROIs. The ROIs can be modified according to the size of the input frames
roi1 = [0, 0, width, height//2]
roi2 = [0, height//2, width, height]
img_draw.rectangle(roi1, outline="green")
img_draw.rectangle(roi2, outline="green")

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

# Define tracking variables
detections1 = []
detections2 = []
tracked_objects = {}

# Define tracking parameters
max_distance = 50
max_age = 5

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

# Assign detections to ROIs
for bbox in bboxes:
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    center = ((x1+x2)//2, (y1+y2)//2)
    if center[1] < height//2:
        detections1.append({'bbox': bbox, 'timestamp': time.time()})
    else:
        detections2.append({'bbox': bbox, 'timestamp': time.time()})

# Track objects in ROI1
for obj_id, obj in tracked_objects.items():
    if not obj['tracked']:
        continue
    if obj['last_seen'] + max_age < time.time():
        del tracked_objects[obj_id]
    else:
        obj['tracked'] = False
for detection in detections1:
    best_match = None
    best_distance = max_distance
    for obj_id, obj in tracked_objects.items():
        if not obj['tracked']:
            distance = ((detection['bbox'][0] + detection['bbox'][2])//2 - (obj['bbox'][0] + obj['bbox'][2])//2)**2 + \
                       ((detection['bbox'][1] + detection['bbox'][3])//2 - (obj['bbox'][1] + obj['bbox'][3])//2)**2
            if distance < best_distance:
                best_distance = distance
                best_match = obj_id
            if best_match is not None:
                tracked_objects[best_match]['tracked'] = True
                tracked_objects[best_match]['last_seen'] = time.time()
                tracked_objects[best_match]['bbox'] = detection['bbox']
            else:
                new_id = len(tracked_objects) + 1
                tracked_objects[new_id] = {'tracked': True, 'last_seen': time.time(), 'bbox': detection['bbox']}

# Track objects in ROI2
for obj_id, obj in tracked_objects.items():
    if not obj['tracked']:
        continue
    if obj['last_seen'] + max_age < time.time():
        del tracked_objects[obj_id]
    else:
        obj['tracked'] = False

    for detection in detections2:
        best_match = None
        best_distance = max_distance
        for obj_id, obj in tracked_objects.items():
            if not obj['tracked']:
                distance = ((detection['bbox'][0] + detection['bbox'][2])//2 - (obj['bbox'][0] + obj['bbox'][2])//2)**2 + \
                           ((detection['bbox'][1] + detection['bbox'][3])//2 - (obj['bbox'][1] + obj['bbox'][3])//2)**2
                if distance < best_distance:
                    best_distance = distance
                    best_match = obj_id
        if best_match is not None:
            tracked_objects[best_match]['tracked'] = True
            tracked_objects[best_match]['last_seen'] = time.time()
            tracked_objects[best_match]['bbox'] = detection['bbox']
        else:
            new_id = len(tracked_objects) + 1
            tracked_objects[new_id] = {'tracked': True, 'last_seen': time.time(), 'bbox': detection['bbox']}

# Crop and save image for segmentation module input
for obj_id, obj in tracked_objects.items():
    bbox = obj['bbox']
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    cropped_img = img.crop((x1, y1, x2, y2))
    cropped_img.save(f'path/to/save/cropped_image_{obj_id}.jpg')
