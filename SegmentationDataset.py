import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Normalize
from PIL import Image
import os
import json

class SegmentationDataset(Dataset):
    def __init__(self, root_dir, transforms):
        self.root_dir = root_dir
        self.transforms = transforms
        self.image_dir = os.path.join(root_dir, "images")
        self.masks_json = os.path.join(root_dir, "masks.json")
        with open(self.masks_json, "r") as f:
            self.masks_dict = json.load(f)
        
        # Create a dictionary to map category names to IDs
        self.category_map = {}
        for category in self.masks_dict["categories"]:
            self.category_map[category["name"]] = category["id"]
    
    def __len__(self):
        return len(self.masks_dict["annotations"])
    
    def __getitem__(self, idx):
        # Load image
        image_id = self.masks_dict["annotations"][idx]["image_id"]
        img_path = os.path.join(self.image_dir, f"{str(image_id).zfill(5)}.jpg")
        image = Image.open(img_path).convert("RGB")
        
        # Get mask coordinates for this image
        annotation = self.masks_dict["annotations"][idx]
        category_id = annotation["category_id"]
        category_name = next(cat["name"] for cat in self.masks_dict["categories"] if cat["id"] == category_id)
        mask_coords = annotation["segmentation"][0]
        
        # Apply transforms
        image, mask_coords = self.transforms(image, mask_coords)
        
        # Convert category name to ID
        category_id = self.category_map[category_name]
        
        # Create binary mask from mask coordinates
        mask = self.create_mask(image.size, mask_coords)
        
        return image, mask, category_id
    
    def create_mask(self, image_size, mask_coords):
        mask = Image.new("L", image_size, 0)
        draw = ImageDraw.Draw(mask)
        draw.polygon(mask_coords, fill=1)
        return mask

