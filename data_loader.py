import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

# Define the palette and classes
palette = {
    (128, 0, 0): 0,   # 'plain'
    (0, 128, 0): 1,   # 'first layer'
    (128, 128, 0): 2, # 'caveline'
    (0, 0, 128): 3,   # 'second layer'
    (128, 0, 128): 4, # 'open area'
    (0, 128, 128): 5, # 'attachment rock'
    (128, 128, 128): 6, # 'arrow'
    (64, 0, 0): 7,    # 'reel'
    (192, 0, 0): 8,   # 'cookie'
    (64, 128, 0): 9,  # 'diver'
    (192, 128, 0): 10, # 'stalactite'
    (64, 0, 128): 11, # 'stalagmite'
    (192, 0, 128): 12 # 'column'
}

def rgb_to_class(mask, palette):
    h, w, _ = mask.shape
    class_mask = np.zeros((h, w), dtype=np.int64)
    for color, class_idx in palette.items():
        matches = np.all(mask == color, axis=-1)
        class_mask[matches] = class_idx
    return class_mask

class BaseUnderwaterDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_filenames, transforms=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_filenames = image_filenames
        self.transforms = transforms

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_name = img_name.replace('.jpg', '.png')
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(img_path).convert("RGB")
        mask = np.array(Image.open(mask_path).convert("RGB"))

        # Convert RGB mask to class indices
        mask = rgb_to_class(mask, palette)

        if self.transforms:
            image = self.transforms(image)
        mask = torch.tensor(mask, dtype=torch.long)

        return image, mask

def get_loader_number(image_dir, mask_dir, transforms=None):
    image_filenames = [
        f"{i:05d}.jpg" for i in range(824, 12927)
        if os.path.exists(os.path.join(image_dir, f"{i:05d}.jpg"))
    ]
    return DataLoader(
        BaseUnderwaterDataset(image_dir, mask_dir, image_filenames, transforms),
        batch_size=1, shuffle=False, num_workers=1
    )

def get_dataset_frame(image_dir, mask_dir, transforms=None):
    image_filenames = [
        f"frame{i:03d}.jpg" for i in range(2, 2937)
        if os.path.exists(os.path.join(image_dir, f"frame{i:03d}.jpg"))
    ]
    return DataLoader(
        BaseUnderwaterDataset(image_dir, mask_dir, image_filenames, transforms),
        batch_size=1, shuffle=False, num_workers=1
    )

# Define transformations (only normalization)
transform = transforms.Compose([
    transforms.ToTensor(),
])
