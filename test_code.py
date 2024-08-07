import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from data_loader import BaseUnderwaterDataset, rgb_to_class, palette
from Unet import MultiTaskUNet

# Define the transformations (only normalization)
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Paths for the test images and annotations
test_image_dir = "/blue/mdjahiduislam/share/ImageSeg/my_ADE/images/testing"
test_annotation_dir = "/blue/mdjahiduislam/share/ImageSeg/my_ADE/annotations/testing"
save_path = "/blue/mdjahiduislam/junliang.liu/code/cavedepth/test_imgs"
model_path = "/blue/mdjahiduislam/junliang.liu/code/cavedepth/checkpoints/model_epoch_30.pth"

# Create the test loader
image_filenames = os.listdir(test_image_dir)
test_dataset = BaseUnderwaterDataset(test_image_dir, test_annotation_dir, image_filenames, transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

# Initialize the model
num_classes = 13  # Number of classes in your dataset
model = MultiTaskUNet(in_channels=3, seg_out_channels=num_classes, depth_out_channels=1)

# Load the saved model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()

# Function to convert class indices to RGB
def class_to_rgb(mask, palette):
    h, w = mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for color, class_idx in palette.items():
        rgb_mask[mask == class_idx] = color
    return rgb_mask

# Run inference and save the output images
with torch.no_grad():
    for image_filename, (images, masks) in zip(image_filenames, test_loader):
        images = images.to(device)
        seg_output, _ = model(images)
        seg_output = seg_output.argmax(dim=1).squeeze(0).cpu().numpy()

        # Convert the output class indices to RGB
        seg_output_rgb = class_to_rgb(seg_output, palette)

        # Save the image with the original name
        output_image_name = f"output_{os.path.splitext(image_filename)[0]}.png"
        output_image = Image.fromarray(seg_output_rgb)
        output_image.save(os.path.join(save_path, output_image_name))

print("Test images saved successfully.")
