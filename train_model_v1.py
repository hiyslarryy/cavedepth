import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from data_loader import get_loader_number, get_dataset_frame, BaseUnderwaterDataset  
from Unet import MultiTaskUNet  

# Define transformations (only normalization)
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Set paths
image_dir = "/blue/mdjahiduislam/share/ImageSeg/my_ADE/images/training"
annotation_dir = "/blue/mdjahiduislam/share/ImageSeg/my_ADE/annotations/training"
save_path = "/blue/mdjahiduislam/junliang.liu/code/cavedepth/checkpoints"

# Create loaders
loader_number = get_loader_number(image_dir, annotation_dir, transform)
dataset_frame = get_dataset_frame(image_dir, annotation_dir, transform)

# Define the number of classes in your segmentation task
num_classes = 13  # Number of classes in your dataset

# Initialize the model
model = MultiTaskUNet(in_channels=3, seg_out_channels=num_classes, depth_out_channels=1)  # Adjust output channels as needed

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Multi-class Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
num_epochs = 100 # Set the number of epochs

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for images, masks in loader_number:
        images = images.to(device)
        masks = masks.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        seg_output, _ = model(images)
        
        # Compute the segmentation loss
        loss = criterion(seg_output, masks)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Update running loss
        running_loss += loss.item() * images.size(0)
    
    epoch_loss = running_loss / len(loader_number.dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    
    # Save the model checkpoint
    if (epoch + 1) % 10 == 0:  # Save every 5 epochs
        checkpoint_path = os.path.join(save_path, f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)

print("Training completed.")
