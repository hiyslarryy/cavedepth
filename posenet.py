import torch
import torch.nn as nn
import torch.nn.functional as F

class PoseNet(nn.Module):
    def __init__(self):
        super(PoseNet, self).__init__()
        self.conv1 = nn.Conv2d(9, 64, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)
        self.pose_pred = nn.Conv2d(1024, 12, kernel_size=1)  # Predicting 12 values (2 sets of 6-DoF)

    def forward(self, img_seq):
        x = F.relu(self.conv1(img_seq))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        pose = self.pose_pred(x)
        pose = pose.mean(dim=[2, 3])  # Global average pooling
        pose = pose.view(-1, 12)  # Reshape to (batch_size, 12)
        # Assuming `pose` is the output from PoseNet with shape [batch_size, 12]
        pose12 = pose[:, :6]  # Relative pose from image 1 to image 2
        pose23 = pose[:, 6:]  # Relative pose from image 2 to image 3
        return pose12, pose23
    
    # Example usage
if __name__ == "__main__":
    posenet = PoseNet()
    img_seq = torch.randn(1, 9, 960, 540)  # Example input: batch size of 1, 9 channels, 960x540
    pose12, pose23 = posenet(img_seq)
    print(f"Pose from image 1 to 2: {pose12.shape}")  # Should print [batch_size, 6]
    print(f"Pose12:{pose12}")
    print(f"Pose from image 2 to 3: {pose23.shape}")  # Should print [batch_size, 6]
    print(f"Pose12:{pose23}")
    print(f"Total:{[pose12,pose23]}")
