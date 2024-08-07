import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNetEncoder(nn.Module):
    def __init__(self, in_channels):
        super(UNetEncoder, self).__init__()
        self.encoder1 = DoubleConv(in_channels, 64)
        self.encoder2 = DoubleConv(64, 128)
        self.encoder3 = DoubleConv(128, 256)
        self.encoder4 = DoubleConv(256, 512)
        self.encoder5 = DoubleConv(512, 1024)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x1 = self.encoder1(x)
        p1 = self.pool(x1)
        x2 = self.encoder2(p1)
        p2 = self.pool(x2)
        x3 = self.encoder3(p2)
        p3 = self.pool(x3)
        x4 = self.encoder4(p3)
        p4 = self.pool(x4)
        x5 = self.encoder5(p4)
        return x1, x2, x3, x4, x5

class UNetDecoder(nn.Module):
    def __init__(self, out_channels):
        super(UNetDecoder, self).__init__()
        self.bottleneck = DoubleConv(1024, 1024)  # Bottleneck layer
        self.upconv5 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder5 = DoubleConv(1024, 512)
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(512, 256)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(256, 128)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(128, 64)
        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)  # Adjusted channels

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x1, x2, x3, x4, x5):
        x5 = self.bottleneck(x5)

        d5 = self.upconv5(x5)
        d5 = F.interpolate(d5, size=x4.shape[2:], mode='bilinear', align_corners=True)
        d5 = torch.cat((d5, x4), dim=1)
        d5 = self.decoder5(d5)

        d4 = self.upconv4(d5)
        d4 = F.interpolate(d4, size=x3.shape[2:], mode='bilinear', align_corners=True)
        d4 = torch.cat((d4, x3), dim=1)
        d4 = self.decoder4(d4)

        d3 = self.upconv3(d4)
        d3 = F.interpolate(d3, size=x2.shape[2:], mode='bilinear', align_corners=True)
        d3 = torch.cat((d3, x2), dim=1)
        d3 = self.decoder3(d3)

        d2 = self.upconv2(d3)
        d2 = F.interpolate(d2, size=x1.shape[2:], mode='bilinear', align_corners=True)
        d2 = torch.cat((d2, x1), dim=1)
        d2 = self.decoder2(d2)

        d1 = self.upconv1(d2)
        d1 = F.interpolate(d1, size=x1.shape[2:], mode='bilinear', align_corners=True)
        d1 = torch.cat((d1, x1), dim=1)
        d1 = self.decoder2(d1)  # Using decoder2 instead of decoder1 as we have no decoder1

        output = self.final_conv(d1)
        return output

class MultiTaskUNet(nn.Module):
    def __init__(self, in_channels, seg_out_channels, depth_out_channels):
        super(MultiTaskUNet, self).__init__()
        self.encoder = UNetEncoder(in_channels)
        self.seg_decoder = UNetDecoder(seg_out_channels)
        self.depth_decoder = UNetDecoder(depth_out_channels)

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.encoder(x)
        
        # Segmentation output
        seg_output = self.seg_decoder(x1, x2, x3, x4, x5)
        
        # Depth output
        depth_output = self.depth_decoder(x1, x2, x3, x4, x5)
        
        return seg_output, depth_output

# Example usage
if __name__ == "__main__":
    model = MultiTaskUNet(in_channels=3, seg_out_channels=13, depth_out_channels=1)  # Adjust output channels as needed
    x = torch.randn(1, 3, 960, 540)  # Example input
    seg_output, depth_output = model(x)
    print(f"Segmentation Output Shape: {seg_output.shape}")
    print(f"Depth Output Shape: {depth_output.shape}")
