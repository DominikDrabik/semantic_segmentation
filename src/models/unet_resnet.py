"""
Defines a U-Net style semantic segmentation model with a ResNet-50 encoder backbone.
- Encoder: pretrained ResNet-50 layers (conv1 → layer4)
- Bottleneck: two 3×3 convolutional blocks with optional dropout
- Decoder: four upsampling stages with skip-connections and convolutional blocks
- Output: 1×1 convolution to `num_classes`, bilinearly upsampled to input resolution
"""


from torchvision import models
import torch
import torch.nn as nn
from torch.nn import functional as F

class ConvBlock(nn.Module):
    """Convolutional block with batch normalization and ReLU activation"""
    def __init__(self, in_channels, out_channels, dropout=False, dropout_rate=0.3):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout2d(dropout_rate))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class UNetResNet(nn.Module):
    def __init__(self, num_classes=19, pretrained=True):
        super().__init__()
        
        self.resnet = models.resnet50(pretrained=pretrained)
        
        self.encoder1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu
        )  # 64 channels
        
        self.pool = self.resnet.maxpool
        self.encoder2 = self.resnet.layer1  # 256 channels
        self.encoder3 = self.resnet.layer2  # 512 channels
        self.encoder4 = self.resnet.layer3  # 1024 channels
        self.encoder5 = self.resnet.layer4  # 2048 channels
        
        self.bottleneck = ConvBlock(2048, 2048, dropout=True)
        
        self.upconv5 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.decoder5 = ConvBlock(2048, 1024)  # 1024 + 1024 input
        
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = ConvBlock(1024, 512)   # 512 + 512 input
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = ConvBlock(512, 256)    # 256 + 256 input
        
        self.upconv2 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)
        self.decoder2 = ConvBlock(128, 64)     # 64 + 64 = 128 input
        
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        input_size = x.size()[2:]
        # Encoder
        e1 = self.encoder1(x)        
        p1 = self.pool(e1)           
        e2 = self.encoder2(p1)       
        e3 = self.encoder3(e2)       
        e4 = self.encoder4(e3)       
        e5 = self.encoder5(e4)       
        
        b = self.bottleneck(e5)
        
        # Decoder 
        d5 = self.upconv5(b)
        if d5.size()[2:] != e4.size()[2:]:
            d5 = F.interpolate(d5, size=e4.size()[2:], mode='bilinear', align_corners=False)
        d5 = self.decoder5(torch.cat([d5, e4], dim=1))
        
        d4 = self.upconv4(d5)
        if d4.size()[2:] != e3.size()[2:]:
            d4 = F.interpolate(d4, size=e3.size()[2:], mode='bilinear', align_corners=False)
        d4 = self.decoder4(torch.cat([d4, e3], dim=1))
        
        d3 = self.upconv3(d4)
        if d3.size()[2:] != e2.size()[2:]:
            d3 = F.interpolate(d3, size=e2.size()[2:], mode='bilinear', align_corners=False)
        d3 = self.decoder3(torch.cat([d3, e2], dim=1))
        
        d2 = self.upconv2(d3)
        if d2.size()[2:] != e1.size()[2:]:
            d2 = F.interpolate(d2, size=e1.size()[2:], mode='bilinear', align_corners=False)
        d2 = self.decoder2(torch.cat([d2, e1], dim=1))
        
        d1 = self.upconv1(d2)
        
        output = self.final_conv(d1)
        
        if output.size()[2:] != input_size:
            output = F.interpolate(output, size=input_size, mode='bilinear', align_corners=False)
            
        return output