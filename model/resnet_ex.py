import torch
import torch.nn as nn
import torch.nn.functional as F
# from .atd_arch import ATD
# from .EMMA.Ufuser import Ufuser
from einops import rearrange
import torchvision.transforms as transforms
from .ops.OmniSR import OmniSR
import time
class BasicBlock(nn.Module):
    """基础残差块（2层卷积）"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                # nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = nn.PReLU(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        return nn.PReLU(out)

class SimpleResNet(nn.Module):
    """3层ResNet（输入/输出均为3通道）"""
    def __init__(self,inc=3,outc=3,donw=1):
        super().__init__()
        # 初始卷积层
        self.conv1 = nn.Conv2d(inc, 16, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(16)
        
        # 3个残差层（每层包含1个BasicBlock）
        self.layer1 = BasicBlock(16, 16)
        self.layer2 = BasicBlock(16, 32, stride=donw) 
        self.layer3 = BasicBlock(32, outc, stride=1)    # 输出通道恢复为3
        
        # 上采样恢复尺寸（若输入输出尺寸需保持一致）
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        out = nn.PReLU(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # 若输入为224x224，此处输出为112x112，需上采样恢复尺寸
        if out.size()[-2:] != x.size()[-2:]:
            out = self.upsample(out)
        return out

class ComplexResNet(nn.Module):
    """3层ResNet（输入/输出均为3通道）"""
    def __init__(self,inc=3,outc=3,dim=64):
        super().__init__()
        # 初始卷积层
        self.conv1 = nn.Conv2d(inc, dim, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(16)
        
        # 3个残差层（每层包含1个BasicBlock）
        self.layer1 = BasicBlock(dim, dim)
        self.layer2 = BasicBlock(dim, dim, stride=1) 
        self.layer3 = BasicBlock(dim, dim, stride=1)    # 输出通道恢复为3
        self.layer4 = BasicBlock(dim, dim, stride=1) 
        self.layer5 = BasicBlock(dim, outc, stride=1)    # 输出通道恢复为3
        
        # 上采样恢复尺寸（若输入输出尺寸需保持一致）
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        out = nn.PReLU(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # 若输入为224x224，此处输出为112x112，需上采样恢复尺寸
        if out.size()[-2:] != x.size()[-2:]:
            out = self.upsample(out)
        return out

class ResidualBlock(nn.Module):
    """残差块（参考ResNet设计）"""
    def __init__(self, in_channels, out_channels, groups=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.PReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        # self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 短连接适配层
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.PReLU()
                # nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        # out = self.bn2(out)
        out = self.relu(out)
        out += self.shortcut(residual)  # 残差连接
        # out = self.relu(out)
        return out

class color_fuse_net(nn.Module):
    def __init__(self, in_channels=3, outc=2):
        super().__init__()
        # 编码器（下采样）
        self.c_enc1 = ResidualBlock(3, 64)
        self.c_enc2 = ResidualBlock(64, 64)
        self.c_enc3 = ResidualBlock(64, 64)
        self.c_enc4 = ResidualBlock(64, 64)
        self.cdown1 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.cdown2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.cdown3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.cdown4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)

        self.d_enc1 = ResidualBlock(3, 64)
        self.d_enc2 = ResidualBlock(64, 64)
        self.d_enc3 = ResidualBlock(64, 64)
        self.d_enc4 = ResidualBlock(64, 64)
        self.ddown1 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.ddown2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.ddown3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.ddown4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)

        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        # self.conv3 = nn.Conv2d(64, 256, kernel_size=3, padding=1)
        # self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv_up4 = nn.Conv2d(64, 256, kernel_size=3, padding=1)
        self.conv_up3 = nn.Conv2d(64, 256, kernel_size=3, padding=1)
        self.conv_up2 = nn.Conv2d(64, 256, kernel_size=3, padding=1)
        self.conv_up1 = nn.Conv2d(64, 256, kernel_size=3, padding=1)

        # self.pool4 = nn.MaxPool2d(2)
        self.relu = nn.PReLU()

        # 桥接层
        self.bridge = ResidualBlock(64, 64)

        self.omni4=OmniSR()
        self.omni3=OmniSR()
        self.omni2=OmniSR()
        self.omni1=OmniSR()

        # # 解码器（上采样）
        self.up1 = nn.PixelShuffle(2)
        self.dec1 = ResidualBlock(128, 64)
        self.up2 = nn.PixelShuffle(2)
        self.dec2 = ResidualBlock(128, 64)
        self.up3 = nn.PixelShuffle(2)
        self.dec3 = ResidualBlock(128, 64)
        self.up4 = nn.PixelShuffle(2)
        self.dec4 = ResidualBlock(128, 64)
        
        # 输出层
        self.out = nn.Conv2d(64, outc, kernel_size=1)

    def forward(self, d,c):
        # 编码器
        c = F.interpolate(c, size=d.shape[2:],mode='bicubic',align_corners=False)
        # c=x[:,3:,:,:]
        # d=x[:,:3,:,:]
        c1 = self.c_enc1(c)
        c2 = self.c_enc2(self.cdown1(c1))
        c3 = self.c_enc3(self.cdown2(c2))
        c4 = self.c_enc4(self.cdown3(c3))

        d1 = self.d_enc1(d)
        d2 = self.d_enc2(self.ddown1(d1))
        d3 = self.d_enc3(self.ddown2(d2))
        d4 = self.d_enc4(self.ddown3(d3))
        
        # with torch.no_grad():
        #     # a=self.ddown4(d4)[0]
        #     a=self.cdown4(c4)[0]
        #     for i in range(len(a)):
        #         img=transforms.ToPILImage()(a[i])
        #         img.save(f'/data/vjuicefs_ai_camera_jgroup_livephoto/11182563/color_transfer/methodOmni/results/temp/{i}.png')

        # 桥接
        # print(e2.shape,c2.shape)
        e= torch.cat((self.ddown4(d4),self.cdown4(c4)),1)
        e=self.conv4(e)
        fuse1=self.omni4(e)
        fuse2=self.omni4(fuse1)
        fuse3=self.omni4(fuse2)
        fuse=self.omni4(fuse3+e)

        # 解码器（含跳跃连接）
        fuse=self.conv_up4(fuse)
        fuse=self.relu(fuse)
        o1 = self.dec1(torch.cat([self.up1(fuse), d4], dim=1))
        # o1 = self.dec1(self.up1(fuse) + d4)
        # o1=self.omni3(o1)
        o1=self.conv_up3(o1)
        o1=self.relu(o1)

        o2 = self.dec2(torch.cat([self.up2(o1), d3], dim=1))
        # o2 = self.dec2(self.up2(o1) + d3)
        # o2=self.omni2(o2)
        o2=self.conv_up2(o2)
        o2=self.relu(o2)

        o3 = self.dec3(torch.cat([self.up3(o2), d2], dim=1))
        # o3 = self.dec3(self.up3(o2) + d2)
        # o3=self.omni1(o3)
        o3=self.conv_up1(o3)
        o3=self.relu(o3)

        o4 = self.dec4(torch.cat([self.up4(o3), d1], dim=1))
        # o4 = self.dec3(self.up4(o3) + d1)
        
        return self.out(o4)
