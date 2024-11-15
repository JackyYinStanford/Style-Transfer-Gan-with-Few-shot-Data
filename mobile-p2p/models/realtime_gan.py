#import numpy as np
import torch
import torch.nn as nn
import pickle
from torchsummaryX import summary
import cv2


class Generator(nn.Module):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, conv_dim=12, deconv_kernel_size = 3, repeat_num=5, init_channels=12):
        super(Generator, self).__init__()
        self._name = 'generator'
        self.init_channels = init_channels
        self.enc0 = nn.Sequential(
            nn.Conv2d(3, self.init_channels, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True))
        self.enc1 = nn.Sequential(ResBlock_Enc(dim_in=self.init_channels, dim_out=conv_dim*2))
        self.enc2 = nn.Sequential(ResBlock_Enc(dim_in=conv_dim*2, dim_out=conv_dim*4))
        self.enc3 = nn.Sequential(ResBlock_Enc(dim_in=conv_dim*4, dim_out=conv_dim*6))

        self.trans = nn.Sequential(*[ResBlock_Trans(dim_in=conv_dim*6, dim_group=384, dim_out=conv_dim*6) for i in range(repeat_num)])

        self.dec3 = nn.Sequential(ResBlock_Dec(dim_in=conv_dim*6, dim_group=384, dim_out=conv_dim*4, deconv_kernel_size=deconv_kernel_size)) 
        self.dec2 = nn.Sequential(ResBlock_Dec(dim_in=conv_dim*4, dim_group=320, dim_out=conv_dim*2, deconv_kernel_size=deconv_kernel_size))
        self.dec1 = nn.Sequential(ResBlock_Dec(dim_in=conv_dim*2, dim_group=192, dim_out=self.init_channels, deconv_kernel_size=deconv_kernel_size))
        self.dec0 = nn.Sequential(
            nn.Conv2d(self.init_channels, 96, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(96, 96, kernel_size=deconv_kernel_size, stride=1, padding=deconv_kernel_size // 2, groups=96, bias=True),
            nn.ReLU(inplace=True),
            )
        self.img_branch = nn.Sequential(
            nn.Conv2d(96, 3, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Tanh())
        self.alpha_branch = nn.Sequential(
            nn.Conv2d(96, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid())
          
    def forward(self, x):
        
        enc0 = self.enc0(x)
        enc1 = self.enc1(enc0)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)

        trans = self.trans(enc3)

        dec3 = self.dec3(trans) + enc2
        dec2 = self.dec2(dec3) + enc1
        dec1 = self.dec1(dec2) + enc0
        dec0 = self.dec0(dec1)
        rgb = self.img_branch(dec0)             
        alpha = self.alpha_branch(dec0)
        return rgb, alpha
        

class ResBlock_Enc(nn.Module):
    """Residual Block. for encoder"""
    def __init__(self, dim_in, dim_out):
        super(ResBlock_Enc, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=2, stride=2, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_in, dim_in, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0, bias=True))
        self.conn = nn.Sequential(
            nn.MaxPool2d(2, 2, ceil_mode=True),
            nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0, bias=True))

    def forward(self, x):
        return self.conn(x) + self.main(x)

class ResBlock_Trans(nn.Module):
    """Residual Block. for transform"""
    def __init__(self, dim_in, dim_group, dim_out):
        super(ResBlock_Trans, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_group, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_group, dim_group, kernel_size=3, stride=1, padding=1, groups=dim_group, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_group, dim_out, kernel_size=1, stride=1, padding=0, bias=True))

    def forward(self, x):
        return x + self.main(x)

class ResBlock_Dec(nn.Module):
    """Residual Block. for decoder"""
    def __init__(self, dim_in, dim_group, dim_out, deconv_kernel_size=5):
        super(ResBlock_Dec, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_group, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(dim_group, dim_group, kernel_size=deconv_kernel_size, stride=1, padding=deconv_kernel_size//2, groups=dim_group, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_group, dim_out, kernel_size=1, stride=1, padding=0, bias=False))

    def forward(self, x):
        return self.main(x)

#######################################################################################
class Cond_Patch_Discriminator(nn.Module):
    """Discriminator. PatchGAN."""
    def __init__(self, image_size=256, conv_dim=64, repeat_num=4): #repeat=4,out=10*10
        super(Cond_Patch_Discriminator, self).__init__()
        self._name = 'conditional patch discriminator'

        layers = []
        layers.append(nn.Conv2d(6, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            curr_dim = curr_dim * 2

        layers.append(nn.Conv2d(curr_dim, 1, kernel_size=4, stride=1, padding=1))
        self.main = nn.Sequential(*layers)

    def forward(self, x):

        return self.main(x)


net = Cond_Patch_Discriminator()
print('here')
print(net)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
summary(net, torch.zeros((1, 3, 256, 256)))