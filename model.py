import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from data_loader import get_loader


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)

class ConvTranspose2d_Layer(nn.Module):

    """Residual Block with instance normalization."""

    def __init__(self, dim_in, dim_out,ks,s,padding = 1):

        super(ConvTranspose2d_Layer, self).__init__()

        self.main = nn.Sequential(

            nn.ConvTranspose2d(dim_in, dim_out, kernel_size=ks, stride=s, padding=padding, bias=False),

            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),

            nn.ReLU(inplace=True))



    def forward(self, x):

        return self.main(x)

class Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, num_speakers=4, repeat_num=10):
        super(Generator, self).__init__()
        c_dim = num_speakers
        layers = []
        layers.append(nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride = 1,padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=3, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
        nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
        nn.ReLU(inplace=True)
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        # Replicate spatially and concatenate domain information.
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        return self.main(x)

class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self,  conv_dim=64, num_speakers=4):
        super(Discriminator, self).__init__()
        c_dim = num_speakers
        layers = []
        layers.append(nn.Conv2d(3+ c_dim, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        
        
        layers.append(nn.Conv2d(conv_dim, conv_dim*4, kernel_size=4, stride=2, padding=1))
        layers.append(nn.InstanceNorm2d(conv_dim*4, affine=True, track_running_stats=True))
        layers.append(nn.LeakyReLU(0.01))

        layers.append(nn.Conv2d(conv_dim*4, 1, kernel_size=1, stride=1, padding=0, bias=False))
        
        self.main = nn.Sequential(*layers)
        
        
        
    def forward(self, x, c):
        
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        h = self.main(x)
        
        
        return h

class DomainClassifier(nn.Module):

    def __init__(self, conv_dim=64, repeat_num=5, num_speakers=4):

        super(DomainClassifier, self).__init__()

        layers = []

        layers.append(nn.Conv2d(3, conv_dim, kernel_size=[1,12], stride=[1,12]))

        layers.append(nn.LeakyReLU(0.01))

        layers.append(nn.Conv2d(conv_dim, conv_dim*2,kernel_size =[4,1],stride =[4,1]))
        layers.append(nn.InstanceNorm2d(conv_dim*2, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(conv_dim*2, conv_dim*4,kernel_size =[2,1],stride = [2,1]))
        layers.append(nn.InstanceNorm2d(conv_dim*4, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))
        
        layers.append(nn.Conv2d(conv_dim*4, conv_dim*8,kernel_size =[8,1],stride =[8,1]))
        layers.append(nn.InstanceNorm2d(conv_dim*8, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))
        
        layers.append(nn.Conv2d(conv_dim*8, 4,kernel_size =[1,7],stride =[1,7]))

        self.main = nn.Sequential(*layers)

        
        

    def forward(self, x):

        # Replicate spatially and concatenate domain information.

        x = self.main(x)

        x = x.view(x.size(0),x.size(1))


        return x
