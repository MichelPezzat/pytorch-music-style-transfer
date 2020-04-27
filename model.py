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

class Conv2d_Layer(nn.Module):

    """Residual Block with instance normalization."""

    def __init__(self, dim_in, dim_out,ks,s,padding = 1):

        super(Conv2d_Layer, self).__init__()

        self.main = nn.Sequential(

            nn.Conv2d(dim_in, dim_out, kernel_size=ks, stride=s, padding=padding, bias=False),

            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),

            nn.ReLU(inplace=True))



    def forward(self, x):

        return self.main(x)

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




        #layers.append(nn.ReflectionPad2d((3,3,3,3))) 



        # Down-sampling layers.

        

        layers.append(Conv2d_Layer(dim_in=3, dim_out=conv_dim,ks = 7,s = 1,padding=3))
        layers.append(Conv2d_Layer(dim_in=conv_dim, dim_out=conv_dim*2,ks =4,s = 2))
        layers.append(Conv2d_Layer(dim_in=conv_dim*2, dim_out=conv_dim*4,ks = 4,s = 2))

        



        # Bottleneck layers.

        for i in range(repeat_num):

            layers.append(ResidualBlock(dim_in=conv_dim*4, dim_out=conv_dim*4))

        self.downsample = nn.Sequential(*layers)





        # Up-sampling layers.
        self.up1 = ConvTranspose2d_Layer(dim_in=conv_dim*4+c_dim, dim_out=conv_dim*2,ks = 4,s = 2)
        self.up2 = ConvTranspose2d_Layer(dim_in=conv_dim*2+c_dim, dim_out=conv_dim,ks = 4,s = 2)
        self.up3 = nn.Conv2d(conv_dim+c_dim, 3, kernel_size=7, stride=1, bias=False,padding =3)
        

        



    def forward(self, x, c):

        # Replicate spatially and concatenate domain information.

        x = self.downsample(x)
        c = c.view(c.size(0), c.size(1), 1, 1)

        c1 = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c1], dim=1)
        x = self.up1(x)

        c2 = c.repeat(1,1,x.size(2), x.size(3))
        x = torch.cat([x, c2], dim=1)
        x = self.up2(x)

        c3 = c.repeat(1,1,x.size(2), x.size(3))
        x = torch.cat([x, c3], dim=1)
        x = self.up3(x)

        s = nn.Sigmoid()



        

        return s(x)



class Discriminator(nn.Module):

    """Discriminator network with PatchGAN."""

    def __init__(self, conv_dim=64, repeat_num=5, num_speakers=3):

        super(Discriminator, self).__init__()

        c_dim = num_speakers

        layers = []

        layers.append(nn.Conv2d(1+c_dim, conv_dim, kernel_size=4, stride=2))

        layers.append(nn.LeakyReLU(0.01))

        layers.append(Conv2d_Layer(dim_in=conv_dim, dim_out=conv_dim*4,ks = 4,s = 2))

        layers.append(nn.Conv2d(conv_dim*4, 1, kernel_size=1, stride=1))
        

        self.main = nn.Sequential(*layers)

        
        

    def forward(self, x, c):

        # Replicate spatially and concatenate domain information.

        c = c.view(c.size(0), c.size(1), 1, 1)

        c = c.repeat(1, 1, x.size(2), x.size(3))

        x = torch.cat([x, c], dim=1)

        return self.main(x)

class DomainClassifier(nn.Module):

    def __init__(self, conv_dim=64, repeat_num=5, num_speakers=10):

        super(DomainClassifier, self).__init__()

        layers = []

        layers.append(nn.Conv2d(1, conv_dim, kernel_size=[1,12], stride=[1,12]))

        layers.append(nn.LeakyReLU(0.01))

        layers.append(Conv2d_Layer(dim_in=conv_dim, dim_out=conv_dim*2,ks =[4,1],s =[4,1]))

        layers.append(Conv2d_Layer(dim_in=conv_dim*2, dim_out=conv_dim*4,ks =[2,1],s = [2,1]))
        
        layers.append(Conv2d_Layer(dim_in=conv_dim*4, dim_out=conv_dim*8,ks =[8,1],s =[8,1]))
        
        layers.append(nn.Conv2d(conv_dim*8, 3,kernel_size =[1,7],stride =[1,7]))

        self.main = nn.Sequential(*layers)

        
        

    def forward(self, x):

        # Replicate spatially and concatenate domain information.

        x = self.main(x)

        x = x.view(x.size(0),x.size(1))


        return x



t = torch.rand([1,3,64,84])
l = torch.FloatTensor([[0,1,0,0]])
print(l.size())

G = Generator()
o1 = G(t, l)



#C=DomainClassifier()
#o1 = C(t)
print(o1.size())