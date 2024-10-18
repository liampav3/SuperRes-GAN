import torch
from torch import nn


'''
A basic convolution network block for Discriminator
Consists of a single conv layer with Leaky ReLU activation.
No residual connection.
Does NOT maintain image resolution
'''
class DiscriminatorBlock(nn.Module):
    def __init__(self, kernel_size, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size)
        self.act = nn.LeakyReLU()
        self.bn = nn.BatchNorm2d(channels)
    def forward(self, X):
        return self.act(self.bn(self.conv(X)))

'''
A basic network block with residual learning.
Consists 2 conv layers with Leaky ReLU activation.
Includes a residual connection with residual scaling.
Kernel size MUST be odd to maintain image resolution
'''
class ResidualBlock(nn.Module):
    def __init__(self, kernel_size, channels, residual_scale):
        super().__init__()
        padding = (kernel_size-1)//2
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding)

        self.act = nn.LeakyReLU()
        self.res_scale = residual_scale

    def forward(self, X):
        f = self.conv1(X)
        f = self.act(f)
        f = self.conv2(X)
        return f + self.res_scale*X

'''
The generator network of the GAN model.
Formed of a series of Residual Blocks for feature extraction in low-resolution space
followed by bicubic upscaling and a few refinement convolutions.
Contains one long residual connection that skips the series of Residual Blocks
'''
class Generator(nn.Module):
    def __init__(self, num_blocks=5, kernel_size=3, in_channels=3, hchannels=128, residual_scale=.7, upscale_mag=4):
        super().__init__()

        padding = (kernel_size-1)//2
        self.res_scale = residual_scale
        
        self.act = nn.LeakyReLU()

        
        self.conv1 = nn.Conv2d(in_channels, hchannels, kernel_size, padding=padding) #transition to feature space
        self.conv2 = nn.Conv2d(hchannels, hchannels, kernel_size, padding=padding) #post-feature extraction blocks

        #High-res refinements
        self.conv3 = nn.Conv2d(hchannels, hchannels, kernel_size, padding=padding)
        self.conv4 = nn.Conv2d(hchannels, hchannels, kernel_size, padding=padding)

        self.fconv = nn.Conv2d(hchannels, in_channels, 1) #Pooling conv before output

        self.blocks = nn.ModuleList([ResidualBlock(kernel_size, hchannels, residual_scale) for i in range(num_blocks)])

        

        self.upscale = nn.Upsample(scale_factor=upscale_mag) 
        self.upscale_mag = upscale_mag

        


    def forward(self, X):
        f = self.act(self.conv1(X))
        r = f #saving residual

        for block in self.blocks:
            f = block(f)

        f = self.conv2(f)
        f = f + self.res_scale*r #big residual connection
        
        f = self.act(self.upscale(self.conv3(f)))
        f = self.act(self.conv4(f))
        f = self.fconv(f)
        return f

'''
Discriminator network of the GAN model. 
Formed by a series of convolutional blocks with batch normalization followed
by a linear feed-forward network and sigmoid activtion
'''
class Discriminator(nn.Module):
    def __init__(self, image_res, num_blocks=10, kernel_size=5, inchannels=3, hchannels=128):
        super().__init__()

        self.act = nn.LeakyReLU()

        self.conv1 = nn.Conv2d(inchannels, hchannels, kernel_size) #transition into feature space
        
        self.blocks = nn.ModuleList([DiscriminatorBlock(kernel_size, hchannels) for i in range(num_blocks)])
        
        self.conv2 = nn.Conv2d(hchannels, 16, kernel_size) #paring down number of channels for smaller flattened vector
        self.bn = nn.BatchNorm2d(16)
        self.flatten = nn.Flatten()
        flen = 16*(image_res - (num_blocks+2)*(kernel_size-1))**2

        self.dense1 = nn.Linear(flen, 1024)
        self.dense2 = nn.Linear(1024, 1)
        self.sig = nn.Sigmoid()

    
    def forward(self, X):
        f = self.act(self.conv1(X))
        
        for block in self.blocks:
            f = block(f)

        f = self.act(self.bn(self.conv2(f)))

        f = self.flatten(f)
        f = self.act(self.dense1(f))

        f = self.sig(self.dense2(f))

        return f





