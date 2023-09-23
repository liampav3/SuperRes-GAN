'''
Liam Pavlovic
CS7180 Project 1
9/23/23
'''
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.io import read_image, write_png
from torch.nn.functional import interpolate

#Loading in BSDS300 test set
#Images are cropped to 300x300 to deal with differences in orientation
test_ds = ImageFolder('BSDS300/images/test', transform=transforms.Compose([transforms.RandomCrop(300), transforms.ToTensor()]))
test_dl = DataLoader(test_ds, batch_size=16)

#Loading in trained generator
gan_model = torch.load('generator.torch')

#Main quantitative evaluation loop
psnr_gan = 0
psnr_bc = 0
for i, batch in enumerate(test_dl):
    high_imgs = batch[0]
    #Downsampling reference images
    low_imgs = interpolate(high_imgs, scale_factor=1/4, mode='bicubic')

    #Generating super-resolution images with bicubic and GAN model
    bc_imgs = interpolate(low_imgs, scale_factor=4, mode='nearest')
    gan_imgs = gan_model(low_imgs)
    
    #Computing MSE for GAN and bicubic
    bc_mse = torch.mean((bc_imgs-high_imgs)**2, dim=(1,2,3))
    gan_mse = torch.mean((gan_imgs-high_imgs)**2, dim=(1,2,3))
    
    #Computing GAN and bicubic PSNR for this batch and adding to running total
    psnr_gan += torch.sum(20*torch.log10(1/torch.sqrt(gan_mse)))
    psnr_bc += torch.sum(20*torch.log10(1/torch.sqrt(bc_mse)))
#Taking average PSNR over dataset
psnr_gan /= len(test_ds)
psnr_bc /= len(test_ds)

print(f'PSNR GAN: {psnr_gan}')
print(f'PSNR Bicubic: {psnr_bc}')

#Making image for qualitative analysis
#Loading in 2K test image and converting to float RBG (what network was trained on)
fimg = read_image('test.png')
img = fimg.float()/255
#Downscaling image
img = img.unsqueeze(dim=0)
low_img = interpolate(img, scale_factor=1/4, mode='bicubic')

#Upscaling with Bicubic and GAN
bc_img = interpolate(low_img, scale_factor=4, mode='nearest')
gan_img = gan_model(low_img)

#Converting back to integer RGB to write pngs
bc_img = 255*((bc_img - bc_img.min())/(bc_img.max()-bc_img.min()))
bc_img = bc_img.type(torch.uint8).squeeze()
gan_img = 255*(gan_img - gan_img.min())/(gan_img.max()-gan_img.min())

gan_img = gan_img.type(torch.uint8).squeeze()

#Saving generated super-resolutions
write_png(gan_img, 'gan.png')
write_png(bc_img, 'bc.png')

#Saving identical close-up crop of all images
write_png(gan_img[:, 800:1000, 700:900], 'gan_crop.png')
write_png(bc_img[:, 800:1000, 700:900], 'bc_crop.png')
write_png(fimg[:, 800:1000, 700:900], 'img_crop.png')







