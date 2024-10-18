from models import Generator, Discriminator

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.nn.functional import interpolate
from torchvision.models import vgg19

import matplotlib.pyplot as plt

'''
Custom weight intialization scheme for generator and discriminator for small weights
Code snippet taken from https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
'''
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm') != -1: 
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)    

'''
Content loss: measures difference between output's and reference's features in a pretrained VGG19 net
Code for extracting intermediate layers of VGG from https://discuss.pytorch.org/t/how-to-extract-features-from-intermediate-layers-of-vgg16/76571
'''
def content_loss(output, ref, network, layer_nums):
    #output, ref = output.cpu(), ref.cpu()
    features = list(network.children())[0]
    loss = 0
    for lnum in layer_nums:
        subnet = features[:lnum]
        output_feat, ref_feat = subnet(output), subnet(ref)
        loss += torch.sum(torch.mean((output_feat-ref_feat)**2, dim=(1,2,3)))

    return loss

'''
Main body of training script begins here
'''
#Training hyperparameters
lr_disc = .0001
lr_gen = .0001
epochs = 1500

#Building a dataloader for the BSDS200 dataset
train_ds = ImageFolder('../data/BSDS300/images/train', transform=transforms.Compose([transforms.RandomCrop(96), transforms.ToTensor()]))
train_dl = DataLoader(train_ds, batch_size=16, shuffle=False, num_workers=4)
print("Dataset ready")

#Configuring cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on device: {device}")

#Initializing models
generator = Generator(upscale_mag=4).to(device)
discriminator = Discriminator(96).to(device)


    
#initliazing separate optimizers for disciminator and generator
dopt = torch.optim.Adam(discriminator.parameters(), lr=lr_disc)
gopt = torch.optim.Adam(generator.parameters(), lr=lr_gen)

#Loss rate schedulers to reduce rate after first 1000 epochs
dsched = torch.optim.lr_scheduler.StepLR(dopt, step_size=1000)
gsched = torch.optim.lr_scheduler.StepLR(gopt, step_size=1000)

#Using custom initialization scheme for small weights
generator.apply(weights_init)
discriminator.apply(weights_init)


adv_loss = nn.BCELoss()

#Loss network setup for content loss
loss_net = vgg19(pretrained=True).to(device)
loss_layers = [3, 8, 17] #Selected layers to extract features from, all pre-activation conv

#Main training loop
for epoch in range(epochs):
    disc_losses = []
    gen_losses = []
    for i, batch in enumerate(train_dl):

            
        discriminator.zero_grad()

        #Computing discriminator loss
        high_imgs, _ = batch
        high_imgs = high_imgs.to(device)
        low_imgs = interpolate(high_imgs, scale_factor=1/generator.upscale_mag, mode='bicubic')

        gen_imgs = generator(low_imgs)
        rate_gen, rate_high = discriminator(gen_imgs.detach()), discriminator(high_imgs)
        high_labels = torch.ones(rate_high.shape).to(device)
        gen_labels = torch.zeros(rate_gen.shape).to(device)
        
        disc_loss = (adv_loss(rate_gen, gen_labels) + adv_loss(rate_high, high_labels)).cpu()
        disc_loss.backward()
        dopt.step()

        generator.zero_grad()
        
        #Computing generator loss
        rate_gen = discriminator(gen_imgs)
        gen_loss = (content_loss(gen_imgs, high_imgs, loss_net, loss_layers) + .01 *  adv_loss(rate_gen, high_labels)).cpu()
        gen_loss.backward()
        gopt.step()
            
        gen_losses.append(gen_loss)
        disc_losses.append(disc_loss)
    
    dsched.step()
    gsched.step()

    disc_loss, gen_loss = sum(disc_losses)/len(disc_losses), sum(gen_losses)/len(gen_losses)
    print(f"Epoch {epoch}: \t DISC LOSS: {disc_loss:.5f} \t GEN LOSS: {gen_loss:.5f}")
    if epoch % 10 == 0:
        torch.save(generator.cpu(), 'generator_{epoch}.torch')
        torch.save(discriminator.cpu(), 'discriminator_{epoch}.torch')

generator, discriminator = generator.cpu(), discriminator.cpu()
torch.save(generator, 'generator_final.torch')
torch.save(discriminator, 'discriminator_final.torch')
