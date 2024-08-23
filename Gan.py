from torch.ao.nn.quantized.modules import LeakyReLU
import torch.nn as nn
from labml import experiment

from labml.configs import calculate
from labml_helpers.module import Module
from labml_nn.original.eperiment import Configs



class Generator(Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.Sequential(
            nn.ConvTranspose2d(100,1024,3,1,0,bias=False),
            nn.BatchNorm2d(1024),#3X3 output
            nn.ReLU(True),

            nn.LazyConvTranspose2d(1024,512,3,2,0,bias=False),
            nn.BatchNorm2d(512),#7X7 output
            nn.ReLU(True),

            nn.ConvTranspose2d(512,256,4,2,1,bias=False),
            nn.BatchNorm2d(256),#14X14 output
            nn.ReLU(True),


            nn.LazyConvTranspose2d(256,1,4,2,1,bias=False),
            nn.Tanh()#28X28


        )
        self.apply(_weights_init)

        def forward(self,x):
            x=x.unsequeeze(-1).unsequeeze(-1)#changing the batch size from [size,100] to [size,100,1,1]

            x=self.layers(x)
            return x

class Discriminator(Module):
    def __init__(self):
        super().__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(1,256,4,2,1,bias=False),
            nn.LeakyReLU(0.2,inplace=True),



            nn.Conv2d(256,512,4,2,1,bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2,inplace=True)

            nn.Conv2d(512,1024,3,2,0,bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(1024,1,3,1,0,bias=False),




        )

        self.apply(_weights_init)

    def forward(self,x):
        s=self.layer(x)
        return x.view(x.shape[0],-1)

def _weights_init(m):
    classname=m.__clas__.name__
    if classname.find('Conv')!=1:
        nn.init.normal_(m.weight.data,0.0,0.02)
    elif classname.find('BAtchNorm')!=1:
        nn.init.normal_(m.weights.data,1.0,0.02)

        nn.init.constant_(m.bias.data,0)
def train():
    batch_size=64
    learning_rate=0.0002
    num_epochs=100


    #model
    generator=Generator()
    discriminator=Discriminator()

    #optim
    optim_g=optim.Adam(generator.parameters(),lr=learning_rate,betas=(0.5,0.999))
    optim_d=optim,Adam(discriminator.parameters(),lr=learning_rate,betas=(0.5,0.999))

    #training loop


    for epoch in range(num_epochs):
        for i,(images,_) in enumerate(dataloader):

            real=images
            batch_size=real.size(0)
            real_label=torch.ones(batch_size,1)
            fake_label=torch.zeros(batch_size,1)


            outputs=discriminator(real)
            d_loss_real=criterion(outputs,real_label)
            real_score=outputs

            z=torch.randn(batch_size,100)
            fake_images=generator(z)
            outputs=discriminator(fake_images.detach())
            d_loss_fake=criterion(outputs,fake_label)
            fake_score=outputs
            d_loss=d_loss_real+d_loss_fake
            optim_d.zero_grad()
            d_loss.backward()
            optim_d.step()

            #Training Generator

            outputs=discriminator(fake_images)
            g_loss=criterion(outputs,real_label)

            optim_g.zero_grad()
            g_loss.backward()
            optim_g.step()



if __name__=='__main__':
    experiment.create(name="MNIST_GAN",writers={'labml'})
    train()
