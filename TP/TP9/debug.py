mport numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.utils.data
import torchvision.datasets as dset
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn



if __name__ == '__main__':

    image_size = 64
    dataset = dset.MNIST(root="data",
                         transform=transforms.Compose([
                             transforms.Resize(image_size),
                             transforms.CenterCrop(image_size),
                             transforms.ToTensor(),
                             transforms.Normalize(0.5, 0.5),
                         ]))

    import matplotlib.pyplot as plt
    import os

    seed=0
    torch.manual_seed(seed)
    np.random.seed(seed)
    device=0
    if device>=0 and torch.cuda.is_available():
      cudnn.benchmark = True
      torch.cuda.device(device)
      torch.cuda.manual_seed(seed)
    else:
      device=-1

    batch_size = 128
    workers = 2

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=workers)


    nc = 1  # Nombre de canaux de l'entrée
    ndf = 64  # Facteur du nombre de canaux de sortie des différentes couches de convolution


    # Initialisation recommandee pour netG et netD dans DCGAN
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    class Discriminator(nn.Module):
        def __init__(self):
            super(Discriminator, self).__init__()
            self.main = nn.Sequential(
                # input is (nc) x 64 x 64
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 32 x 32
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 16 x 16
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x 8 x 8
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*8) x 4 x 4
                nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )

        def forward(self, input):
            return self.main(input)


    netD = Discriminator().to(device)
    netD.apply(weights_init)
    print(netD)


    nz = 100  # Taille du vecteur z donné en entrée du générateur
    ngf = 64  # Facteur du nombre de canaux de sortie des différentes couches de deconvolution

    class Generator(nn.Module):
        def __init__(self):
            super(Generator, self).__init__()
            self.main = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                # state size. (ngf*8) x 4 x 4
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                # state size. (ngf*4) x 8 x 8
                nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                # state size. (ngf*2) x 16 x 16
                nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                # state size. (ngf) x 32 x 32
                nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
                nn.Tanh()
                # state size. (nc) x 64 x 64
            )

        def forward(self, input):
            return self.main(input)

    netG = Generator().to(device)
    netG.apply(weights_init)
    print(netG)


    n_epochs = 10
    lr = 1e-3
    criterion = torch.nn.BCELoss()


    def train(netG, netD, dataloader):
        optD = torch.optim.Adam(params=netD.parameters(), lr=lr)
        optG = torch.optim.Adam(params=netG.parameters(), lr=lr)

        avg_lossD_real = []
        avg_lossD_fake = []
        avg_lossD = []
        avg_lossG = []

        for epoch in range(n_epochs):
            epoch_lossD_real = 0
            epoch_lossD_fake = 0
            epoch_lossD = 0
            epoch_lossG = 0
            for idx, batch in enumerate(dataloader):
                # Learning D
                # real data
                netD.zero_grad()
                real_data = batch[0].to(device)
                disc = netD(real_data).view(-1)
                label = torch.ones(real_data.size(0), device=device)
                lossD_real = criterion(disc, label)
                lossD_real.backward()

                # generate the negative (ie fake) data for D
                noise = torch.randn(real_data.size(0), nz, 1, 1, device=device)
                fake = netG(noise)  # we don't use no_grad because we will backprop on this later
                disc = netD(fake.detach()).view(-1)
                label = torch.zeros(real_data.size(0), device=device)
                lossD_fake = criterion(disc, label)
                lossD_fake.backward()

                lossD = lossD_real + lossD_fake
                optD.step()

                # Learning G
                netG.zero_grad()
                label = torch.ones(real_data.size(0), device=device)
                disc = netD(fake).view(-1)
                lossG = criterion(disc, label)
                lossG.backward()
                optG.step()

                epoch_lossD_real += lossD_real.item()
                epoch_lossD_fake += lossD_fake.item()
                epoch_lossD += lossD.item()
                epoch_lossG += lossG.item()

            print(f'Epoch {epoch}: Average Discriminator Loss: {epoch_lossD } Average Generator Loss: {epoch_lossG } realDloss: {epoch_lossD_real }, fakeDloss{epoch_lossD_fake}')

            avg_lossG.append((epoch_lossG / len(dataloader)))
            avg_lossD.append((epoch_lossD / (2*len(dataloader))))
            avg_lossD_fake.append((epoch_lossD_fake / len(dataloader)))
            avg_lossD_real.append((epoch_lossD_real / len(dataloader)))

        return avg_lossG, avg_lossD, avg_lossD_real, avg_lossD_fake


    train(netG, netD, dataloader)

    noise = torch.randn(64, nz, 1, 1, device=device)
    with torch.no_grad():
        netG.eval()
        fake = netG(noise).detach().cpu()
    img = vutils.make_grid(fake, padding=2, normalize=True)
    plt.figure(figsize=(15, 15))
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img.cpu(), (1, 2, 0)))
    plt.savefig("fake.png")
