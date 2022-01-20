import torch
import logging
import datetime
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torch.utils.data
import torchvision.datasets as dset
import torchvision.utils as vutils

from DCGAN_32 import DCGAN_32


logging.basicConfig(level=logging.INFO)
start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter("TP/TP9/runs/tag-" + start_time + '_MNIST')

image_size = 32

nc  = 1
ndf = 64
nz  = 100
ngf = 64

dataset = dset.MNIST(root="../../../data",
                     transform=transforms.Compose([
                         transforms.Resize(image_size),
                         transforms.CenterCrop(image_size),
                         transforms.ToTensor(),
                         transforms.Normalize(0.5, 0.5),
                     ]))
print(dataset)
seed = 0
torch.manual_seed(seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

batch_size = 128
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

model = DCGAN_32(n_channels=nc, lattent_space_size=nz, n_disc_filters=ndf, n_gen_filters=ngf, device=device)

model.fit(dataloader, n_epochs=5, lr=1e-3)

noise = torch.randn(64, nz, 1, 1, device=device)
fake = model.generate(noise)
img = vutils.make_grid(fake, padding=2, normalize=True)
plt.figure(figsize=(15, 15))
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img.cpu(), (1, 2, 0)))
plt.savefig("fake.png")
