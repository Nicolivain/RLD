import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from VAE import VAE


device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_epoch = 10
batch_size = 1000
lr = 0.001

dataset = dset.MNIST(root='data', download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))]))
dataloader = DataLoader(dataset, batch_size=batch_size)

model = VAE(784, 100, hidden_layers=[512, 254, 128], device=device, logger=None)
model.fit(dataloader, lr=lr, n_epochs=n_epoch)
