import torch
import torchvision.datasets as dset
from torch.utils.data import DataLoader
from VAE import VAE


device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 1000

dataset = dset.MNIST(root='data', download=True)
dataloader = DataLoader(dataset, batch_size=batch_size)




