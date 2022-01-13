import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from VAE import VAE


device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_epoch = 20
batch_size = 64
lattent = 16
layers = [512]
lr = 0.0001

train_dataset = dset.MNIST(root='data', download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))]), train=True)
dataloader = DataLoader(train_dataset, batch_size=batch_size)
v_dataset = dset.MNIST(root='data', download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))]), train=False)
v_dataloader = DataLoader(v_dataset, batch_size=batch_size)

model = VAE(784, lattent, hidden_layers=layers, device=device, logger=None)
model.fit(dataloader, validation_data=v_dataloader, lr=lr, n_epochs=n_epoch, save_images_freq=1)
