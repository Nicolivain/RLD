import torch
import datetime
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from TP.TP10.VAE import VAE, ConvVAE

start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_epoch = 50
batch_size = 64
lattent = 16
layers = [512]
lr = 0.0001


train_dataset = dset.MNIST(root='data', download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))]), train=True)
dataloader = DataLoader(train_dataset, batch_size=batch_size)
v_dataset = dset.MNIST(root='data', download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))]), train=False)
v_dataloader = DataLoader(v_dataset, batch_size=batch_size)

writer = SummaryWriter("TP/TP10/runs/tag-" + start_time + '_VAE')
model = VAE(784, lattent, hidden_layers=layers, device=device, logger=writer.add_scalar, tag='VAE')
model.fit(dataloader, validation_data=v_dataloader, lr=lr, n_epochs=n_epoch, save_images_freq=5)

layers = [4]  # we use less filters

train_dataset = dset.MNIST(root='data', download=True, transform=transforms.Compose([transforms.ToTensor()]), train=True)
dataloader = DataLoader(train_dataset, batch_size=batch_size)
v_dataset = dset.MNIST(root='data', download=True, transform=transforms.Compose([transforms.ToTensor()]), train=False)
v_dataloader = DataLoader(v_dataset, batch_size=batch_size)

writer = SummaryWriter("TP/TP10/runs/tag-" + start_time + '_ConvVAE')
model = ConvVAE(image_shape=(28, 28), in_channels=1, lattent_space=lattent, hidden_filters=layers, device=device, logger=writer.add_scalar, tag='ConvVAE')
model.fit(dataloader, validation_data=v_dataloader, lr=lr, n_epochs=n_epoch, save_images_freq=5)
