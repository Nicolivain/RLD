import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from VAE import VAE


device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_epoch = 10
batch_size = 1000
lr = 0.001

dataset = dset.MNIST(root='data', download=True, transform=transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=batch_size)

model = VAE(784, 100, hidden_layers=[512, 254, 128], device=device)

criterion = torch.nn.BCELoss()
opt = torch.optim.Adam(params=model.parameters(), lr=lr)

for epoch in range(n_epoch):
    epoch_kll  = 0
    epoch_bce  = 0
    epoch_loss = 0
    for batch in dataloader:
        batch_x = batch[0].to(device)
        batch_x = batch_x.view(batch_x.shape[0], -1)

        z, mu, std = model.encode(batch_x)
        KLloss = model.kl_divergence(z, mu, std)

        xhat = model.decode(z)

        bce = criterion(xhat, batch_x)

        loss = bce + KLloss
        opt.zero_grad()
        loss.backward()
        opt.step()

        epoch_kll += KLloss.item()
        epoch_bce += bce.item()
        epoch_loss += loss.item()

    epoch_loss /= len(dataloader)
    epoch_bce  /= len(dataloader)
    epoch_kll  /= len(dataloader)

    print(f'Epoch {epoch}: Loss {epoch_loss}, BCE {epoch_bce}, KLL {epoch_kll}')





