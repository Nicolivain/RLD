import torch
import torch.nn as nn
import torch.nn.functional as F
from PytorchVAE import PytorchVAE


class VAE(PytorchVAE):
    def __init__(self, in_feature, lattent_space, hidden_layers=[], device='cpu', logger=None):
        super().__init__(device=device, logger=logger)

        self.in_features = in_feature
        self.lattent_space = lattent_space

        enc = [in_feature] + hidden_layers
        self.encoder_layers = torch.nn.ModuleList([torch.nn.Linear(i, o) for (i, o) in zip(enc[:-1], enc[1:])])

        self.mu  = torch.nn.Linear(enc[-1], lattent_space)
        self.logvar = torch.nn.Linear(enc[-1], lattent_space)

        enc.reverse()
        enc = [self.lattent_space] + enc
        self.decoder_layers = torch.nn.ModuleList([torch.nn.Linear(i, o) for (i, o) in zip(enc[:-1], enc[1:])])

        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

        self.to(self.device)

    def encode(self, x):
        for layer in self.encoder_layers:
            x = layer(x)
            x = self.relu(x)

        mu  = self.mu(x)
        logvar = self.logvar(x)

        std = torch.exp(0.5 * logvar)  # standard deviation
        eps = torch.randn_like(std)  # `randn_like` as we need the same size
        z = mu + (eps * std)  # sampling as if coming from the input spac

        return z, mu, logvar

    def decode(self, z):
        for layer in self.decoder_layers:
            z = layer(z)
        z = self.sigmoid(z)
        return z

    def forward(self, x):
        z, _, _ = self.encode(x)
        xhat = self.decode(z)
        return xhat

    @staticmethod
    def lattent_reg(mu, logvar):
        # KL divergence for lattent space regularization
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return kld
