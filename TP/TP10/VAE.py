import torch
import torch.nn as nn
import torch.nn.functional as F
from PytorchVAE import PytorchVAE


class VAE(PytorchVAE):
    def __init__(self, in_feature, lattent_space, hidden_layers=[], device='cpu', logger=None, tag=''):
        super().__init__(device=device, logger=logger, tag=tag)

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


class ConvVAE(PytorchVAE):
    def __init__(self, image_shape, in_channels, lattent_space, hidden_filters=[], kernel_size=(2, 2), strides=(1, 1), device='cpu', logger=None, tag=''):
        super().__init__(device=device, logger=logger, tag=tag)

        self.image_shape = image_shape
        self.n_pixel = image_shape[0] * image_shape[1]
        self.in_channels = in_channels
        self.lattent_space = lattent_space

        self.ks = kernel_size
        self.strides = strides

        enc = [in_channels] + hidden_filters + [1]
        self.encoder_layers = torch.nn.ModuleList([torch.nn.Conv2d(i, o, stride=self.strides, kernel_size=self.ks) for (i, o) in zip(enc[:-1], enc[1:])])

        h_out, w_out = self.get_output_shape(image_shape[0], image_shape[1], len(enc)-1)
        self.h_out = h_out
        self.w_out = w_out

        self.mu           = torch.nn.Linear(h_out * w_out, self.lattent_space)
        self.logvar       = torch.nn.Linear(h_out * w_out, self.lattent_space)
        self.decode_ready = torch.nn.Linear(self.lattent_space, h_out * w_out)

        enc.reverse()
        self.decoder_layers = torch.nn.ModuleList([torch.nn.ConvTranspose2d(i, o, stride=self.strides, kernel_size=self.ks) for (i, o) in zip(enc[:-1], enc[1:])])

        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

        self.to(self.device)

    def encode(self, x):
        for layer in self.encoder_layers:
            x = layer(x)
            x = self.relu(x)
        x = x.view(x.shape[0], -1)

        mu  = self.mu(x)
        logvar = self.logvar(x)

        std = torch.exp(0.5 * logvar)  # standard deviation
        eps = torch.randn_like(std)  # `randn_like` as we need the same size
        z = mu + (eps * std)  # sampling as if coming from the input spac

        return z, mu, logvar

    def decode(self, z):
        z = self.decode_ready(z).view(z.shape[0], 1, self.h_out, self.w_out)
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

    def get_output_shape(self, hin, win, n_conv):
        for k in range(n_conv):
            hin = (hin - (self.ks[0] - 1) - 1)/self.strides[0] + 1
            win = (win - (self.ks[1] - 1) - 1)/self.strides[1] + 1
        return int(hin), int(win)
