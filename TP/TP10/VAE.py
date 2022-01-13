import torch
from PytorchVAE import PytorchVAE


class VAE(PytorchVAE):
    def __init__(self, in_feature, lattent_space, hidden_layers=[], device='cpu', logger=None):
        super().__init__(device=device, logger=logger)

        self.in_features = in_feature
        self.lattent_space = lattent_space

        enc = [in_feature] + hidden_layers + [lattent_space]
        self.encoder_layers = torch.nn.ModuleList([torch.nn.Linear(i, o) for (i, o) in zip(enc[:-1], enc[1:])])

        self.mu  = torch.nn.Linear(lattent_space, lattent_space)
        self.logsigma = torch.nn.Linear(lattent_space, lattent_space)

        enc.reverse()
        self.decoder_layers = torch.nn.ModuleList([torch.nn.Linear(i, o) for (i, o) in zip(enc[:-1], enc[1:])])

        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

        self.to(self.device)

    def encode(self, x):
        for layer in self.encoder_layers:
            x = layer(x)
            x = self.relu(x)
        x = self.sigmoid(x)

        mu  = self.mu(x)
        logsigma = self.logsigma(x)
        z = mu + logsigma.exp() * torch.randn(self.lattent_space, device=self.device)

        return z, mu, logsigma.exp()

    def decode(self, z):
        for layer in self.decoder_layers:
            z = layer(z)
            z = self.relu(z)
        z = self.sigmoid(z)
        return z

    def forward(self, x):
        z, _, _ = self.encode(x)
        xhat = self.decode(z)
        return xhat

    @staticmethod
    def lattent_reg(z, mu, std):
        # KL divergence for lattent space regularization
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl.mean()
