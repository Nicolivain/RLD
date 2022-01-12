import torch


class VAE(torch.nn.Module):
    def __init__(self, in_feature, lattent_space, hidden_layers=[]):
        super().__init__()

        self.in_features = in_feature
        self.lattent_space = lattent_space

        enc = [in_feature] + hidden_layers + lattent_space
        self.encoder_layers = torch.nn.ModuleList([torch.nn.Linear(i, o) for (i, o) in zip(enc[:-1], enc[1:])])

        self.mu  = torch.nn.Linear(lattent_space, lattent_space)
        self.logsigma = torch.nn.Linear(lattent_space, lattent_space)

        enc = enc.reverse()
        self.decoder_layers = torch.nn.ModuleList([torch.nn.Linear(i, o) for (i, o) in zip(enc[:-1], enc[1:])])

        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        for layer in self.encoder_layers:
            x = layer(x)
            x = self.relu(x)
        x = self.sigmoid(x)

        mu  = self.mu(x)
        logsigma = self.logsigma(x)
        z = mu + logsigma.exp() * torch.randn(self.lattent_space)

        for layer in self.decoder_layers:
            z = layer(z)
            z = self.relu(z)
        z = self.sigmoid(z)
        return z
