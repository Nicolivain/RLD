import os
import torch
import logging
import datetime
import seaborn as sns
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
from linearFlow import LinFlowModule
from utils import GenerativeFlow


logging.basicConfig(level=logging.INFO)
start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
save_path = "runs/tag-" + start_time + '_lin'
writer = SummaryWriter(save_path)

mu    = torch.Tensor([1, 2, 3])
sigma = torch.Tensor([4, 5, 6])
in_features = 3
bs = 1000
n = 3
lr = 0.001
epochs = 10000

prior = torch.distributions.normal.Normal(torch.zeros(in_features), torch.ones(in_features))
posterior = torch.distributions.normal.Normal(mu, sigma)

prior = torch.distributions.independent.Independent(prior, 1)
posterior = torch.distributions.independent.Independent(posterior, 1)

mod = GenerativeFlow(prior, *[LinFlowModule(in_features) for _ in range(n)])

optim = torch.optim.Adam(params=mod.parameters(), lr=lr)

for e in range(epochs):
    sample = posterior.sample((bs,))
    logprob, zs, logdet = mod.f(sample)
    optim.zero_grad()
    loss = - (logprob + logdet).mean()
    loss.backward()
    optim.step()
    writer.add_scalar('Negative Likelihood', loss.item(), e)
    print('Negative Likelihood: ', loss.item())

ds = 10000
z = prior.sample((ds,))
t = posterior.sample((ds,))
with torch.no_grad():
    y, _ = mod.invf(z)
for d in range(in_features):
    sns.kdeplot(data=y[-1][:, d], label='posterior')
    sns.kdeplot(data=t[:, d], label='target posterior')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(save_path, f'kdeplot_dim{d}.png'))
    plt.show()
