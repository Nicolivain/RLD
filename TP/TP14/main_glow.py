import os
import torch
import logging
import datetime
import seaborn as sns
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
from GLOW import GlowModule
from utils import GenerativeFlow

from sklearn import datasets


logging.basicConfig(level=logging.INFO)
start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
save_path = "runs/tag-" + start_time + '_glow'
writer = SummaryWriter(save_path)

data = ['circles', 'moons'][0]
n_samples = 10000
mu    = torch.Tensor([0, 0, 0])
sigma = torch.Tensor([1, 2, 3])
in_features = 2
bs = 1000
n = 10
lr = 0.0001
epochs = 10

if data == 'circles':
    data, _ = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05, random_state=0)
elif data == 'moons':
    data, _ = datasets.make_moons(samples=n_samples, shuffle=True, noise=0.05, random_state=0)
else:
    raise ValueError('Unknown dataset')

prior = torch.distributions.normal.Normal(torch.zeros(in_features), torch.ones(in_features))
posterior = torch.distributions.normal.Normal(mu, sigma)

prior = torch.distributions.independent.Independent(prior, 1)
posterior = torch.distributions.independent.Independent(posterior, 1)

mod = GenerativeFlow(prior, *[GlowModule(in_features) for _ in range(n)])

optim = torch.optim.Adam(params=mod.parameters(), lr=lr)

for e in range(epochs):
    for k in range(n_samples//bs):
        sample = torch.from_numpy(data[k*bs:(k+1)*bs, :]).float()
        logprob, zs, logdet = mod.f(sample)
        optim.zero_grad()
        loss = - (logprob + logdet).mean()
        loss.backward()
        optim.step()
        writer.add_scalar('Negative Likelihood', loss.item(), e)
        print('Negative Likelihood: ', loss.item())


prior_sample = prior.sample((n_samples,))
with torch.no_grad():
    output, _ = mod.invf(prior_sample)
plt.scatter(output[-1][:, 0], output[-1][:, 1])
plt.show()
plt.clf()
plt.scatter(data[:, 0], data[:, 1])
plt.show()
