import os
import torch
import logging
import datetime
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
from GLOW import ActNorm, CouplingLayer, LuConv
from utils import GenerativeFlow

from sklearn import datasets


torch.manual_seed(1234)

logging.basicConfig(level=logging.INFO)
start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
save_path = "runs/tag-" + start_time + '_glow'
writer = SummaryWriter(save_path)

ds = ['circles', 'moons'][0]
in_features = 2
bs = 256
n = 5
lr = 0.0001
epochs = 50000

prior = torch.distributions.normal.Normal(torch.zeros(in_features), torch.ones(in_features))
prior = torch.distributions.independent.Independent(prior, 1)

convs = [LuConv(in_features=2) for i in range(n)]
norms = [ActNorm(in_features=2) for _ in range(n)]
couplings = [CouplingLayer(in_features=2, parity=i%2) for i in range(n)]
flows = []
for cv, nm, cp in zip(convs, norms, couplings):
    flows += [nm, cv, cp]
mod = GenerativeFlow(prior, *flows)

optim = torch.optim.Adam(params=mod.parameters(), lr=lr)

it_count = 0
best_loss = 10*10
best_params = None
for e in range(epochs):

    if ds == 'circles':
        data, _ = datasets.make_circles(n_samples=bs, factor=0.5, noise=0.05, random_state=0)
    elif ds == 'moons':
        data, _ = datasets.make_moons(n_samples=bs, shuffle=True, noise=0.05, random_state=0)
    else:
        raise ValueError('Unknown dataset')

    sample = torch.from_numpy(data).float()
    logprob, zs, logdet = mod.f(sample)
    optim.zero_grad()
    loss = - (logprob + logdet).mean()
    loss.backward()
    optim.step()
    writer.add_scalar('Negative Likelihood', loss.item(), it_count)
    print('Iteration: ', it_count, 'Negative Likelihood: ', loss.item(), 'Mean Logprob: ', logprob.mean().item(), 'Mean Logdet: ', logdet.mean().item())
    it_count += 1
    if loss.item() < best_loss:
        best_loss = loss.item()
        best_params = mod.state_dict()

print('Loading state dict with loss of ', best_loss)
mod.load_state_dict(best_params)

prior_sample = prior.sample((1000,))
with torch.no_grad():
    output, _ = mod.invf(prior_sample)
for i in range(len(output)):
    plt.scatter(output[i][:, 0], output[i][:, 1])
    plt.savefig(os.path.join(save_path, f'output_dist_{i}.png'))
    plt.show()
    plt.clf()
plt.scatter(data[:, 0], data[:, 1])
plt.show()
