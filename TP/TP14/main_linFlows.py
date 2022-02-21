import torch
import logging
import datetime
from torch.utils.tensorboard import SummaryWriter
from linearFlow import LinFlowModule
from utils import GenerativeFlow

logging.basicConfig(level=logging.INFO)
start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
save_path = "TP/TP14/runs/tag-" + start_time + '_lin'
writer = SummaryWriter(save_path)

mu    = torch.Tensor([0, 0, 0])
sigma = torch.Tensor([1, 2, 3])
in_features = 3
bs = 1000
n = 3
lr = 0.002
epochs = 1000

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
    print(f'Negative Likelihood: ', loss.item())
