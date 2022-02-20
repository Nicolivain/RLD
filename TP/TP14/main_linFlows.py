import torch
from linearFlow import LinFlowModule
from utils import GenerativeFlow

mu    = torch.Tensor([0, 0, 0])
sigma = torch.Tensor([1, 2, 3])
in_features = 3
bs = 100
n = 3
lr = 0.001
epochs = 50

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
