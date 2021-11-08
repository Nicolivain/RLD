import torch


def batched_dkl(p, q):
    """
    :param p: a tensor of shape (batch, dist) representing a distribution distribution (sum=1)
    :param q: a tensor of shape (batch, dist) representing a distribution distribution (sum=1)
    :return: KL divergence of P relative to Q
    """
    assert len(p.shape) == 2, 'p should be 2 dimensional with shape (n_sample, distrib)'
    assert p.shape == q.shape, 'p and q should have the same shape'

    n = p.shape[0]
    prod = torch.log(p/q) * p
    return prod.reshape(-1).sum() / n
