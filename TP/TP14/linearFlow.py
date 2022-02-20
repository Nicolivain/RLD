import torch
from utils import FlowModule


class LinFlowModule(FlowModule):
    def __init__(self, in_features):
        super().__init__()
        self.in_features = in_features

        self.s = torch.nn.Parameter(torch.randn(self.in_features))
        self.t = torch.nn.Parameter(torch.randn(self.in_features))

    def f(self, x):
        y = x * self.s.exp() + self.t
        log_det = self.s.sum()
        return y, log_det

    def invf(self, x):
        y = (x - self.t) * (-self.s).exp()
        log_det = 1 / self.s.sum()
        return y, log_det
