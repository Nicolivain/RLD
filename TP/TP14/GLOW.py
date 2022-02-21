import torch
from utils import FlowModule, MLP


class GlowModule(FlowModule):
    def __init__(self, in_features):
        super().__init__()
        self.in_features = in_features

        # act norm
        self.s = torch.randn(self.in_features)
        self.t = torch.randn(self.in_features)
        with torch.no_grad():
            self.s = torch.nn.Parameter((self.s - self.s.mean()) / self.s.std())
            self.t = torch.nn.Parameter((self.t - self.t.mean()) / self.t.std())

        # coupling layer
        self.scale = MLP(self.in_features//2, self.in_features - self.in_features//2, 100)
        self.shift = MLP(self.in_features//2, self.in_features - self.in_features//2, 100)

        # lu convolution
        init = torch.nn.init.orthogonal_(torch.randn(self.in_features, self.in_features))
        d = torch.lu(init)
        self.p, l, u = torch.lu_unpack(d[0], d[1])
        self.ud = torch.nn.Parameter(torch.diag(u))
        self.u = torch.nn.Parameter(torch.triu(u, 1))
        self.l = torch.nn.Parameter(torch.tril(u, -1))
        self.id = torch.eye(self.in_features)

    def _get_weight(self):
        return self.p @ (self.l + self.id) @ (self.u + torch.diag(self.ud))

    def f(self, x):
        # linear
        y = x * self.s.exp() + self.t
        log_det = self.s.sum()

        # coupling
        ync, yc = y[:, :y.shape[1]//2], y[:, y.shape[1]//2:]
        sc = self.scale(ync)
        sh = self.shift(ync)
        yc = yc * sc.exp() + sh
        y = torch.cat([ync, yc], axis=1)
        log_det += sc.sum()

        # conv
        y = y @ self._get_weight()
        log_det += self.ud.abs().log().sum()

        return y, log_det

    def invf(self, x):
        # linear
        y = (x - self.t) * (-self.s).exp()
        log_det = self.s.sum()

        # coupling layer
        ync, yc = y[:, :y.shape[1] // 2 + 1], y[:, y.shape[1] // 2 + 1:]
        sc = self.scale(ync)
        sh = self.shift(y[:, y.shape[1] // 2 + 1:])
        yc = (yc - sh) * (-sc).exp()
        y = torch.cat([ync, yc], axis=1)
        log_det += sc.sum()

        # conv
        y = y @ self._get_weight().inverse()
        log_det += self.ud.abs().log().sum()

        log_det = 1 / log_det
        return y, log_det
