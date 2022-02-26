import torch
from utils import FlowModule, MLP, logabs


class ActNorm(FlowModule):
    def __init__(self, in_features):
        super().__init__()
        self.in_features = in_features

        self.s = torch.nn.Parameter(torch.randn(1, self.in_features, requires_grad=True))
        self.t = torch.nn.Parameter(torch.randn(1, self.in_features, requires_grad=True))
        self.init_done = False

    def f(self, x):
        if not self.init_done:
            self.s.data = (-torch.log(x.std(dim=0, keepdim=True))).detach()
            self.t.data = (-(x * torch.exp(self.s)).mean(dim=0, keepdim=True)).detach()
            self.init_done = True
        y = x * self.s.exp() + self.t
        log_det = self.s.sum()
        return y, log_det

    def invf(self, x):
        y = (x - self.t) * (-self.s).exp()
        log_det = -self.s.sum()
        return y, log_det


class CouplingLayer(FlowModule):
    def __init__(self, in_features, parity):
        super().__init__()
        self.in_features = in_features
        self.scale = MLP(self.in_features // 2, self.in_features - self.in_features // 2, 100)
        self.shift = MLP(self.in_features // 2, self.in_features - self.in_features // 2, 100)
        self.parity = parity

    def f(self, x):
        x0, x1 = x[:, :self.in_features//2], x[:, self.in_features//2:]
        if self.parity:
            x0, x1 = x1, x0
        s = self.scale(x0)
        t = self.shift(x0)
        z0 = x0
        z1 = torch.exp(s) * x1 + t  #
        if self.parity:
            z0, z1 = z1, z0
        z = torch.cat([z0, z1], dim=1)
        log_det = torch.sum(s, dim=1)
        return z, log_det

    def invf(self, z):
        z0, z1 = z[:, :self.in_features//2], z[:, self.in_features//2:]
        if self.parity:
            z0, z1 = z1, z0
        s = self.scale(z0)
        t = self.shift(z0)
        x0 = z0
        x1 = (z1 - t) * torch.exp(-s)
        if self.parity:
            x0, x1 = x1, x0
        x = torch.cat([x0, x1], dim=1)
        log_det = torch.sum(-s, dim=1)
        return x, log_det


class LuConv(FlowModule):
    def __init__(self, in_features):
        super().__init__()
        self.in_features = in_features

        weight = torch.randn(self.in_features, self.in_features)
        weight = torch.nn.init.orthogonal_(weight)
        d = torch.lu(weight)
        w_p, w_l, w_u = torch.lu_unpack(d[0], d[1])
        w_s = torch.diag(w_u)
        w_u = torch.triu(w_u, 1)
        u_mask = torch.triu(torch.ones_like(w_u), 1)
        l_mask = u_mask.T

        self.register_buffer("w_p", w_p)
        self.register_buffer("u_mask", u_mask)
        self.register_buffer("l_mask", l_mask)
        self.register_buffer("s_sign", torch.sign(w_s))
        self.register_buffer("l_eye", torch.eye(l_mask.shape[0]))
        self.w_l = torch.nn.Parameter(w_l)
        self.w_s = torch.nn.Parameter(logabs(w_s))
        self.w_u = torch.nn.Parameter(w_u)

    def _get_weight(self):
        weight = self.w_p @ (self.w_l * self.l_mask + self.l_eye) @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        return weight

    def f(self, x):
        y = x @ self._get_weight()
        log_det = self.w_s.sum()
        return y, log_det

    def invf(self, x):
        y = x @ self._get_weight().inverse()
        log_det = -self.w_s.sum()
        return y, log_det


class GlowModule(FlowModule):
    def __init__(self, in_features, parity):
        super().__init__()
        self.in_features = in_features

        self.act_norm = ActNorm(self.in_features)
        self.coupling = CouplingLayer(self.in_features, parity)
        self.luconv   = LuConv(self.in_features)

    def f(self, x):
        # linear
        y, log_det = self.act_norm.f(x)
        y, nld  = self.luconv.f(y)
        y, nnld = self.coupling.f(y)
        log_det = log_det + nld + nnld
        return y, log_det

    def invf(self, x):
        # linear
        y, log_det = self.act_norm.invf(x)
        y, nld  = self.luconv.invf(y)
        y, nnld = self.coupling.invf(y)
        log_det = log_det + nld + nnld
        return y, log_det
