import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def hard_sigmoid(x):
    return torch.min(torch.max(x, torch.zeros_like(x)), torch.ones_like(x))


class ConcreteGate(nn.Module):

    def __init__(self, n_heads, temp=2 / 3, epsilon=0.1):
        """
        Base class of layers using L0 Norm
        :param n_heads: number of gates
        :param temp: temperature
        :param epsilon: parameter to range stretch
        """
        super(ConcreteGate, self).__init__()

        self.n_heads = n_heads
        self.loc = nn.Parameter(torch.zeros(n_heads).normal_(std=0.01))
        self.temp = temp

        self.gamma = -epsilon  # lower limit
        self.zeta = 1 + epsilon  # upper limit
        self.gamma_zeta_ratio = math.log(-self.gamma / self.zeta)

    def get_gates(self):

        gates = self._get_gates()
        #hard_gates = (gates > 0.5).float()  # binarize
        #gates += (hard_gates - gates).detach()

        return gates

    def _get_gates(self):

        if self.training:
            u = torch.zeros(self.n_heads, device=self.loc.device)
            u.uniform_()
            s = F.sigmoid((torch.log(u) - torch.log(1 - u) + self.loc) / self.temp)
            s = s * (self.zeta - self.gamma) + self.gamma # s_bar
        else:
            s = F.sigmoid(self.loc) * (self.zeta - self.gamma) + self.gamma

        return hard_sigmoid(s)

    def get_penality(self, x=None):
        # l0
        if self.training:
            penalty = F.sigmoid(self.loc - self.temp * self.gamma_zeta_ratio).sum()
        else:
            penalty = 0

        return penalty

    def forward(self, x):

        gates = self.get_gates()

        penalty = self.get_penality()

        return torch.einsum('abcd,b->abcd',x,gates), penalty / len(x)
