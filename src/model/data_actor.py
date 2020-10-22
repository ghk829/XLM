import torch
import torch.nn as nn


class BaseActor(nn.Module):

    def __init__(self, number_of_domains):
        super(BaseActor, self).__init__()

        self.bias = torch.nn.Linear(number_of_domains, 1)

        for p in self.bias.parameters():
            p.data.fill_(0.)

    def forward(self, feature):
        logits = self.bias.weight * feature
        return logits
