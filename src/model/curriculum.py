import numpy as np
import torch
import torch.nn as nn


class FeatureWeight(nn.Module):

    def __init__(self, number_of_features):
        super(FeatureWeight, self).__init__()

        self.feature_weights = torch.nn.Parameter(torch.zeros(number_of_features))

    def forward(self, feature):
        out = nn.functional.softmax(self.feature_weights,dim=-1) * feature  # col x col
        return out


class CurriculumConstructor:

    def __init__(self, H):
        self.H = H
        self.step = 0

    def next_ratio(self):
        top_ratio = np.power(1 / 2, self.step / self.H)
        self.step += 1
        return top_ratio

    def next(self, n_sentence):
        dynamic_selection = np.zeros(n_sentence)

        tops = int(np.ceil(len(dynamic_selection) * self.next_ratio()))
        dynamic_selection[:tops] = 1

        return 1 / tops * dynamic_selection
