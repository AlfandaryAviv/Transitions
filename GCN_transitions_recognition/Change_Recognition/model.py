import numpy as np
import torch
from torch.nn.modules.module import Module
from torch.nn import ModuleList
from torch.nn import functional
from torch.nn import Dropout
from torch.nn import Linear
from torch.nn import Sigmoid, Softmax


class Graphs_Rec(Module):
    def __init__(self, in_features=21, hid_features=10, out_features=15, activation=functional.relu, dropout = 0):
        super(Graphs_Rec, self).__init__()
        self._layer1 = Linear(in_features, hid_features)
        self._activation = activation
        self._layer1_5 = Linear(hid_features, 10)
        self._layer2 = Linear(10, 1)
        self._dropout = Dropout(p=dropout)
        self._out_act = Sigmoid()


    def forward(self, input):
        a1 = self._layer1(input)
        h1 = self._activation(a1)
        h1 = self._dropout(h1)
        h1_5= self._layer1_5(h1)
        h1_5 = self._activation(h1_5)
        h1_5 = self._dropout(h1_5)
        a2 = self._layer2(h1_5)
        y = torch.sigmoid(a2)
        return y
