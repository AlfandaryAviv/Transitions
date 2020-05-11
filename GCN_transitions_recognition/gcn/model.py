import numpy as np
import torch
from torch.nn.modules.module import Module
from torch.nn import ModuleList
from torch.nn import functional
from torch.nn import Dropout
from torch_geometric.nn import GCNConv


class GCN(Module):
    def __init__(self, in_features=4, hid_features=10, out_features=15, activation=functional.relu, dropout = 0):
        super(GCN, self).__init__()
        self._layer1 = GCNConv(in_features, hid_features)
        self._activation = activation
        self._layer2 = GCNConv(hid_features, out_features)
        self._dropout = Dropout(p=dropout)

    @staticmethod
    def adj_to_coo(adj, device):
        edges_tensor = np.vstack((adj.row, adj.col))
        return torch.tensor(edges_tensor, dtype=torch.long).to(device)

    def forward(self, x, adj):
        adj = self.adj_to_coo(adj, x.device)
        x = self._layer1(x, adj)
        z = self._activation(x)
        z = self._dropout(z)
        h = self._layer2(z, adj)
        h = self._dropout(h)
        adj=adj.to("cpu")
        x=x.to("cpu")
        return z, torch.softmax(h, dim=1)
