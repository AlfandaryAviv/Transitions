from itertools import product


from torch.optim import Adam, SGD
from model_runner import main_gcn

import os

import numpy as np
import networkx as nx
import pickle



class GCNTemporalCommunities:
    def __init__(self, nni=False):
        self._load_data()
        self._nni = nni

    def _load_data(self):

        graphs = []
        labels = []
        mx_s = []
        for i in range(10):
            with open(os.path.join('graphs_by_years', 'graph_' + str(i) + '.pkl'), 'rb') as f:
                g = pickle.load(f)
            with open(os.path.join('graphs_by_years', 'labels_' + str(i) + '.pkl'), 'rb') as f:
                l = pickle.load(f)
            with open(os.path.join('graphs_by_years', 'mx_' + str(i) + '.pkl'), 'rb') as f:
                mx = pickle.load(f)
            graphs.append(g)
            labels.append(l)
            mx_s.append(mx)
        # print("loaded data")
        self._adjacency_matrices = [nx.adjacency_matrix(g).tocoo() for g in graphs]
        self._feature_matrices = mx_s
        self._labels = labels

    def train(self, input_params=None):
        if input_params is None:

 

            _ = main_gcn(feature_matrices=self._feature_matrices, adj_matrices=self._adjacency_matrices,
                         labels=self._labels,
                         hid_features=40,
                         epochs=40, dropout=0.022311848472689362, lr=0.2296195, l2_pen=0.0008338610292950139,
                         temporal_pen=0.09797882823017063,
                         iterations=1, dumping_name='',
                         optimizer=Adam,
                         is_nni=self._nni)

        else:
            _ = main_gcn(feature_matrices=self._feature_matrices, adj_matrices=self._adjacency_matrices,
                         labels=self._labels,
                         hid_features=input_params["hid_features"],
                         epochs=input_params["epochs"], dropout=input_params["dropout"],
                         lr=input_params["lr"], l2_pen=input_params["regularization"],
                         temporal_pen=input_params["temporal_pen"],
                         iterations=3, dumping_name='',
                         optimizer=input_params["optimizer"],
                         is_nni=self._nni)
        return None


if __name__ == "__main__":
    # Available features: Degree ('Degree'), In-Degree ('In-Degree'), Out-Degree ('Out-Degree'),
    #                     Betweenness Centrality ('Betweenness'), BFS moments ('BFS'), motifs ('Motif_3', 'Motif_4') and
    #                     the extra features based on the motifs ('additional_features')

    # gcn_detector = GCNCliqueDetector(200, 0.5, 10, True, features=['Motif_3', 'additional_features'],
    #                                  norm_adj=True)
    # gcn_detector.train()
    # gcn_detector = GCNCliqueDetector(500, 0.5, 15, False,
    #                                  features=['Degree', 'Betweenness', 'BFS'], new_runs=0, norm_adj=True)
    # gcn_detector.train()
    # plt.rcParams.update({'figure.max_open_warning': 0})
    gg = GCNTemporalCommunities()
    gg.train()
    t = 0
