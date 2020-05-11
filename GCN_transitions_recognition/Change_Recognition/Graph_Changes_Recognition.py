from itertools import product
import torch
from torch.optim import Adam, SGD
from model_runner import main_gcn
import os
import numpy as np
import networkx as nx
import pickle
import matplotlib.pyplot as plt


class Graph_Changes_Rec():
    def __init__(self, nni=False):
        self._load_data()
        self._nni = nni

    def _load_data(self):
        with open('input_7_tag.pkl', 'rb') as f:
            input_data = pickle.load(f)
            #labels_per_writer_7_tag
        with open('labels_per_writer_7_tag.pkl', 'rb') as f:
            labels = pickle.load(f)
        # with open('input_7_tag_real_tags.pkl',"rb") as f:
        #     real = pickle.load(f)

        self._input_data = input_data
        self._labels = labels

    def train(self, input_params=None):
        if input_params is None:

            _ = main_gcn(input_data=self._input_data,
                         labels=self._labels,
                         hid_features=20,
                         epochs=100, dropout=0.01, lr=0.01, l2_pen=0.005,
                         iterations=1, dumping_name='',
                         optimizer=Adam,
                         is_nni=self._nni)


        else: #This part is for the nni, todo change according to above parameters name
            _ = main_gcn(input_data=self._input_data,
                         labels=self._labels,
                         hid_features=input_params["hid_features"],
                         epochs=input_params["epochs"], dropout=input_params["dropout"],
                         lr=input_params["lr"], l2_pen=input_params["regularization"],
                         iterations=3, dumping_name='',
                         optimizer=input_params["optimizer"],
                         is_nni=self._nni)
        return None



if __name__ == "__main__":
    torch.set_printoptions(precision=10)
    gg = Graph_Changes_Rec()
    gg.train()
    t = 0
