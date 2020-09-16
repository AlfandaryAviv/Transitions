import time
from itertools import product
from loggers import CSVLogger, PrintLogger, FileLogger, multi_logger
import torch
from torch.optim import Adam
import os
import networkx as nx
import pickle
import numpy as np
import logging
from model_runner import ModelRunner,execute_runners
import cProfile

Dataset_name = 'Tmall'                  #'Tmall','DBLP','IMDB'
Time_inds = 9
Hid_size = [10]
Epochs = 3
Dropout = [0.3]
LR = [0.01]
Regularization = [0.002]
Temporal_pen = [0.002]
Optimizer = Adam
Iterations = 1
Number_Of_Classes = 2                   # 2 for 'Tmall', 15 for 'DBLP', 11 for 'IMDB'
Is_NNI = False
Train_test_split = 'bipartite'          #'bipartite', 'all_labeled', 'partialy_labeled'
Bipartite_products = 200                #200 either None
loss_weights = 'sqrt(N/Nj)'             #None #'1/Njs', 'sqrt(N/Nj)'

class GCNTemporalCommunities(object):
    def __init__(self, nni=False):
        self._nni = nni
        self._device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self._adjacency_matrices = None
        self._feature_matrices = None
        self._labels = None

    def load_data(self):
        graphs = []
        labels = []
        mx_s = []
        for i in range(Time_inds):
            with open(os.path.join('dataset',Dataset_name,'input', 'graph_' + str(i) + '.pkl'), 'rb') as f:
                g = pickle.load(f)
            with open(os.path.join('dataset',Dataset_name,'input', 'labels_' + str(i) + '.pkl'), 'rb') as f:
                l = pickle.load(f)
            with open(os.path.join('dataset',Dataset_name,'input', 'mx_' + str(i) + '.pkl'), 'rb') as f:
                mx = pickle.load(f)
            graphs.append(g)
            labels.append(l)
            mx_s.append(mx)

        self._adjacency_matrices = [nx.adjacency_matrix(g).tocoo() for g in graphs]
        self._feature_matrices = mx_s
        self._labels = labels

    def fix_logger(self, dumping_name):
        # os.getcwd() returns current working directory of a process
        products_path = os.path.join(os.getcwd(), 'dataset', Dataset_name, "logs", dumping_name,
                                     time.strftime("%Y%m%d_%H%M%S"))
        if not os.path.exists(products_path):
            os.makedirs(products_path)

        logger = multi_logger([PrintLogger("MyLogger", level=logging.DEBUG),
                               FileLogger("results_%s" % dumping_name,
                                          path=products_path, level=logging.INFO)], name=None)
        return logger

    def prep_trial(self, input_params, grid_logger, grid_logger_avg):
        runners = []

        for it in range(input_params['iterations']):

            # train and test split
            if Train_test_split == 'bipartite':
                person_data = np.delete(np.arange(len(self._labels[0])), np.arange(Bipartite_products))
                rand_test_indices = np.random.choice(person_data, round(len(person_data) * 0.9), replace=False)
                rand_train_indices = np.delete(np.arange(len(self._labels[0])), rand_test_indices)
            else:
                rand_test_indices = np.random.choice(len(self._labels[0]), round(len(self._labels[0]) * 0.9), replace=False)
                rand_train_indices = np.delete(np.arange(len(self._labels[0])), rand_test_indices)

            train = [[k for k in rand_train_indices if self._labels[j][k] != -1] for j in range(len(self._labels))]
            test = [[k for k in rand_test_indices if self._labels[j][k] != -1] for j in range(len(self._labels))]
            test_labels = [torch.tensor([self._labels[j][k] for k in rand_test_indices if self._labels[j][k] != -1],
                                        dtype=torch.double).to(self._device) for j in range(input_params['time_inds'])]
            train_labels = [torch.tensor([self._labels[j][k] for k in rand_train_indices if self._labels[j][k] != -1],
                                        dtype=torch.double).to(self._device) for j in range(input_params['time_inds'])]

            input_params['it_num'] = it
            input_params['activation'] = torch.nn.functional.relu
            input_params['train_person'] = rand_train_indices
            input_params['test_person'] = rand_test_indices
            input_params['training_inds'] = train
            input_params['test_inds'] = test
            input_params['training_labels'] = train_labels
            input_params['test_labels'] = test_labels
            input_params['adj_matrices'] = self._adjacency_matrices
            input_params['feature_matrices'] = self._feature_matrices

            dumping_name = ""
            logger = self.fix_logger(dumping_name)
            runner = ModelRunner(input_params, logger=logger)
            runners.append(runner)


        execute_runners(runners, grid_logger, grid_logger_avg, is_nni=self._nni)

def main(params, grid_logger, grid_logger_avg):
    gtc = GCNTemporalCommunities(nni=params['is_nni'])
    gtc.load_data()
    gtc.prep_trial(params, grid_logger, grid_logger_avg)


if __name__ == "__main__":

    pr = cProfile.Profile()
    pr.enable()
    grid_outputs_folder = time.strftime("%Y%m%d_%H%M%S")

    res_path = os.path.join(os.getcwd(), "dataset", Dataset_name, "grid", grid_outputs_folder)
    if not os.path.exists(res_path):
        os.makedirs(res_path)

    grid_logger = CSVLogger("results_%s" % 'grid' + time.strftime("%Y%m%d_%H%M%S"), path=res_path)
    grid_logger_avg = CSVLogger("results_%s" % 'grid_it_avg' + time.strftime("%Y%m%d_%H%M%S"), path=res_path)
    grid_logger.info("iteration", "total_it", "lr", "do", "hid_size", "wd", "temp_pen", "epochs",
                     "train_reg_loss", "train_temp_loss", "total_train_loss", "train_acc_f1_macro", "train_f1_micro",
                     "test_reg_loss", "test_temp_loss", "total_test_loss", "test_f1_macro", "test_f1_micro")

    grid_logger_avg.info("iterations", "lr", "do", "hid_size", "wd", "temp_pen", "epochs",
                         "train_reg_loss", "train_temp_loss", "total_train_loss", "train_f1_macro", "train_f1_micro",
                         "test_reg_loss", "test_temp_loss", "total_test_loss", "test_f1_macro", "test_f1_micro")

    num_of_grids = len(LR) * len(Hid_size) * len(Regularization) * len(Temporal_pen) * len(Dropout)
    grid_counter = 0

    configurations = list(product(*[LR, Hid_size, Regularization, Temporal_pen, Dropout]))
    for LR, Hid_size, Regularization, Temporal_pen, Dropout in configurations:
        grid_counter += 1

        print("\ngrid {} out of {}:".format(grid_counter, num_of_grids))

        params = {"hid_size": Hid_size,
                  "epochs": Epochs,
                  "dropout": Dropout,
                  "lr": LR,
                  "weight_decay": Regularization,
                  "temporal_pen": Temporal_pen,
                  "optimizer": Optimizer,
                  "iterations": Iterations,
                  "time_inds": Time_inds,
                  "optim_name": 'Adam',
                  "dataset_name": Dataset_name,
                  "number_of_classes": Number_Of_Classes,
                  "is_nni": False,
                  "name": "lr_" + str(LR) + "_do_" + str(Dropout) + "_wd_" + str(Regularization) + "_Tempen_" + str(
                      Temporal_pen) + "_hid_size_" + str(Hid_size),
                  "grid_output_folder": grid_outputs_folder,
                  "loss_weights_type": loss_weights}
        main(params, grid_logger, grid_logger_avg)
    pr.disable()
    pr.print_stats(sort="time")
