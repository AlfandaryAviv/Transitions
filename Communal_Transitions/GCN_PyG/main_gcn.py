# import csv
import math
# import operator
import time
# import sys
from sklearn.neighbors import KDTree
from pathlib import Path
# sys.path.insert(1, "../graph-measures")
# from graph_measures.loggers import PrintLogger
from loggers import PrintLogger
import torch
import os
import pickle
import networkx as nx
import numpy as np
from torch_geometric.data import DataLoader, Data
from torch import nn, optim
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
# import copy
# from sklearn.metrics import roc_auc_score
from sklearn import metrics
import matplotlib.pyplot as plt
# from itertools import product

DataSetName = "dblp"
Net = "gcn"
Trials = 1
Logger = PrintLogger()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GCNNet2(nn.Module):
    def __init__(self, num_features, num_classes, h_layers=[16,16], dropout=0.5, activation="relu"):

        super(GCNNet2, self).__init__()
        self._conv1 = GCNConv(num_features, h_layers[0])
        self._conv2 = GCNConv(h_layers[0], num_classes)
        self._dropout = dropout
        self._activation_func = F.relu
        # self._device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, Data):
        x, adj_mx = Data.x.to(self._device), Data.adj_mx.to(self._device)
        x = self._conv1(x, adj_mx)
        x = self._activation_func(x)
        x = F.dropout(x, p=self._dropout)
        x = self._conv2(x, adj_mx)
        return F.softmax(x, dim=1)


class GCNNet3(nn.Module):
    
    def __init__(self, num_features, num_classes, h_layers=[20, 20], dropout=0.5, activation = "relu"):
        super(GCNNet3, self).__init__()
        self._conv1 = GCNConv(num_features,h_layers[0])
        self._conv2 = GCNConv(h_layers[0], h_layers[1])
        self._conv3 = GCNConv(h_layers[1], num_classes)
        self._dropout = dropout
        self._activation_func = F.relu
        # self._device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, Data):
        x, adj_mx = Data.x.to(self._device), Data.adj_mx.to(self._device)
        x = self._conv1(x, adj_mx)
        x = self._activation_func(x)
        x = F.dropout(x, p=self._dropout)
        x = self._conv2(x, adj_mx)
        x = self._activation_func(x)
        x = self._conv3(x, adj_mx)
        return F.softmax(x, dim=1)

class Model:
    def __init__(self, parameters):
        self._params = parameters
        # self._device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._data = None
        self._criterion = self._soft_ce_loss

    def load_data(self,year, gnx, labels_vec, feature_matrix, adj, train, test, train_tags, test_tags):
        self.year = year
        nodes = sorted(gnx.nodes)
        x = torch.tensor(np.vstack([feature_matrix[node] for node in nodes]), device=self._device)
        dict = {x: i for i, x in enumerate(nodes)}
        edges = torch.tensor(np.vstack([[dict[x[0]] for x in gnx.edges],
                                        [dict[x[1]] for x in gnx.edges]]), dtype=torch.long, device=self._device)
        self._data = Data(x=x, edge_index=edges)
        self._data.train_idx = train
        self._data.test_idx = test
        self._data.train_tags = torch.tensor(train_tags, dtype=torch.double, device=self._device)
        self._data.test_tags = torch.tensor(test_tags, dtype=torch.double, device=self._device)
        self._data.adj_mx = torch.tensor(np.vstack((adj.row, adj.col)), dtype=torch.long, device=self._device)
        self._num_features = x.shape[1]
        self._num_classes = self._params['num_of_classes']
        self._net = self._params['net'](self._num_features, self._num_classes, h_layers=self._params['hidden_sizes'],
                           dropout=self._params['dropout_rate'],
                           activation=self._params['activation'])
        self._net.to(self._device)
        self._optimizer = optim.Adam(self._net.parameters(), lr=self._params['learning_rate'],
                                     weight_decay=self._params['weight_decay'])
        self.out = []
        return self._data

    @property
    def data(self):
        return self._data.clone()

    def _soft_ce_loss(self, predicted, target):
        return -(target * torch.log(predicted)).sum(dim=1).mean().to(self._device)

    def train(self):
        self._net.train()
        train_labels = self._data.train_tags
        test_labels = self._data.test_tags
        train_inds = self._data.train_idx
        test_inds = self._data.test_idx

        tr_loss, te_loss, tr_acc, te_acc = [],[],[],[]
        for epoch in range(int(self._params["epochs"])):
            self._optimizer.zero_grad()
            output = self._net(self._data)
            output = output[train_inds,:]
            loss = self._criterion(output, train_labels)
            mse_for_acc = torch.nn.MSELoss()
            mse_for_acc = mse_for_acc(output, train_labels.float())
            tr_loss.append(loss.data.cpu().item())
            tr_acc.append(mse_for_acc.data.cpu().item())
            loss.backward()
            self._optimizer.step()

            #valid
            self._net.eval()
            val_output = self._net(self._data)
            val_output = val_output[test_inds,:]
            val_loss = self._criterion(val_output, test_labels)
            mse_for_acc_test = torch.nn.MSELoss()
            mse_for_acc_test = mse_for_acc_test(val_output, test_labels.float())
            te_loss.append(val_loss.data.cpu().item())
            te_acc.append(mse_for_acc_test.data.cpu().item())

            if epoch == (int(self._params["epochs"]) -1):
                # grid_writer.writerow([str(self._net.__class__.__name__), str(self._params['learning_rate']), str(self._params['weight_decay']), str(self._params["dropout_rate"]),str(self._params["hidden_size"]), loss.data.cpu().numpy(), val_loss.data.cpu().numpy(), mse_for_acc.data.cpu().numpy(), mse_for_acc_test.data.cpu().numpy()])
                # print("\n layers: {}, lr: {}, wd: {}, do: {}").format(str(self._net.__class__.__name__), str(self._params['learning_rate']),str(self._params['weight_decay']), str(self._params["dropout_rate"]) )
                print("year: {}, epoch: {}, train loss: {:.5f}, test loss:{:.5f}, train mse acc: {:.5f}, test mse acc: {:.5f} ".format(self.year, epoch+1, loss, val_loss, mse_for_acc, mse_for_acc_test))
            self._net.train()
        self.plt_results(tr_loss, te_loss, tr_acc, te_acc)
        return output



    def plt_results(self, tr_loss, te_loss, tr_acc, te_acc):

        year = str(self.year)
        lr = str(self._params["learning_rate"])
        wd = str(self._params['weight_decay'])
        hs = str(self._params["hidden_sizes"])
        do = str(self._params["dropout_rate"])

        fig, axes = plt.subplots(2, 2, figsize=(10,10))
        plt.suptitle("year: "+year+", lr: " + lr+", wd: "+wd+", hidden_size: "+hs+", dropout: "+ do, fontsize=12, y=0.99)
        epoch = [e for e in range(self._params["epochs"])]
        plt.tight_layout()

        plt.subplots_adjust(left=0.125,right=0.9,bottom=0.1,top=0.9,wspace=0.4,hspace=0.4)
        axes[0, 0].set_title('train ce loss',fontsize=10)
        axes[0, 0].set_xlabel("epoch",fontsize=8)
        axes[0, 0].set_ylabel("loss",fontsize=8)
        axes[0, 0].plot(epoch, tr_loss)

        axes[0, 1].set_title('test ce loss',fontsize=10)
        axes[0, 1].set_xlabel("epoch",fontsize=8)
        axes[0, 1].set_ylabel("test loss",fontsize=8)
        axes[0, 1].plot(epoch, te_loss)

        axes[1, 1].set_title('train accuracy(mse loss)',fontsize=10)
        axes[1, 1].set_xlabel("epoch",fontsize=8)
        axes[1, 1].set_ylabel("train accuracy(mse loss)",fontsize=8)
        axes[1, 1].plot(epoch, tr_acc)

        axes[1, 0].set_title('test accuracy(mse loss)',fontsize=10)
        axes[1, 0].set_xlabel("epoch",fontsize=8)
        axes[1, 0].set_ylabel("test accuracy(mse loss)",fontsize=8)
        axes[1, 0].plot(epoch, te_acc)

        check("./dataset/"+self._params["data_name"]+"/plots/GCN_results/")

        plt.savefig("./dataset/"+self._params["data_name"]+"/plots/GCN_results/GCN_results_year_" + year +
                    "_lr_" + lr + "_hs_" + hs + "_do_" + do + "_wd_" + wd +".jpg",bbox_inches='tight')
        plt.close(fig)
        return

    def test(self):
        # test_labels = self._data.test_tags
        test_inds = self._data.test_idx

        self._net.eval()
        output = self._net(self._data)
        out_test = output[test_inds, :]
        return output, out_test



def load_input(parameters):
    graphs, labels, feature_mx = [], [], []
    check("./dataset/"+str(parameters["data_name"])+"/pkl/gcn_input/")
    for i in range(parameters['time_ins']):
        g = nx.read_gpickle("./dataset/"+str(parameters["data_name"])+"/pkl/gcn_input/graph_{}.pkl".format(str(i)))
        l = pickle.load(open("./dataset/"+str(parameters["data_name"])+"/pkl/gcn_input/labels_{}.pkl".format(str(i)), "rb"))
        mx = pickle.load(open("./dataset/"+str(parameters["data_name"])+"/pkl/gcn_input/mx_{}.pkl".format(str(i)), "rb"))
        graphs.append(g)
        labels.append(l)
        feature_mx.append(mx)
    adjacency_matrices = [nx.adjacency_matrix(g).tocoo() for g in graphs]

    train_old = pickle.load(open("./dataset/"+str(parameters["data_name"])+"/pkl/train_person.pkl", "rb"))
    test_old = pickle.load(open("./dataset/"+str(parameters["data_name"])+"/pkl/test_person.pkl", "rb"))
    old_to_new_id = pickle.load(open("./dataset/"+str(parameters["data_name"])+"/pkl/old_to_new_nid.pkl","rb"))

    train_set, test_set = [], []
    for id in train_old:
        new = old_to_new_id[id]
        train_set.append(new)
    for id in test_old:
        new = old_to_new_id[id]
        test_set.append(new)

    train = [[k for k in train_set if labels[j][k] != -1] for j in range(len(labels))]
    test = [[k for k in test_set if labels[j][k] != -1] for j in range(len(labels))]

    test_labels = [[labels[j][k] for k in test_set if labels[j][k] != -1] for j in range(parameters['time_ins'])]
    train_labels = [[labels[j][k] for k in train_set if labels[j][k] != -1] for j in range(parameters['time_ins'])]

    return graphs, labels, feature_mx, adjacency_matrices, train, test, train_labels, test_labels

def sim_edges(out, parameters):
    t=time.time()
    id_year = []
    out_list = []
    new_edges=set()
    nodes = pickle.load(open("./dataset/"+str(parameters["data_name"])+"/pkl/nodes_set.pkl", "rb"))  # old nodes

    new_to_old_id = pickle.load(open("./dataset/"+str(parameters["data_name"])+"/pkl/new_to_old_nid.pkl", "rb"))

    for j in range(len(out)):
        for n in range(len(out[j])):
            name = str(str(new_to_old_id[n])+"_"+str(j))
            if name in nodes:
                id_year.append(name)
                x = out[j][n].data.cpu().numpy().reshape((-1,1))
                out_list.append(x)

    out_list = np.asarray(out_list).reshape(-1, parameters["num_of_classes"])
    t=time.time()
    tree = KDTree(out_list, leaf_size=100)
    print("creating the tree: {:.5f}".format(time.time()-t))
    s = pickle.dumps(tree)
    tree_copy = pickle.loads(s)
    for output in range(1, len(out_list)+1):
        if output == 1:
            dist, ind = tree_copy.query(out_list[:output], k=11)
        else:
            dist, ind = tree_copy.query(out_list[output-1:output], k=11)
        for i in ind[0]:
            if id_year[output-1]== id_year[i]:
                continue
            else:
                pair = (id_year[output-1], id_year[i])
                new_edges.add(pair)

    res = {tuple(sorted(t)) for t in new_edges}
    new_edges = list(res)

    pickle.dump(new_edges, open("./dataset/" + str(parameters["data_name"]) + "/pkl/similarity_edges.pkl", "wb"))
    # with open(os.path.join("dataset",str(parameters["data_name"]), "pkl","similarity_edges.pkl"), "wb") as f:
    #     pickle.dump(new_edges, f)

    print("kdtree time with pickle: {:.5f}".format(time.time()-t))
    return new_edges

def accuracy(out, test_labels, name, parameters ):
    y_true = [[] for i in range(parameters["num_of_classes"])]
    y_pred = [[] for i in range(parameters["num_of_classes"])]
    for j in range(len(out)):
        for k in range(len(out[j])):
            for i in range(parameters["num_of_classes"]):
                if test_labels[j][k][i] > 0.5:
                    test_labels[j][k][i] = 1
                elif test_labels[j][k][i] <= 0.5:
                    test_labels[j][k][i] = 0
                y_true[i].append(test_labels[j][k][i])
                y_pred[i].append(out[j][k][i].data.cpu().item())
    l=[]
    for i in range(0,parameters["num_of_classes"]):

        fpr, tpr, thresholds = metrics.roc_curve(np.array(y_true[i]), np.array(y_pred[i]), pos_label=1)
        x = metrics.auc(fpr, tpr)
        l.append(x)

    if name == "test":
        print("----AUC test----")
    elif name == "train":
        print("----AUC train----")
    for i in range(len(l)):
        print("tag: {}, auc: {:.3f} ".format(i, l[i]))
    return l

def cal_average(num):
    sum_num = 0.
    i=0
    for t in num:
        if math.isnan(t):
            t=0
            i+=1
        else:
            sum_num += t
    total = len(num)-i
    if total != 0:
        avg = float(sum_num / (len(num)-i))
    else:
        avg = 0
    return avg

def plot_auc(auc_test, auc_train,parameters):
    x = [i for i in range(len(auc_test))]
    index = np.arange(parameters['num_of_classes'])
    width = 0.35
    y_tr = [auc_train[i] for i in range(len(auc_train))]
    y_te = [auc_test[i] for i in range(len(auc_test))]

    print("avg train: {:.4f}, avg test: {:.4f}".format(cal_average(y_tr), cal_average(y_te)))
    fig, ax = plt.subplots()
    ax.bar(index - width/2, y_te, color='red',width=width, label='AUC test')
    ax.bar(index + width/2, y_tr, color='blue',width=width, label='AUC train')
    plt.xlabel('tag')
    plt.ylabel('AUC')
    plt.title('AUC of {} years'.format(parameters["time_ins"]))
    plt.xticks(index + width/2, x)
    ax.legend()
    fig.tight_layout()

    lr = str(parameters["learning_rate"])
    wd = str(parameters['weight_decay'])
    hs = str(parameters["hidden_sizes"])
    do = str(parameters["dropout_rate"])

    plt.savefig("./dataset/" + parameters["data_name"] + "/plots/GCN_results/auc_timeinds" +str(parameters["time_ins"])+
                "_lr_" + lr + "_hs_" + hs + "_do_" + do + "_wd_" + wd + ".jpg", bbox_inches='tight')
    plt.close(fig)

def run_trial(parameters):
    parameters.update({'trials': 1})
    print("Starting")
    print(parameters)
    for trial in range(parameters['trials']):
        graphs, labels, feature_mx, adjacency_matrices, train_ids, test_ids, train_labels, test_labels = load_input(parameters)
        year = [i for i in range(parameters['time_ins'])]
        out = []
        out_train = []
        total_out =[]
        for y, g,l,fmx, adj,tr, te, tr_tags, te_tags in zip(year, graphs, labels, feature_mx, adjacency_matrices,train_ids, test_ids, train_labels, test_labels):
            print("\n----year: {}----".format(y))
            model = Model(parameters)
            model.load_data(y,g,l,fmx, adj,tr, te, tr_tags, te_tags)
            out_train.append(model.train())
            all_out, out_test = model.test()
            out.append(out_test)
            total_out.append(all_out)
        auc_test = accuracy(out,test_labels, "test", parameters)
        auc_train = accuracy(out_train, train_labels, "train", parameters)
        plot_auc(auc_test, auc_train,parameters)
        new_edges = sim_edges(total_out, parameters)
        return new_edges


def check(file_path,current_path=False):
    directory = os.path.dirname(file_path)

    if not os.path.exists(directory):
        os.makedirs(directory)
        print("A new directory was created: "+str(file_path))
    else:
        print("directory exists: "+str(file_path))

    if current_path:
        if os.path.isfile(current_path):
            Path(current_path).rename(file_path)
            print("input files moved to directory:", str(file_path))
        elif os.path.isfile(file_path):
            print("input file exists in directory:", str(file_path))
        else:
            print("input file is missing in directory", str(file_path))
    return file_path


if __name__ == '__main__':
    # net = [GCNNet3]
    # lr = [0.03]
    # wd = [0.00001, 0.0005]
    # hs = [[10,10], [20,20], [30,30], [40,40],[50,50]]
    # do = [0.3, 0.6]
    # with open('grid_params.csv', mode='w') as grid_params_file:
    #     grid_writer = csv.writer(grid_params_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #     configurations = list(product(*[net, lr, wd, hs, do]))
    #     for net, lr, wd, hs, do in configurations:
    params_ = {"data_name": DataSetName,
              "net": GCNNet3,
              "trials": Trials,         # number of trials to run. output is the average score over all trials
              "epochs": 120,
              "activation": "relu",
              "dropout_rate": 0.3,
              "hidden_sizes": [10,10],
              "learning_rate": 0.03,
              "weight_decay": 0.0005,
              "time_ins": 21,
               "num_of_classes":15
            }
    run_trial(params_)