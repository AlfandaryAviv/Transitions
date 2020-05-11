import time
import os
from random import shuffle
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import nni
import logging
from loggers import EmptyLogger, CSVLogger, PrintLogger, FileLogger, multi_logger
from model import GCN
import pickle



class ModelRunner:
    def __init__(self, conf, logger, data_logger=None, is_nni=False):
        self._logger = logger
        self._data_logger = EmptyLogger() if data_logger is None else data_logger
        self._conf = conf
        self.bar = 0.5
        self._lr = conf["lr"]
        self._is_nni = is_nni
        self._device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self._ce_loss = self._soft_ce_loss
        self._temporal_loss = torch.nn.MSELoss(reduction='sum').to(self._device)


    @property
    def logger(self):
        return self._logger

    @property
    def data_logger(self):
        return self._data_logger

    def _soft_ce_loss(self, predicted, target,weights = None):

        predicted = torch.clamp(predicted, 1e-9, 1 - 1e-9)
        if weights is None:
            return -(target * torch.log(predicted)).sum(dim=1).sum().to(self._device)
        weights = torch.FloatTensor(weights).to(device=self._device)
        weights[weights==0] = 0.01
        b = -(torch.sqrt((weights).sum()/weights) * target * torch.log(predicted)).sum(dim=1).sum().to(self._device)
        return b

    def _get_model(self):
        model = GCN(in_features=self._conf["training_mat"][0].shape[1],
                    hid_features=self._conf["hid_features"], out_features=15,
                    activation=self._conf["activation"], dropout= self._conf["dropout"])
        opt = self._conf["optimizer"](model.parameters(), lr=self._conf["lr"], weight_decay=self._conf["weight_decay"])
        ##checged : added "feature_matrices"
        return {"model": model, "optimizer": opt,
                "training_mats": self._conf["training_mat"],
                "training_labels": self._conf["training_labels"],
                "test_mats": self._conf["test_mat"],
                "test_labels": self._conf["test_labels"],
                "adj_matrices": self._conf["adj_matrices"],
                "feature_matrices": self._conf["feature_matrices"]}

    # verbose = 0 - silent
    # verbose = 1 - print test results
    # verbose = 2 - print train for each epoch and test results
    def run(self, verbose=2):
        if self._is_nni:
            verbose = 0
        model = self._get_model()
        ##
        loss_train, acc_train, intermediate_acc_test, losses_train, losses_tempo, accs_train, test_results = self.train(self._conf["epochs"],
                                                                                            model=model,
                                                                                            verbose=verbose)
        ##
        # Testing
        result = self.test(model=model, verbose=verbose if not self._is_nni else 0, print_to_file=True)
        if self._is_nni:
            self._logger.debug('Final loss train: %3.4f' % loss_train)
            self._logger.debug('Final accuracy train: %3.4f' % acc_train)
            final_results = result["acc"]
            self._logger.debug('Final accuracy test: %3.4f' % final_results)
            # _nni.report_final_result(test_auc)

        if verbose != 0:
            names = ""
            vals = ()
            for name, val in result.items():
                names = names + name + ": %3.4f  "
                vals = vals + tuple([val])
                self._data_logger.info(name, val)
        parameters = {"temporal_pen" : self._conf["temporal_pen"],"lr":self._conf["lr"], "weight_decay":self._conf["weight_decay"],
                      "dropout":self._conf["dropout"], "optimizer":self._conf["optim_name"]}
        return loss_train, acc_train, intermediate_acc_test, result, losses_train,losses_tempo, accs_train, test_results, parameters



    def train(self, epochs, model=None, verbose=2):
        loss_train = 0.
        acc_train = 0.
        losses_train = []
        tempo_losses_train = []
        accs_train = []
        test_results = []
        intermediate_test_acc = []
        for epoch in range(epochs):
            loss_train, tempo_loss, acc_train = self._train(epoch, model, verbose)
            ##
            losses_train.append(loss_train)
            accs_train.append(acc_train)
            tempo_losses_train.append(tempo_loss)
            ##
            # /----------------------  FOR NNI  -------------------------
            if epoch % 5 == 0:
                test_res = self.test(model, verbose=verbose if not self._is_nni else 0)
                test_results.append(test_res)
                if self._is_nni:
                    test_acc = test_res["acc"]
                    intermediate_test_acc.append(test_acc)

        return loss_train, acc_train, intermediate_test_acc, losses_train, \
               tempo_losses_train, accs_train, test_results

    def _train(self, epoch, model, verbose=2):
        model_ = model["model"]
        model_ = model_.to(self._device)
        optimizer = model["optimizer"]
        z_vals = [[] for _ in range(len(model["adj_matrices"]))]
        outputs = [[] for _ in range(len(model["adj_matrices"]))]
        labeled_indices = [[i for i in range(len(model["training_labels"][t])) if model["training_labels"][t][i] != -1]# and not np.array_equal([1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],model["training_labels"][t][i])]
                           for t in range(len(model["training_labels"]))]
        labels = [torch.DoubleTensor([model["training_labels"][t][i]
                                      for i in labeled_indices[t]]).to(self._device) for t in
                  range(len(model["training_labels"]))]
        model_.train()
        optimizer.zero_grad()
        for idx, adj in enumerate(model["adj_matrices"]):
            training_mat = torch.from_numpy(model["feature_matrices"][idx]).to(dtype=torch.float, device=self._device)
            z, output = model_(training_mat, adj)
            z_vals[idx].append(z)  # After 1 GCN layer only
            outputs[idx].append(output)  # Final guesses
        tempo_loss = 0.
        for t in range(len(z_vals) - 1):
            #tempo_loss += self._conf["temporal_pen"] * sum( [self._temporal_loss(z_vals[t + 1][0][j], z_vals[t][0][j]) for j in range(len(z_vals[t][0]))])
            tempo_loss += self._conf["temporal_pen"] * self._temporal_loss(z_vals[t + 1][0], z_vals[t][0])
        loss_train = 0.
        for u in range(len(outputs)):  # For all times
            Nj_s = [sum([labels[u][t][j] for t in range(len(labels[u]))]) for j in range(15)]
            out = outputs[u][0][labeled_indices[u], :]
            loss_train += self._ce_loss(out, labels[u], Nj_s)

        tempo_loss /= ((len(z_vals) - 1) * model["feature_matrices"][0].shape[0])
        loss_train /= sum([len(labeled_indices[u]) for u in range(len(outputs))])
        total_loss = loss_train + tempo_loss
        total_loss.backward()
        optimizer.step()



        ###acc_train = [self.accuracy(outputs[i][0][labeled_indices[i], :], labels[i]) for i in range(len(labels))]
        acc_train = self.accuracy(outputs[-1][0][labeled_indices[-1], :], labels[-1])
        #acc_train/= len(outputs[-1][0][labeled_indices[-1], :])
        # TODO: Right now the training accuracy is only on the last time. Should change?

        if verbose == 2:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            self._logger.debug('Epoch: {:04d} '.format(epoch + 1) +
                               'ce_loss_train: {:.4f} '.format(loss_train.data.item()) +
                               'temp_loss: {:.4f} '.format(tempo_loss.data.item()) +
                               'acc_train: {:.4f} '.format(acc_train))
        return loss_train, tempo_loss, acc_train

    def test(self, model=None, verbose=2, print_to_file=False):
        model_ = model["model"]
        z_vals = [[] for x in range(len(model["adj_matrices"]))]
        outputs = [[] for x in range(len(model["adj_matrices"]))]

        labeled_indices = [[i for i in range(len(model["test_labels"][t])) if model["test_labels"][t][i] != -1]# and not np.array_equal([1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],model["test_labels"][t][i])]
                           for t in range(len(model["test_labels"]))]
        labels = [torch.DoubleTensor([model["test_labels"][t][i]
                                      for i in labeled_indices[t]]).to(self._device) for t in
                  range(len(model["test_labels"]))]

        model_.eval()
        for idx, adj in enumerate(model["adj_matrices"]):
            test_mat = torch.from_numpy(model["feature_matrices"][idx]).to(self._device)
            z, output = model_(*[test_mat, adj])
            z_vals[idx].append(z)
            outputs[idx].append(output)
            
            
        if print_to_file:
            self._logger.debug("\nprint to files")
            for i in range(len(model["adj_matrices"])):
                np_output = outputs[i][0].cpu().data.numpy()
                with open(os.path.join("gcn_MSE_weightedLoss", "gcn_" + str(i) + ".pkl"), "wb") as f:
                    pickle.dump(np_output, f, protocol=pickle.HIGHEST_PROTOCOL)

                    
        tempo_loss = 0.
        for t in range(len(z_vals) - 1):
            tempo_loss += self._conf["temporal_pen"] * self._temporal_loss(z_vals[t + 1][0], z_vals[t][0])
        loss_test = 0.
        for u in range(len(outputs)):  # For all times
            out = outputs[u][0][labeled_indices[u], :]
            loss_test += self._ce_loss(out, labels[u])


        tempo_loss /= ((len(z_vals) - 1) * model["feature_matrices"][0].shape[0])
        loss_test /= sum([len(labeled_indices[u]) for u in range(len(outputs))])
        total_loss = loss_test + tempo_loss

        ##acc_test = self.accuracy(outputs[-1][labeled_indices[-1], :], labels[-1])  # TODO: Same for accuracy on train.
        acc_test = self.accuracy(outputs[-1][0][labeled_indices[-1], :],
                                 labels[-1])  # TODO: Same for accuracy on train.

        if verbose != 0:
            self._logger.info("Test: ce_loss= {:.4f} ".format(loss_test.data.item()) +
                              "temp_loss= {:.4f} ".format(tempo_loss.data.item()) + "acc= {:.4f}".format(acc_test))
        result = {"loss": loss_test.data.item(), "acc": acc_test, "tempo_loss": tempo_loss.data.item()}
        return result

    @staticmethod
    def accuracy(output, labels):  # should be named kl_divergence and best at lowest value
        labs = labels.cpu().data.numpy()
        out = output.cpu().data.numpy()
        d_per_sample = np.zeros(labels.size(0))
        for i in range(labels.size(0)):
            for j in range(labels.size(1)):
                if labs[i, j] == 0:
                    continue
                else:
                    d_per_sample[i] += labs[i, j] * np.log(labs[i, j] / out[i, j])
        mean_d = np.mean(d_per_sample)
        return mean_d


def plot_graphs(info):
    # info[4] is list of train losses 1 . info[5] is list of train losses 2 - tempo. list[6] is list of acc train.
    #info [7] is list of dictionaries, each dictionary is for epoch, each one contains "loss" - first loss,"acc"- acc,  "tempo_loss" - tempo loss
    #info[8] is the temporal_oen
    parameters = info[8]
    regulariztion = str(parameters["weight_decay"])
    lr = str(parameters["lr"])
    temporal_pen = str(parameters["temporal_pen"])# the origin reg first e
    optimizer = str(parameters["optimizer"])
    dropout = str(parameters["dropout"])

    #train

    # Share a X axis with each column of subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    plt.suptitle("Train: tmplre="+temporal_pen +" lr="+lr+" reg= "+regulariztion+ " dropout= "+dropout+" opt= "+optimizer, fontsize=16, y=0.99)

    epoch = [e for e in range(1, len(info[4])+1)]
    axes[0, 0].set_title('CE loss')
    axes[0, 0].set_xlabel("Iterations")
    axes[0, 0].set_ylabel("loss")
    axes[0, 0].plot(epoch, info[4])

    axes[0, 1].set_title('temporal loss')
    axes[0, 1].set_xlabel("Iterations")
    axes[0, 1].set_ylabel("loss")
    axes[0, 1].plot(epoch, info[5])

    axes[1, 1].set_title('accuracy')
    axes[1, 1].set_xlabel("Iterations")
    axes[1, 1].set_ylabel("accuracy")
    axes[1, 1].plot(epoch, info[6])
    fig.delaxes(axes[1,0])

    plt.savefig("figures/Train_all_y_"+"tmplre_"+temporal_pen+"lr_"+lr+" reg= "+regulariztion+ " dr= "+dropout+" opt= "+optimizer+".png")

    #Test

    # Share a X axis with each column of subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    plt.suptitle("Test: tmplre="+temporal_pen +" lr="+lr+" reg= "+regulariztion+ " dropout= "+dropout+" opt= "+optimizer, fontsize=16, y=0.99)


    epoch = [e for e in range(1, len(info[7])+1)]
    test_ce_loss = [ info[7][i]["loss"] for i in range(len(info[7])) ]
    test_tempo_loss = [ info[7][i]["tempo_loss"] for i in range(len(info[7])) ]
    acc_test =  [ info[7][i]["acc"] for i in range(len(info[7])) ]
    axes[0, 0].set_title('CE loss')
    axes[0, 0].set_xlabel("Iterations")
    axes[0, 0].set_ylabel("loss")
    axes[0, 0].plot(epoch, test_ce_loss)

    axes[0, 1].set_title('temporal loss')
    axes[0, 1].set_xlabel("Iterations")
    axes[0, 1].set_ylabel("loss")
    axes[0, 1].plot(epoch, test_tempo_loss)

    axes[1, 1].set_title('accuracy')
    axes[1, 1].set_xlabel("Iterations")
    axes[1, 1].set_ylabel("accuracy")
    axes[1, 1].plot(epoch, acc_test)
    fig.delaxes(axes[1,0])

    #fig.set_size_inches(3, 6, forward=True)
    plt.savefig("figures/Test_all_y_"+"tmplre_"+temporal_pen+"lr_"+lr+" reg= "+regulariztion+ " dr= "+dropout+" opt= "+optimizer+".png")
    plt.clf()
    #plt.show()




def execute_runner(runners, is_nni=False):
    train_losses = []
    train_accuracies = []
    test_intermediate_results = []
    test_losses = []
    test_accuracies = []
    for idx_r, runner in enumerate(runners):
        rs = runner.run(verbose=2)
        train_losses.append(rs[0])
        train_accuracies.append(rs[1])
        test_intermediate_results.append(rs[2])
        test_losses.append(rs[3]["loss"])
        test_accuracies.append(rs[3]["acc"])
        if idx_r == 0:
            plot_graphs(rs)
    if is_nni:
        mean_intermediate_res = np.mean(test_intermediate_results, axis=0)
        for i in mean_intermediate_res:
            nni.report_intermediate_result(i)
        nni.report_final_result(np.mean(test_accuracies))

    runners[-1].logger.info("*" * 15 + "Final accuracy train: %3.4f" % np.mean(train_accuracies))
    runners[-1].logger.info("*" * 15 + "Std accuracy train: %3.4f" % np.std(train_accuracies))
    runners[-1].logger.info("*" * 15 + "Final accuracy test: %3.4f" % np.mean(test_accuracies))
    runners[-1].logger.info("*" * 15 + "Std accuracy train: %3.4f" % np.std(train_accuracies))
    runners[-1].logger.info("Finished")
    return


def build_model(training_data, training_labels, test_data, test_labels, adjacency_matrices,
                hid_features, activation, optimizer, epochs, dropout, lr, l2_pen, temporal_pen,
                dumping_name, feature_matrices, is_nni=False):
    optim_name="SGD"
    if optimizer==optim.Adam:
        optim_name = "Adam"
    conf = {"hid_features": hid_features, "dropout": dropout, "lr": lr, "weight_decay": l2_pen,
            "temporal_pen": temporal_pen,
            "training_mat": training_data, "training_labels": training_labels,
            "test_mat": test_data, "test_labels": test_labels, "adj_matrices": adjacency_matrices,
            "optimizer": optimizer, "epochs": epochs, "feature_matrices": feature_matrices, "activation": activation,"optim_name":optim_name}

    products_path = os.path.join(os.getcwd(), "logs", dumping_name, time.strftime("%Y%m%d_%H%M%S"))
    if not os.path.exists(products_path):
        os.makedirs(products_path)

    logger = multi_logger([
        PrintLogger("MyLogger", level=logging.DEBUG),
        FileLogger("results_%s" % dumping_name, path=products_path, level=logging.INFO)], name=None)

    data_logger = CSVLogger("results_%s" % dumping_name, path=products_path)
    data_logger.info("model_name", "loss", "acc")

    ##
    logger.info('STARTING with lr= {:.4f} '.format(lr) + ' dropout= {:.4f} '.format(dropout)+ ' regulariztion_l2_pen= {:.4f} '.format(l2_pen)
                + ' temporal_pen= {:.10f} '.format(temporal_pen)+ ' optimizer= %s ' %optim_name)
    logger.debug('STARTING with lr=  {:.4f} '.format(lr) + ' dropout= {:.4f} '.format(dropout) + ' regulariztion_l2_pen= {:.4f} '.format(l2_pen)
        + ' temporal_pen= {:.10f} '.format(temporal_pen) + ' optimizer= %s ' %optim_name)
    ##

    runner = ModelRunner(conf, logger=logger, data_logger=data_logger, is_nni=is_nni)
    return runner


def main_gcn(feature_matrices, adj_matrices, labels, hid_features,
             optimizer=optim.Adam, epochs=200, dropout=0.3, lr=0.01, l2_pen=0.005, temporal_pen=1e-6,
             iterations=1, dumping_name='', is_nni=False):
    runners = []
    for it in range(iterations):
        rand_test_indices = np.random.choice(len(labels[0]), round(len(labels[0]) * 0.9), replace=False)
        # TODO: Choose in a smarter way (takes into account whether a node is labeled).
        train_indices = np.delete(np.arange(len(labels[0])), rand_test_indices)
        # rand_test_indices, train_indices = train_test_split(labels)

        test_features = [feature_matrices[j][rand_test_indices, :] for j in range(len(feature_matrices))]
        test_labels = [[labels[j][k] for k in rand_test_indices] for j in range(len(test_features))]

        training_features = [feature_matrices[j][train_indices, :] for j in range(len(feature_matrices))]
        training_labels = [[labels[j][k] for k in train_indices] for j in range(len(training_features))]

        activation = torch.nn.functional.relu
        '''runner = build_model(training_features, training_labels, test_features, test_labels,
                             adj_matrices, hid_features, activation, optimizer, epochs, dropout, lr,
                             l2_pen, temporal_pen, dumping_name, is_nni=is_nni) #itay'''

        ##changed: training_features is not used. added feature_matrices parameter so gcn will learn on the whole graph.
        runner = build_model(training_features, training_labels, test_features, test_labels,
                             adj_matrices, hid_features, activation, optimizer, epochs, dropout, lr,
                             l2_pen, temporal_pen, dumping_name, feature_matrices=feature_matrices, is_nni=is_nni)

        runners.append(runner)
    execute_runner(runners, is_nni=is_nni)
    return


'''def train_test_split(labels):
    train_indices = []
    test_indices = []
    to_shuffle_indices = {}
    for id in labels:
        if idx_0 != -1:
            if labels[idx_0, 0] == 1:
                continue
                # test_indices.append(idx_0)
            else:
                to_shuffle_indices[id] = (idx_0,)
        if idx_1 != -1:
            if labels[idx_1, 0] == 1:
                continue
                # test_indices.append(idx_1)
            else:
                if id in to_shuffle_indices:
                    to_shuffle_indices[id] += (idx_1, )
                else:
                    to_shuffle_indices[id] = (idx_1, )
    indices = list(to_shuffle_indices.keys())
    shuffle(indices)
    train, test = indices[:int(len(indices) * 0.1)], indices[int(len(indices) * 0.1):]
    for id in train:
        train_indices += [idx for idx in to_shuffle_indices[id]]
    for id in test:
        test_indices += [idx for idx in to_shuffle_indices[id]]
    return train_indices, test_indices
'''
