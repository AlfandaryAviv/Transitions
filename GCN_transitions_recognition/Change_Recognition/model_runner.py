import time
import os
from random import shuffle
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss, BCELoss, BCEWithLogitsLoss
import nni
import logging
from loggers import EmptyLogger, CSVLogger, PrintLogger, FileLogger, multi_logger
from model import Graphs_Rec
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
        self._loss = BCELoss()

    @property
    def logger(self):
        return self._logger

    @property
    def data_logger(self):
        return self._data_logger

    def my_loss(self, output, target, weights=None):
        output = torch.clamp(output, min=1e-8, max=1 - 1e-8)

        if weights is not None:
            assert len(weights) == 2
            loss = weights[1] * (target * torch.log(output)) + \
                   weights[0] * ((1 - target) * torch.log(1 - output))
        else:
            loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)
        ret = torch.neg(torch.mean(loss))
        return ret


    def accuracy(self,output, labels):
        #output = torch.sigmoid(output) ##todo USE it only with BCEWithLogit
        idxs_1_labeled = torch.where(labels == 1)
        answers = output[idxs_1_labeled]
        true_pos =torch.where(answers>=0.5) # tuple (,)
        return len(true_pos[0])/len(idxs_1_labeled[0])




    def _get_model(self):
        model = Graphs_Rec(in_features=self._conf["train_data"][0].shape[0],
                    hid_features=self._conf["hid_features"], out_features=1,
                    activation=self._conf["activation"], dropout= self._conf["dropout"])
        opt = self._conf["optimizer"](model.parameters(), lr=self._conf["lr"], weight_decay=self._conf["weight_decay"])
        ##checged : added "feature_matrices"
        return {"model": model, "optimizer": opt,
                "train_data": self._conf["train_data"],
                "training_labels": self._conf["training_labels"],
                "test_data": self._conf["test_data"],
                "test_labels": self._conf["test_labels"]}

    # verbose = 0 - silent
    # verbose = 1 - print test results
    # verbose = 2 - print train for each epoch and test results
    def run(self, verbose=2):
        if self._is_nni:
            verbose = 0
        model = self._get_model()
        ##
        loss_train, acc_train, intermediate_acc_test, losses_train, accs_train, test_results = self.train(
            self._conf["epochs"],
            model=model,
            verbose=verbose)
        ##
        # Testing
        result = self.test(model=model, verbose=verbose if not self._is_nni else 0, print_to_file=False)
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
        parameters = {"lr": self._conf["lr"],
                      "weight_decay": self._conf["weight_decay"],
                      "dropout": self._conf["dropout"], "optimizer": self._conf["optim_name"]}
        return loss_train, acc_train, intermediate_acc_test, result, losses_train, accs_train, test_results, parameters



    def train(self, epochs, model=None, verbose=2):
        loss_train = 0.
        acc_train = 0.
        losses_train = []
        accs_train = []
        test_results = []
        intermediate_test_acc = []
        for epoch in range(epochs):
            loss_train, acc_train = self._train(epoch, model, verbose)
            ##
            losses_train.append(loss_train)
            accs_train.append(acc_train)
            ##
            # /----------------------  FOR NNI  -------------------------
            if epoch % 5 == 0:
                test_res = self.test(model, verbose=verbose if not self._is_nni else 0)
                test_results.append(test_res)
                if self._is_nni:
                    test_acc = test_res["acc"]
                    intermediate_test_acc.append(test_acc)

        return loss_train, acc_train, intermediate_test_acc, losses_train, \
               accs_train, test_results


    def _train(self, epoch, model, verbose=2):
        #self._loss = self._loss = BCEWithLogitsLoss(torch.ones([223653]).to(self._device))

        model_ = model["model"]
        model_ = model_.to(self._device)
        optimizer = model["optimizer"]

        ###!
        labels = torch.from_numpy(model["training_labels"]).to(dtype=torch.float, device=self._device)
        labels = torch.DoubleTensor(model["training_labels"]).to(dtype=torch.float, device=self._device)  ###todo
        train = torch.from_numpy(model["train_data"]).to(dtype=torch.float, device=self._device)
        model_.train()
        optimizer.zero_grad()
        self._loss = self.my_loss

        ###send the model
        output = model_(train)
        ###

        loss_train = 0.
        labeld_1_num = len([b for b, item in enumerate(labels) if item == 1])
        output = output.view(output.shape[0])  ###todo!
        # loss_train += self._loss(output, labels, weights=[1,(len(train)-78)/78]) ##weights=[19/len(train),(len(train)-19)/len(train)])
        loss_train += self._loss(output, labels, weights=[len(train)/(len(train)-labeld_1_num),len(train)/labeld_1_num]) ##weights=[19/len(train),(len(train)-19)/len(train)])
        # loss_train /= len(train)
        loss_train.backward()
        optimizer.step()

        acc_train = self.accuracy(output, labels)

        if verbose == 2:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            self._logger.debug('Epoch: {:04d} '.format(epoch + 1) +
                               'loss_train: {:.4f} '.format(loss_train.data.item()) +
                               'acc_train: {:.4f} '.format(acc_train))
        return loss_train, acc_train




    def test(self, model=None, verbose=2, print_to_file=False):
        #self._loss=self._loss = BCEWithLogitsLoss(torch.ones([894618]).to(self._device))
        model_ = model["model"]
        model_ = model_.to(self._device)

        labels = torch.from_numpy(model["test_labels"]).to(dtype=torch.float, device=self._device)
        labels = torch.DoubleTensor(model["test_labels"]).to(dtype=torch.float, device=self._device)  ###todo###
        test = torch.from_numpy(model["test_data"]).to(dtype=torch.float, device=self._device)
        model_.eval()

        '''self._loss = self.my_loss
        pos_weight = torch.ones([len(test)]).to(self._device)  # All weights are equal to 1
        pos_weight *= 79 / (len(test) - 79)
        self._loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)'''


        ###send the model
        output = model_(test)
        ###

        output = output.view(output.shape[0])  ###todo!
        self._loss = self.my_loss
        loss_test = 0.
        loss_test += self._loss(output, labels)#, weights=[1, (len(test) - 20) / 20])

        #loss_test += self._loss(output, labels)
        #loss_test /= len(test)

        acc_test = self.accuracy(output, labels)

        if verbose != 0:
            self._logger.info("Test: loss= {:.4f} ".format(loss_test.data.item()) +
                              "acc= {:.4f}".format(acc_test))
        result = {"loss": loss_test.data.item(), "acc": acc_test}
        return result




def plot_graphs(info):
    # info[4] is list of train losses 1 . info[5] is list of train losses 2 - tempo. list[6] is list of acc train.
    #info [7] is list of dictionaries, each dictionary is for epoch, each one contains "loss" - first loss,"acc"- acc,  "tempo_loss" - tempo loss
    #info[8] is the temporal_oen
    parameters = info[7]
    regulariztion = str(parameters["weight_decay"])
    lr = str(parameters["lr"])
    #temporal_pen = str(parameters["temporal_pen"])# the origin reg first e
    optimizer = str(parameters["optimizer"])
    dropout = str(parameters["dropout"])

    #train

    # Share a X axis with each column of subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    plt.suptitle("Train: "+ "lr="+lr+" reg= "+regulariztion+ " dropout= "+dropout+" opt= "+optimizer, fontsize=16, y=0.99)

    epoch = [e for e in range(1, len(info[4])+1)]
    axes[0, 0].set_title('CE loss')
    axes[0, 0].set_xlabel("Iterations")
    axes[0, 0].set_ylabel("loss")
    axes[0, 0].plot(epoch, info[4])

    '''axes[0, 1].set_title('temporal loss')
    axes[0, 1].set_xlabel("Iterations")
    axes[0, 1].set_ylabel("loss")
    axes[0, 1].plot(epoch, info[5])'''

    axes[1, 1].set_title('accuracy')
    axes[1, 1].set_xlabel("Iterations")
    axes[1, 1].set_ylabel("accuracy")
    axes[1, 1].plot(epoch, info[5])
    fig.delaxes(axes[1,0])

    plt.savefig("figures_0.97/Train_all_y_"+"lr_"+lr+" reg= "+regulariztion+ " dr= "+dropout+" opt= "+optimizer+".png")

    #Test

    # Share a X axis with each column of subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    plt.suptitle("Test: tmplre=" +" lr="+lr+" reg= "+regulariztion+ " dropout= "+dropout+" opt= "+optimizer, fontsize=16, y=0.99)


    epoch = [e for e in range(1, len(info[6])+1)]
    test_ce_loss = [ info[6][i]["loss"] for i in range(len(info[6])) ]
    #test_tempo_loss = [ info[7][i]["tempo_loss"] for i in range(len(info[7])) ]
    acc_test =  [ info[6][i]["acc"] for i in range(len(info[6])) ]

    axes[0, 0].set_title('CE loss')
    axes[0, 0].set_xlabel("Iterations")
    axes[0, 0].set_ylabel("loss")
    axes[0, 0].plot(epoch, test_ce_loss)

    '''axes[0, 1].set_title('temporal loss')
    axes[0, 1].set_xlabel("Iterations")
    axes[0, 1].set_ylabel("loss")
    axes[0, 1].plot(epoch, test_tempo_loss)'''

    axes[1, 1].set_title('accuracy')
    axes[1, 1].set_xlabel("Iterations")
    axes[1, 1].set_ylabel("accuracy")
    axes[1, 1].plot(epoch, acc_test)
    fig.delaxes(axes[1,0])

    #fig.set_size_inches(3, 6, forward=True)
    plt.savefig("figures_0.97/Test_all_y_"+"lr_"+lr+" reg= "+regulariztion+ " dr= "+dropout+" opt= "+optimizer+".png")
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
        '''if idx_r == 0:
            plot_graphs(rs)'''
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


def build_model(training_data, training_labels, test_data, test_labels,
                hid_features, activation, optimizer, epochs, dropout, lr, l2_pen,
                dumping_name, is_nni=False):
    optim_name="SGD"
    if optimizer==optim.Adam:
        optim_name = "Adam"
    conf = {"hid_features": hid_features, "dropout": dropout, "lr": lr, "weight_decay": l2_pen,
            "train_data": training_data, "training_labels": training_labels,
            "test_data": test_data, "test_labels": test_labels,
            "optimizer": optimizer, "epochs": epochs,"activation": activation,"optim_name":optim_name}

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
                + ' optimizer= %s ' %optim_name)
    logger.debug('STARTING with lr=  {:.4f} '.format(lr) + ' dropout= {:.4f} '.format(dropout) + ' regulariztion_l2_pen= {:.4f} '.format(l2_pen)
        + ' optimizer= %s ' %optim_name)
    ##

    runner = ModelRunner(conf, logger=logger, data_logger=data_logger, is_nni=is_nni)
    return runner



def main_gcn(input_data, labels, hid_features, epochs=200, dropout=0.3, lr=0.01, l2_pen=0.005,
             iterations=1, dumping_name='',optimizer=optim.Adam, is_nni=False):
    runners = []
    for it in range(iterations):
        train_indices = []
        test_indices = []
        labeled_moved_data_indices = [b for b,item in enumerate(labels) if item==1]
        labeled_not_moved = [c for c,item in enumerate(labels) if item!=1]
        shuffle(labeled_moved_data_indices)
        shuffle(labeled_not_moved)

        train_indices+= labeled_moved_data_indices[:int(len(labeled_moved_data_indices) * 0.8)]
        train_indices+= labeled_not_moved[:int(len(labeled_not_moved) * 0.8)]
        test_indices += labeled_moved_data_indices[int(len(labeled_moved_data_indices) * 0.8):]
        test_indices+= labeled_not_moved[int(len(labeled_not_moved) * 0.8):]

        shuffle(train_indices)
        shuffle(test_indices)

        train = input_data[train_indices]
        test = input_data[test_indices]

        training_labels = labels[train_indices] #[labels[k] for k in train_indices]# todo: equal unless type?
        test_labels = labels[test_indices] #[labels[k] for k in test_indices]

        activation = torch.nn.functional.relu

        '''runner = build_model(training_features, training_labels, test_features, test_labels,
                             adj_matrices, hid_features, activation, optimizer, epochs, dropout, lr,
                             l2_pen, temporal_pen, dumping_name, is_nni=is_nni) #itay'''

        runner = build_model(training_data=train,training_labels= training_labels,test_data= test,test_labels= test_labels,
                             hid_features= hid_features,activation= activation,optimizer= optimizer,
                             epochs=epochs,dropout= dropout,lr= lr,
                             l2_pen= l2_pen, is_nni=is_nni, dumping_name=dumping_name)
        runners.append(runner)

    execute_runner(runners, is_nni=is_nni)
    return








