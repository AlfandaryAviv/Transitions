import os
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import nni
from sklearn.metrics import f1_score
from loggers import EmptyLogger, CSVLogger, PrintLogger, FileLogger, multi_logger
from model import GCN
import pickle


class ModelRunner:
    def __init__(self, params, logger, data_logger=None, epochs_logger=None):
        self._logger = logger
        self._epoch_logger = epochs_logger
        self._data_logger = EmptyLogger() if data_logger is None else data_logger
        self._parameters = params
        self._lr = params["lr"]
        self._is_nni = params['is_nni']
        self._device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self._mse_loss = self.weighted_mse_loss
        self._temporal_loss = self.weighted_mse_loss
        self.model = GCN(num_of_features=self._parameters["feature_matrices"][0].shape[1],
                         hid_size=self._parameters["hid_size"],
                         num_of_classes=self._parameters["number_of_classes"],
                         activation=self._parameters["activation"],
                         dropout=self._parameters["dropout"])
        self.model = self.model.to(self._device)
        self.opt = self._parameters["optimizer"](self.model.parameters(),
                                                 lr=self._parameters['lr'],
                                                 weight_decay=self._parameters['weight_decay'])

    @property
    def logger(self):
        return self._logger

    @property
    def data_logger(self):
        return self._data_logger


    def weighted_mse_loss(self, pred, target, weights=None):
        if weights is None:
            return ((pred - target) ** 2).sum(dim=1).sum().to(self._device)
        elif self._parameters['loss_weights_type'] == 'sqrt(N/Nj)':
            weights = torch.tensor(weights).to(device=self._device, dtype=torch.float)
            b = (torch.sqrt((weights).sum() / weights) *(pred - target) ** 2).sum(dim=1).sum().to(self._device)
            return b
        elif self._parameters['loss_weights_type'] == '1/Njs':
            weights = torch.tensor(weights).to(device=self._device, dtype=torch.float)
            b = (torch.tensor(1. / weights) * (pred - target) ** 2).sum(dim=1).sum().to(self._device)
            return b


    def run(self):

        train_results_l = []
        test_results = []

        # train
        for epoch in range(self._parameters["epochs"]):
            train_results = self.train(epoch)
            train_results_l.append(train_results)

            if epoch == self._parameters["epochs"]-1: # for grid
                test_res = self.test(epoch)
                test_results.append(test_res)

                if self._is_nni:
                    nni.report_intermediate_result(test_res["f1_score_macro"][-1])

                else:
                    print(self._parameters["it_num"],
                                            self._parameters["iterations"],
                                            epoch + 1,
                                            self._parameters["epochs"],
                                            self._parameters["lr"],
                                            self._parameters["dropout"],
                                            self._parameters["hid_size"],
                                            self._parameters["weight_decay"],
                                            self._parameters["temporal_pen"],
                                            train_results['loss'].item(),
                                            train_results['tempo_loss'].item(),
                                            train_results['loss'].item() + train_results['tempo_loss'].item(),
                                            train_results['f1_score_macro'][-1],
                                            train_results['f1_score_micro'][-1],
                                            test_res["loss"],
                                            test_res["tempo_loss"],
                                            test_res["loss"] + test_res["tempo_loss"],
                                            test_res["f1_score_macro"][-1],
                                            test_res["f1_score_micro"][-1])


            self._logger.debug('Epoch: {:04d} '.format(epoch + 1) +
                               'lr: {:04f} '.format(self._parameters['lr']) +
                               'dropout: {:04f} '.format(self._parameters['dropout']) +
                               'hid_size: {:04f} '.format(self._parameters['hid_size']) +
                               'weight_decay: {:04f} '.format(self._parameters['weight_decay']) +
                               'temporal_pen: {:04f} '.format(self._parameters['temporal_pen']) +
                               'reg_loss_train: {:.4f} '.format(train_results['loss']) +
                               'temp_loss: {:.4f} '.format(train_results['tempo_loss']))

        result = self.test('test', print_to_file=True)

        if self._is_nni:
            nni.report_final_result(result["f1_score_macro"])

        return train_results_l, test_results, result, self._parameters

    def train(self, epoch):

        z_vals, outputs = [], []

        labeled_indices = self._parameters['training_inds']
        labels = self._parameters['training_labels']

        tempo_loss = 0.
        loss_train = 0.

        self.model.train()
        self.opt.zero_grad()

        for idx, adj in enumerate(self._parameters["adj_matrices"]):
            input_features = torch.from_numpy(self._parameters["feature_matrices"][idx]).to(dtype=torch.float,
                                                                                               device=self._device)
            z, output = self.model(input_features, adj)
            output = output[labeled_indices[idx], :]

            # Njs are the weights of the loss using the adj mx.
            # they should be either used or not.
            Nj_s = [sum([labels[u][t][j] for u in range(len(self._parameters["adj_matrices"])) for t in range(len(labels[u]))]) for j in
                                range(self._parameters['number_of_classes'])]

            loss_train += self._mse_loss(output, labels[idx], Nj_s)
            # loss_train += self._mse_loss(output, labels[idx].float())  #without weights using the build-in mse

            z_vals.append(z)  # After 1 GCN layer
            outputs.append(output)  # Final predictions


        # counts the number of cross_year_persons
        z_appearances = 0.
        for t in range(len(z_vals) - 1):

            t_inds = self._parameters['training_inds'][t]
            t_plus_one_inds = self._parameters['training_inds'][t + 1]
            z_inds = [i for i in t_inds if i in t_plus_one_inds]
            z_appearances += len(z_inds)
            z_val_t = z_vals[t][z_inds, :]
            z_val_t_plus_1 = z_vals[t+1][z_inds, :]
            loss = self._temporal_loss(z_val_t_plus_1, z_val_t)
            tempo_loss += self._parameters["temporal_pen"] * loss


        tempo_loss /= z_appearances
        loss_train /= sum([len(labeled_indices[u]) for u in range(len(outputs))])
        total_loss = loss_train + tempo_loss
        total_loss.backward()
        self.opt.step()

        f1_score_macro, f1_score_micro = [],[]

        if epoch == self._parameters['epochs']-1:
            for i in range(len(labels)):
                f1_mac, f1_mic, list_real, list_pred = self.accuracy_f1_score(outputs[i], labels[i])
                f1_score_macro.append(f1_mac)
                f1_score_micro.append(f1_mic)


        result = {"loss": loss_train,
                  "f1_score_macro": f1_score_macro,
                  "f1_score_micro": f1_score_micro,
                  "tempo_loss": tempo_loss}
        return result

    def test(self,epoch, print_to_file=False):

        z_vals, outputs = [], []
        labeled_indices = self._parameters['test_inds']
        labels = self._parameters['test_labels']

        tempo_loss = 0.
        loss_test = 0.
        test_z_appearances = 0.

        self.model.eval()
        for idx, adj in enumerate(self._parameters["adj_matrices"]):
            test_mat = torch.from_numpy(self._parameters["feature_matrices"][idx]).to(self._device)
            z, output = self.model(*[test_mat, adj])
            output = output[labeled_indices[idx], :]
            loss_test += self._mse_loss(output, labels[idx].float())
            z_vals.append(z)
            outputs.append(output)


        if print_to_file:
            grid_outputs_folder = str(self._parameters['name'])
            self._logger.debug("\nprint to files")
            for i in range(len(self._parameters["adj_matrices"])):
                np_output = outputs[i].cpu().data.numpy()
                products_path = os.path.join(os.getcwd(),'dataset',self._parameters["dataset_name"], "gcn_outputs",grid_outputs_folder)
                if not os.path.exists(products_path):
                    os.makedirs(products_path)
                with open(os.path.join("dataset",self._parameters["dataset_name"],"gcn_outputs",grid_outputs_folder, "gcn_" + str(i) + ".pkl"), "wb") as f:
                    pickle.dump(np_output, f, protocol=pickle.HIGHEST_PROTOCOL)


        for t in range(len(z_vals) - 1):
            t_inds = self._parameters['test_inds'][t]
            t_plus_one_inds = self._parameters['test_inds'][t + 1]
            z_inds = [i for i in t_inds if i in t_plus_one_inds]
            test_z_appearances += len(z_inds)
            z_val_t = z_vals[t][z_inds, :]
            z_val_t_plus_1 = z_vals[t + 1][z_inds, :]
            tempo_loss += self._parameters["temporal_pen"] * self._temporal_loss(z_val_t_plus_1, z_val_t)

        tempo_loss /= test_z_appearances
        loss_test /= sum([len(labeled_indices[u]) for u in range(len(outputs))])

        f1_score_macro, f1_score_micro = [], []
        real,pred = [],[]

        if epoch == self._parameters['epochs'] - 1 or epoch == 'test':
            for i in range(len(labels)): #running over the years
                f1_mac, f1_mic, list_real, list_pred = self.accuracy_f1_score(outputs[i], labels[i])
                f1_score_macro.append(f1_mac)
                f1_score_micro.append(f1_mic)
                real.extend(list_real)
                pred.extend(list_pred)

            self.confusion_matrix(real, pred) # of all years normalized to 1 for the last epoch test


        result = {"loss": loss_test.data.item(),
                  "f1_score_macro": f1_score_macro,
                  "f1_score_micro": f1_score_micro,
                  "tempo_loss": tempo_loss.data.item()}
        return result

    def accuracy_f1_score(self,output, labels):
        pred, real = [],[]
        for person in range(labels.size(0)): #range of all persons
            for label in range(labels.size(1)):
                if labels[person,label]==0:
                    continue
                else:
                    argmax = output[person].max(0)[1]
                    real.append(label)
                    pred.append(argmax.cpu().item())

        f1_macro = f1_score(real, pred, average='macro')
        f1_micro = f1_score(real, pred, average='micro')

        return f1_macro, f1_micro, real,pred

    def confusion_matrix(self, list_real, list_pred):
        matrix = np.zeros((self._parameters["number_of_classes"], self._parameters["number_of_classes"])) # classes X classes
        for i in range(len(list_pred)):
            matrix[list_real[i], list_pred[i]] += 1
        row_sums = matrix.sum(axis=1, dtype='float')
        new_matrix = np.zeros((self._parameters["number_of_classes"], self._parameters["number_of_classes"])) # classes X classes
        for i, (row, row_sum) in enumerate(zip(matrix, row_sums)):
            if row_sum == 0:
                new_matrix[i, :] = 0
            else:
                new_matrix[i, :] = row / row_sum

        new_matrix = np.around(new_matrix, 3)
        b = np.asarray(new_matrix)

        self._parameters['diag_sum'] = np.trace(b)
        self._parameters['diag_elements'] = np.diagonal(b)
        print('Diagonal (sum): ', np.trace(b))
        print('Diagonal (elements): ', np.diagonal(b))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(new_matrix, interpolation='nearest')
        fig.colorbar(cax)
        ax.set_yticks(plt.np.arange(self._parameters["number_of_classes"]))
        ax.set_yticklabels(i for i in range(self._parameters["number_of_classes"]))
        ax.set_xticks(plt.np.arange(self._parameters["number_of_classes"]))
        ax.set_xticklabels(i for i in range(self._parameters["number_of_classes"]))
        ax.tick_params(axis='y', labelsize=7)
        ax.tick_params(axis='x', labelsize=7, labelbottom=True, labeltop=False)
        plt.title('Confusion matrix')
        ax.axis('image')
        plt.xlabel("Predicted label")
        plt.ylabel("Real label")
        mypath = "./dataset/"+self._parameters["dataset_name"]+"/figures"
        if not os.path.exists(mypath):
            os.makedirs(mypath)
        plt.savefig("./dataset/"+self._parameters["dataset_name"]+"/figures/cofution_matrix"+str(self._parameters['name'])+time.strftime("%Y%m%d_%H%M%S")+".png")
        plt.clf()
        plt.close()
        return


def plot_graphs(train_results, test_results, parameters):
    train_ce_loss = [train_results[i]["loss"] for i in range(len(train_results))]
    train_tempo_loss = [train_results[i]["tempo_loss"] for i in range(len(train_results))]
    f1_score_macro_train = [train_results[i]["f1_score_macro"] for i in range(len(train_results))]
    f1_score_micro_train = [train_results[i]["f1_score_micro"] for i in range(len(train_results))]

    regulariztion = str(parameters["weight_decay"])
    lr = str(parameters["lr"])
    temporal_pen = str(parameters["temporal_pen"])
    dropout = str(parameters["dropout"])

    # train
    # Subplots colums share the same X axis
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    plt.suptitle(
        "Train: temp_pen=" + temporal_pen + " lr=" + lr + " reg= " + regulariztion + " dropout= " + dropout,
        fontsize=16, y=0.99)

    epoch = [e for e in range(1, len(train_results) + 1)]
    axes[0, 0].set_title('Regular Loss')
    axes[0, 0].set_xlabel("Iterations")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].plot(epoch, train_ce_loss)

    axes[0, 1].set_title('Temporal Loss')
    axes[0, 1].set_xlabel("Iterations")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].plot(epoch, train_tempo_loss)

    axes[1, 1].set_title('F1 Score Macro')
    axes[1, 1].set_xlabel("Iterations")
    axes[1, 1].set_ylabel("F1 Score Macro")
    axes[1, 1].plot(epoch, f1_score_macro_train)

    axes[1, 0].set_title('F1 Score Micro')
    axes[1, 0].set_xlabel("Iterations")
    axes[1, 0].set_ylabel("F1 Score Micro")
    axes[1, 0].plot(epoch, f1_score_micro_train)

    products_path = os.path.join(os.getcwd(),"dataset",parameters["dataset_name"], "figures")
    if not os.path.exists(products_path):
        os.makedirs(products_path)
    plt.savefig("./dataset/"+parameters["dataset_name"]+"figures/Train_all_y_" + "tmplre_" + temporal_pen + "lr_" + lr + " reg= " + regulariztion +
                " dr= " + dropout  +time.strftime("%Y%m%d_%H%M%S")+ ".png")
    plt.clf()
    plt.close()
    
    # Test
    # Subplots colums share the same X axis
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    plt.suptitle("Test: tmplre=" + temporal_pen + " lr=" + lr + " reg= " + regulariztion + " dropout= " + dropout , fontsize=16, y=0.99)

    epoch = [e for e in range(1, len(test_results) + 1)]
    test_ce_loss = [test_results[i]["loss"] for i in range(len(test_results))]
    test_tempo_loss = [test_results[i]["tempo_loss"] for i in range(len(test_results))]

    f1_score_macro_test = [test_results[i]["f1_score_macro"] for i in range(len(test_results))]
    f1_score_micro_test = [test_results[i]["f1_score_micro"] for i in range(len(test_results))]
    axes[0, 0].set_title('CE loss')
    axes[0, 0].set_xlabel("Iterations")
    axes[0, 0].set_ylabel("loss")
    axes[0, 0].plot(epoch, test_ce_loss)

    axes[0, 1].set_title('temporal loss')
    axes[0, 1].set_xlabel("Iterations")
    axes[0, 1].set_ylabel("loss")
    axes[0, 1].plot(epoch, test_tempo_loss)

    axes[1, 1].set_title('f1 score (macro) accuracy')
    axes[1, 1].set_xlabel("Iterations")
    axes[1, 1].set_ylabel("accuracy")
    axes[1, 1].plot(epoch, f1_score_macro_test)

    axes[1, 0].set_title('f1 score (micro) accuracy')
    axes[1, 0].set_xlabel("Iterations")
    axes[1, 0].set_ylabel("accuracy")
    axes[1, 0].plot(epoch, f1_score_micro_test)


    products_path = os.path.join(os.getcwd(),'dataset',parameters["dataset_name"], "figures")
    if not os.path.exists(products_path):
        os.makedirs(products_path)
    plt.savefig("./dataset/"+parameters["dataset_name"]+"figures/Test_all_y_" + "tmplre_" + temporal_pen + "lr_" + lr + " reg= " + regulariztion + " dr= " +
                dropout  + time.strftime("%Y%m%d_%H%M%S") + ".png")
    plt.clf()
    plt.close()


def execute_runners(runners, grid_logger, grid_logger_avg, is_nni=False):
    train_losses = []
    train_tempo_loss = []
    train_total_loss = []
    test_losses = []
    test_tempo_losses = []
    test_total_loss = []
    train_f1_macro, train_f1_micro = [], []
    test_f1_macro, test_f1_micro = [], []
    diag_sum, diag_elements=[],[]

    for idx_r, runner in enumerate(runners):

        train_results, test_results, last_run_test_result, parameters = runner.run()

        train_losses.append(train_results[-1]['loss'].item())
        train_f1_macro.append(np.mean(train_results[-1]['f1_score_macro']))
        train_f1_micro.append(np.mean(train_results[-1]['f1_score_micro']))
        train_tempo_loss.append(train_results[-1]['tempo_loss'].item())
        train_total_loss.append(train_results[-1]['loss'].item() + train_results[-1]['tempo_loss'].item())
        test_losses.append(last_run_test_result["loss"])
        test_f1_macro.append(np.mean(last_run_test_result["f1_score_macro"]))
        test_f1_micro.append(np.mean(last_run_test_result["f1_score_micro"]))
        test_tempo_losses.append(last_run_test_result["tempo_loss"])
        test_total_loss.append(last_run_test_result["loss"] + last_run_test_result["tempo_loss"])
        diag_sum.append(parameters['diag_sum'])
        diag_elements.append(parameters['diag_elements'])
        
        grid_logger.info(idx_r,
                         len(runners),
                         parameters['lr'],
                         parameters["dropout"],
                         parameters["weight_decay"],
                         parameters["hid_size"],
                         parameters["temporal_pen"],
                         parameters["epochs"],
                         train_results[-1]['loss'].item(),
                         train_results[-1]['tempo_loss'].item(),
                         train_results[-1]['loss'].item() + train_results[-1]['tempo_loss'].item(),
                         np.mean(train_results[-1]['f1_score_macro']),
                         np.mean(train_results[-1]['f1_score_micro']),
                         last_run_test_result["loss"],
                         last_run_test_result['tempo_loss'],
                         last_run_test_result["loss"] + last_run_test_result['tempo_loss'],
                         np.mean(last_run_test_result["f1_score_macro"][-1]),
                         np.mean(last_run_test_result["f1_score_micro"][-1]),
                         parameters['diag_sum'],
                         parameters['diag_elements'])

        # if idx_r == 0:
        #     plot_graphs(train_results, test_results, parameters)

    # mean and std for multiple iterations.
    runners[-1].logger.info("*" * 15 + "Final f1 macro train: %3.4f" % np.mean(train_f1_macro))
    runners[-1].logger.info("*" * 15 + "Final f1 micro train: %3.4f" % np.mean(train_f1_micro))
    runners[-1].logger.info("*" * 15 + "Std f1 macro train: %3.4f" % np.std(train_f1_macro))
    runners[-1].logger.info("*" * 15 + "Std f1 micro train: %3.4f" % np.std(train_f1_micro))
    runners[-1].logger.info("*" * 15 + "Final f1 macro test: %3.4f" % np.mean(test_f1_macro))
    runners[-1].logger.info("*" * 15 + "Final f1 micro test: %3.4f" % np.mean(test_f1_micro))
    runners[-1].logger.info("*" * 15 + "Std f1 macro train: %3.4f" % np.std(test_f1_macro))
    runners[-1].logger.info("*" * 15 + "Std accuracy train: %3.4f" % np.std(test_f1_micro))
    runners[-1].logger.info("Finished")
    

    grid_logger_avg.info(len(runners),
                         parameters['lr'],
                         parameters["dropout"],
                         parameters["hid_size"],
                         parameters["weight_decay"],
                         parameters["temporal_pen"],
                         parameters["epochs"],
                         np.mean(train_losses),
                         np.mean(train_tempo_loss),
                         np.mean(train_total_loss),
                         np.mean(train_f1_macro),
                         np.std(train_f1_macro),
                         np.mean(train_f1_micro),
                         np.std(train_f1_micro),
                         np.mean(test_losses),
                         np.mean(test_tempo_losses),
                         np.mean(test_total_loss),
                         np.mean(test_f1_macro),
                         np.std(test_f1_macro),
                         np.mean(test_f1_micro),
                         np.std(test_f1_micro),
                         np.mean(diag_sum),
                         np.mean(diag_elements,axis=0))

    return
