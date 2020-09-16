import cProfile
import datetime
import math
import operator
import time
from random import shuffle, sample
import community
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import pyplot as plt
import numpy as np
import collections
import pandas as pd
import sys
import pickle
import os
from pathlib import Path

#creating the temporal graph
# sys.path.insert(1, "../graph-measures")
sys.path.insert(1, "./multigraph/multi_graph")
from multi_graph import MultiGraph

# preparing the data for the GCN
# sys.path.insert(1, "./GCN_PyG/preparations")
# from preparations import main_prep
import preparations
from features import create_features

# GCN PyG model
sys.path.insert(1, "./GCN_PyG/main_gcn")
from main_gcn import run_trial
from main_gcn import GCNNet3,GCNNet2
# import main_gcn

# real transitions
sys.path.insert(1, "./real_changes/real_changes")
from real_changes import main_real_changes

# dataset parameters
Dataset_Name = 'IMDB'  # can be either DBLP or IMDB
current_nodes_path = "./dataset/" + str(Dataset_Name) + "/input_files/nodes.csv"  # loction of nodes input
current_edges_path = "./dataset/" + str(Dataset_Name) + "/input_files/edges.csv"  # location of edges input
Neighbors_Factor = 7
Timed_Factor = 7
Similarity_Factor = 128 #0.0007525
Communal_Resolution = 64 #184.1835
Dataset_time_range = [2006, 2016]   # [min_year, max year]
All_Dataset_Is_Labeled = 'yes'       # answer: 'yes'/'no'. DBLP = 'no', IMDB = 'yes'
Number_Of_Classes = 20              # on the DBLP there are 14 known labels (the rest are labeled 0)

# GCN Parameters
Net = GCNNet3 #either GCNNet3 (3 layers) or GCNNet2 (2 layers)
Epochs = 300
Dropout_Rate = 0.1
Hidden_Sizes = [10, 10]  #for the GCNNet2 only the Hidden_Sizes[0] is used
Learning_Rate = 0.0003
Weight_Decay = 0.0005
Time_Inds = int(Dataset_time_range[1])+1 - int(Dataset_time_range[0])

class Transitions(object):
    def __init__(self,nodes_path,edges_path):
        self.dataset_name = Dataset_Name
        self.neighbors_factor = Neighbors_Factor
        self.timed_factor = Timed_Factor
        self.similarity_factor = Similarity_Factor
        self.communal_resolution = Communal_Resolution
        self.dataset_time_range = Dataset_time_range
        self.time_inds = Dataset_time_range[1]+1 - Dataset_time_range[0]
        # self.time_inds=21
        self.number_of_unique_labels = Number_Of_Classes
        self.nodes_path = nodes_path
        self.edges_path = edges_path
        self.name = str("_time_" + str(self.timed_factor) +
                        "_sim_" + str(self.similarity_factor) +
                        "_neighbours_" + str(self.neighbors_factor) +
                        "_reso_" + str(self.communal_resolution))
        self.node_com_top_label = None
        self.labels = None
        self.train_person = None
        self.test_person = None
        self.train = None
        self.test = None
        self.community_gnx = None
        self.similarity_edges = None
        self.cd = None
        self.com_nodes = None
        self.top_label = None
        self.test_com_nodes = None
        self.com_size_test = None
        self.per_year_paint = dict()
        self.year_nodes = dict()
        self.person_tup = dict()
        self.test_person_painted = set()


    def sort_by_years(self):
        edges_file = open(self.edges_path)
        df = pd.read_csv(edges_file, skipinitialspace=True,
                         dtype={'n1': str, 'n2': str, 'year': str, 'count': int})
        mg_dict = df.groupby('year')[('n1', 'n2')].apply(lambda g: list(map(tuple, g.values.tolist()))).to_dict()
        print('done')
        return mg_dict

    def create_multigraph(self, mg_dict):
        mg = MultiGraph("dblp", graphs_source=mg_dict)
        mg.sort_by(self.sort_gnx)
        mg.suspend_logger()
        check("./dataset/" + str(self.dataset_name) + "/pkl/")
        pickle.dump(mg, open("./dataset/" + str(self.dataset_name) + "/pkl/mg_" + self.dataset_name + ".pkl", "wb"))
        mg.wake_logger()
        return mg

    def sort_gnx(self, name):
        s_l = [str(v) for v in range(self.dataset_time_range[0], self.dataset_time_range[1]+1)]
        s_d = {name: i for i, name in enumerate(s_l)}
        return s_d[name]

    def create_gnx(self, mg):
        t = time.time()
        community_gnx, total_timed_edges = mg.community_graph(cross_year_factor=self.timed_factor,
                                                                  neighbours_factor=self.neighbors_factor)
        check("./dataset/" + str(self.dataset_name) + "/pkl/")
        pickle.dump(community_gnx, open("./dataset/" + str(self.dataset_name) + "/pkl/community_gnx.pkl", "wb"))
        print("build gnx time: {:.5f}".format(time.time() - t))
        print("number of timed edges: ", total_timed_edges)
        return community_gnx, total_timed_edges

    def nodes_labels(self):
        nodes_file = open(self.nodes_path)
        next(nodes_file)  # skip first row
        labels = {}
        for i, row in enumerate(nodes_file):
            node, year, label, count, percent_year = row.split(",")
            node_id = str(node) + "_" + str(self.sort_gnx(year))
            if node_id not in labels:
                labels[node_id] = {}
            labels[node_id][label] = int(count)
        check("./dataset/" + str(self.dataset_name) + "/pkl/")
        pickle.dump(labels, open("./dataset/" + str(self.dataset_name) + "/pkl/labels.pkl", "wb"))
        nodes_set = set(labels.keys())
        pickle.dump(nodes_set, open("./dataset/" + str(self.dataset_name) + "/pkl/nodes_set.pkl", "wb"))

        return labels

    def train_test_split(self):
        if All_Dataset_Is_Labeled == 'no':
            persons, person_to_years = set(), {}
            for node, label in self.labels.items():
                all_zero = True
                for l in label:
                    if l != '0':
                        all_zero = False
                        break
                if all_zero:
                    continue
                person_id, year = node.split("_")
                persons.add(person_id)
                person_to_years[person_id] = person_to_years.get(person_id, []) + [node]
            data = list(persons)
            shuffle(data)
        elif All_Dataset_Is_Labeled == 'yes':
            persons, person_to_years = set(), {}
            for node, label in self.labels.items():
                person_id, year = node.split("_")
                persons.add(person_id)
                person_to_years[person_id] = person_to_years.get(person_id, []) + [node]
            data = list(persons)
            shuffle(data)


            train_person, test_person = data[:int(len(data) * 0.1)], data[int(len(data) * 0.1):]
            train_person, test_person = set(train_person), set(test_person)
            check("./dataset/" + str(self.dataset_name) + "/pkl/")
            pickle.dump(train_person, open("./dataset/" + str(self.dataset_name) + "/pkl/train_person.pkl", "wb"))
            pickle.dump(test_person, open("./dataset/" + str(self.dataset_name) + "/pkl/test_person.pkl", "wb"))
            new_train, new_test = set(), set()
            for person in train_person:
                for nodes in person_to_years[person]:
                    new_train.add(nodes)
            for person in test_person:
                for nodes in person_to_years[person]:
                    new_test.add(nodes)
            pickle.dump(new_train, open("./dataset/" + str(self.dataset_name) + "/pkl/train_per_year.pkl", "wb"))
            pickle.dump(new_test, open("./dataset/" + str(self.dataset_name) + "/pkl/test_per_year.pkl", "wb"))
            return train_person, test_person, new_train, new_test

    def com_det(self):  # apply community detection
        t = time.time()
        partition = community.best_partition(self.community_gnx,
                                             resolution=self.communal_resolution)  # dict(nodeid_yearid : communityid)
        pickle.dump(partition, open("./dataset/" + str(self.dataset_name) +
                                    "/pkl/cd_" + str(self.name) + ".pkl", "wb"))
        print("community detection time: {:.5f}".format(time.time() - t))
        return partition

    def cd_com_nodes(self):
        com_nodes = dict()  # (com:nodes_list)
        for node, com in self.cd.items():
            com_nodes[com] = com_nodes.get(com, set()) | {node}
        pickle.dump(com_nodes, open("./dataset/" + str(self.dataset_name) + "/pkl/com_nodes_" + str(self.name) + ".pkl", "wb"))
        return com_nodes

    def com_nodes_t(self, t_set):  # returns dict(com: train or test nodes)
        com_nodes_t = {com_id: t_set.intersection(com) for com_id, com in self.com_nodes.items() if
                       len(t_set.intersection(com)) != 0}  # dict(community_id : set(train_nodes))
        return com_nodes_t

    def com_label(self, com_nodes_t):
        com_label = dict()
        for comm, iter_nodes in com_nodes_t.items():
            for n in iter_nodes:
                com_label[comm] = com_label.get(comm, []) + [self.labels[n]]
        for com in com_label:
            com_label[com] = list(map(dict, sorted(list(i.items()) for i in com_label[com])))
        return com_label

    def paint_com(self, t_com_labels):
        top_label = dict()  # dict(communityid : most common label which is not 0
        for com, l_list in t_com_labels.items():
            unite_freqs = {}
            for item in l_list:
                for t in item:
                    unite_freqs[t] = unite_freqs.get(t, 0) + item[t]
            most_common = sorted([(t, unite_freqs[t]) for t in unite_freqs], key=lambda x: x[1], reverse=True)

            # top_label[com] = most_common[0][0] if len(most_common) == 1 or most_common[0][0] != '0' else most_common[1][0]
            if All_Dataset_Is_Labeled=='no':
                if len(most_common) == 1:
                    top_label[com] = most_common[0][0]
                elif len(most_common) > 1 and most_common[0][0] != '0':
                    top_label[com] = most_common[0][0]
                else:
                    top_label[com] = most_common[1][0]
            elif All_Dataset_Is_Labeled=='yes':
                top_label[com] = most_common[0][0]
        pickle.dump(top_label,open("./dataset/" + self.dataset_name + "/pkl/top_label.pkl", "wb"))
        return top_label

    def node_comlabel(self):
        node_comlabel = dict()  # dict(node id : most popular com label)
        for com in self.top_label:
            for node in self.com_nodes[com]:
                node_comlabel[node] = self.top_label[com]
        return node_comlabel

    def total_painting_accuracy(self):
        counter = 0
        e = 0
        for node in self.test:
            if node in self.node_com_top_label:
                for t in self.labels[node]:
                    if self.node_com_top_label[node] == t:
                        counter += 1
                        break
            elif node not in self.node_com_top_label:
                e += 1
        acc = ((counter / (len(self.test) - e))*100)
        return acc

    def check_communal_accuracy_t(self, t, com_size_t, name):
        # communal- real
        count_nodes = {}
        for node in self.node_com_top_label:
            my_com = self.cd[node]
            if node in t:
                for tag in self.labels[node]:
                    if tag == self.node_com_top_label[node]:
                        count_nodes[my_com] = count_nodes.get(my_com, 0) + 1
                        break
        com_accuracy = dict()
        for com, count in count_nodes.items():
            size = com_size_t[com]
            if size != 0:
                com_accuracy[com] = float((count / size) * 100)

        # random communities
        acc_com = []
        acc_rand = []
        com_size = []
        for com in com_accuracy:
            count_rand_nodes = 0
            random_pick = com_size_t[com]
            com_size.append(random_pick)
            acc_com.append(com_accuracy[com])
            test_sample = sample(self.test, k=random_pick)
            for node in test_sample:
                if node not in self.node_com_top_label:
                    # e+=1
                    continue
                for tag in self.labels[node]:
                    if self.node_com_top_label[node] == tag:
                        count_rand_nodes += 1
                        break
            random_sample_accuracy = float((count_rand_nodes / random_pick) * 100)
            acc_rand.append(random_sample_accuracy)

        plt.figure()
        plt.scatter(com_size, acc_com, color='red', marker='*', label='Community Accuracy')
        plt.scatter(com_size, acc_rand, color='blue', marker='*', label='Random Sample Accuracy')
        plt.title(str(name) + " Accuracy Scatter Plot")
        plt.xlabel("Community Test Size")
        plt.ylabel("Accuracy")
        plt.legend()
        check("./dataset/" + str(self.dataset_name) + "/plots/painting/")
        plt.savefig("./dataset/" + str(self.dataset_name) +
                    "/plots/painting/" + str(name) + "accuracy_" + str(self.name) +
                    datetime.datetime.now().strftime("%d%m%Y_%H%M%S") + ".png", bbox_inches='tight')
        plt.close()
        return

    def check_communal_entropy(self, t, com_size_t, name, t_com_label):
        com_entropy_dict = dict()
        for com in t_com_label:
            freqs = {}
            for item in t_com_label[com]:
                for tag in item:
                    freqs[tag] = freqs.get(tag, 0) + item[tag]
            x = self.ent_calculation(freqs)
            com_entropy_dict[com] = x

        com_size = []
        t_com_ent = []  # com entropy
        rand_com_ent = []  # randomly selected from train/test items entropy
        randcom_labels = dict()
        for com in com_entropy_dict:
            t_com_ent.append(com_entropy_dict[com])
            random_pick = com_size_t[com]
            com_size.append(random_pick)
            t_sample = sample(t, k=int(random_pick))
            rand_freqs = {}
            for node in t_sample:
                for label in self.labels[node]:
                    rand_freqs[label] = rand_freqs.get(label, 0) + self.labels[node][label]
            randcom_labels[com] = rand_freqs

        rcom_entropy_dict = {}
        for rand_com_id, labels_l in randcom_labels.items():
            y = self.ent_calculation(labels_l)
            rcom_entropy_dict[rand_com_id] = y
        for rcom, entrop in rcom_entropy_dict.items():
            rand_com_ent.append(entrop)

        plt.figure()
        plt.scatter(com_size, t_com_ent, color='red', marker='*', label='Communal Entropy')
        plt.scatter(com_size, rand_com_ent, color='blue', marker='*', label='Random Sample Entropy')
        plt.title(str(name) + " Entropy Scatter Plot")
        plt.xlabel("Community Size")
        plt.ylabel("Entropy")
        plt.legend()
        check("./dataset/" + str(self.dataset_name) + "/plots/painting/")
        plt.savefig("./dataset/" + str(self.dataset_name) +
                    "/plots/painting/" + str(name) + "entropy_time_" + self.name +
                    datetime.datetime.now().strftime("%d%m%Y_%H%M%S") + ".png", bbox_inches='tight')
        plt.close()
        return

    def ent_calculation(self, freq_d, base=None):
        base = 2 if base is None else base
        value, counts = freq_d.keys(), freq_d.values()
        counts = np.array(list(counts))
        # value, counts = np.unique(freq_l, return_counts=True)
        norm_counts = counts / sum(counts)
        return -(norm_counts * np.log(norm_counts) / np.log(base)).sum()

    def transitions_results(self):
        for community in self.test_com_nodes:
            for node in self.test_com_nodes[community]:
                person, year = node.split("_")
                year = int(year)
                if community in self.top_label:
                    self.test_person_painted.add(person)
                    if person not in self.person_tup:
                        self.person_tup[person] = list()
                    if person not in self.per_year_paint:
                        self.per_year_paint[person] = dict()
                    my_tuple = (year, self.top_label[community])
                    self.person_tup[person].append(my_tuple)
                    self.year_nodes[year] = self.year_nodes.get(year, []) + [self.top_label[community]]
                    self.per_year_paint[person][year] = self.per_year_paint[person].get(year, self.top_label[community])

        pickle.dump(self.test_person_painted, open("./dataset/"+self.dataset_name+"/pkl/test_person_painted.pkl", "wb"))
        pickle.dump(self.person_tup, open("./dataset/" + self.dataset_name + "/pkl/person_tup.pkl", "wb"))
        pickle.dump(self.year_nodes, open("./dataset/" + self.dataset_name + "/pkl/year_nodes.pkl", "wb"))
        pickle.dump(self.per_year_paint, open("./dataset/" + self.dataset_name + "/pkl/per_year_paint.pkl", "wb"))

        tp_all, tn_all, fp_all, fn_all = [], [], [], []
        # self.plot_compaint()
        # self.plot_changes_per_year()
        check('./dataset/' + self.dataset_name + '/transitions_results/')
        f_results = open('./dataset/' + self.dataset_name + '/transitions_results/communities_results' +
                         datetime.datetime.now().strftime("%d%m%Y_%H%M%S") + '.csv', 'w')
        f_results2 = open('./dataset/' + self.dataset_name + '/transitions_results/communities_results_avg' +
                          datetime.datetime.now().strftime("%d%m%Y_%H%M%S") + '.csv', 'w')

        f_results2.write('Timed Weight' + '\t' + 'Similarity Weight' + '\t' + 'Neighbors Weight' + '\t' + 'Communal Resolution' + '\t'
                         + str.upper('tp') + '\t' + str.upper('fp') + '\t' + str.upper('fn') + '\t' + str.upper('tn') + '\t'
                         + str.upper('tp/fp') + '\t' + str.upper('tp/fn') + '\t' + str.upper('tp/(fn + tp)')
                         + '\t' + str.upper('tp/(fp + tp)') + '\t' + 'Person Painted' + '\t' + 'Person In Test' + '\t' + 'Painted/Test' '\n')

        real_transitions = main_real_changes(self.number_of_unique_labels, self.dataset_name,self.time_inds) #creates pickles of the real transitions for each label
        for k in range(0, self.number_of_unique_labels):
            print("tag: ", k)
            transition_real = real_transitions[k] # a transition of a label
            # matrix = np.zeros((2, 2))
            pred_negative, pred_positive = self.prediction_pos_neg(k)
            real_negative, real_positive = self.real_tag_changes(transition_real)
            print("pred neg:", len(pred_negative))
            print("pred pos:", len(pred_positive))
            print("real neg:", len(real_negative))
            print("real pos:", len(real_positive))

            inter_pred = pred_negative.intersection(pred_positive)
            inter_real = real_negative.intersection(real_positive)

            union_pred = pred_negative.union(pred_positive)
            union_real = real_negative.union(real_positive)

            missing_in_real = set(union_pred).difference(union_real)
            missing_in_pred = set(union_real).difference(union_pred)
            e=0


            tp, fp, fn, tn = 0, 0, 0, 0
            for i in pred_positive:
                if i in real_positive:
                    tp += 1
                if i in real_negative:
                    fp += 1

            for i in pred_negative:
                if i in real_positive:
                    fn += 1
                if i in real_negative:
                    tn += 1

            total = tp + fn + fp + tn

            print("total:", total)
            print(" ")

            if k == 0:
                f_results.write('Timed' + '\t' + 'Similarity' + '\t' + 'Neighbors' + '\t' + 'Resolution' + '\t'
                                + 'Tag' + '\t' + str.upper('tp') + '\t' + str.upper('fp') + '\t' + str.upper('fn') + '\t' + str.upper('tn') + '\t'
                                + str.upper('tp/fp') + '\t' + str.upper('tp/fn') + '\t' + str.upper('tp/(fn + tp)')
                                + '\t' + str.upper('tp/(fp + tp)') + '\t' + 'Real Pos' + '\t' + 'Real Neg' + '\t' + 'Pred pos' + '\t' + 'Pred neg' + '\n')


            f_results.write(str(self.timed_factor) + '\t' + str(self.similarity_factor) + '\t' +
                            str(self.neighbors_factor) + '\t' + str(self.communal_resolution) + '\t' +
                            str(k) + '\t' + str(float(tp / total)) + '\t' + str(float(fp / total)) +
                            '\t' + str(float(fn / total)) + '\t' + str(float(tn / total)) + '\t' +
                            str((tp / fp) if fp != 0 else 0) + '\t' + str((tp / fn) if fn != 0 else 0) + '\t' +
                            str((tp / (tp + fn)) if (tp + fn) != 0 else 0) + '\t' +
                            str((tp / (tp + fp)) if (tp + fp) != 0 else 0) + '\t' +
                            str(len(real_positive)) + '\t' + str(len(real_negative)) + '\t' +
                            str(len(pred_positive)) + '\t' + str(len(pred_negative)) + '\n')

            tp_all.append(float((tp / total) if total != 0 else 0))
            tn_all.append(float((tn / total) if total != 0 else 0))
            fp_all.append(float((fp / total) if total != 0 else 0))
            fn_all.append((float(fn / total) if total != 0 else 0))

            tp_avg = float((sum(tp_all) / len(tp_all)) if len(tp_all) != 0 else 0)
            fp_avg = float((sum(fp_all) / len(fp_all)) if len(fp_all) != 0 else 0)
            fn_avg = float((sum(fn_all) / len(fn_all)) if len(fn_all) != 0 else 0)
            tn_avg = float((sum(tn_all) / len(tn_all)) if len(tn_all) != 0 else 0)

            if k == self.number_of_unique_labels-1:
                f_results2.write(str(self.timed_factor) + '\t' + str(self.similarity_factor) + '\t' +
                                 str(self.neighbors_factor) + '\t' + str(self.communal_resolution) + '\t' +
                                 str(tp_avg) + '\t' + str(fp_avg) + '\t' +
                                 str(fn_avg) + '\t' + str(tn_avg) + '\t' +
                                 str(tp_avg / fp_avg) + '\t' + str(tp_avg / fn_avg) + '\t' +
                                 str(tp_avg / (tp_avg + fn_avg)) + '\t' +
                                 str(tp_avg / (tp_avg + fp_avg)) + '\t' +
                                 str(total) + '\t' + str(len(self.test_person)) + '\t' + str(
                    total / len(self.test_person)) + '\n')
                f_results.write('\n')
        return

    def plot_compaint(self):
        biggest_community = max(self.com_size_test.items(), key=operator.itemgetter(1))[1]
        largest_power_of_2 = math.ceil(math.log2(biggest_community))
        bins = np.logspace(0, largest_power_of_2, num=largest_power_of_2+1, base=2.0)
        com_count = {}
        com_bin = {}
        bin_list, all, communityid, paint_list = [], [], [], []

        for community in self.test_com_nodes:
            com_count[community] = len(self.test_com_nodes[community])

        for com, size in com_count.items():
            if com in self.top_label:
                for i in range(0, largest_power_of_2 - 1):
                    b1 = bins[i]
                    b2 = bins[i + 1]
                    if size >= b1 and size < b2:
                        if size > 0:
                            com_bin[b2] = com_bin.get(b2, 0) + 1
                            paint_list.append(self.top_label[com])
                            communityid.append(str(com))
                            bin_list.append(b2)

        colors_list = ['black', 'red', 'tan', 'deepskyblue', 'blue', 'springgreen', 'pink', 'yellow', 'silver',
                           'darkorange', 'cyan', 'deeppink', 'salmon', 'teal', 'mediumpurple', 'rosybrown',
                       'lime', 'lavender','royalblue','darkgreen','indigo','darkviolet','moccasin','gold','olive','brown']

        map = matplotlib.colors.ListedColormap(colors_list)

        paint_color_com = []
        for i in paint_list:
            paint_color_com.append(colors_list[int(i)])

        partial = []
        for bin in bin_list:
            partial.append(1 / com_bin[bin])

        all.append(bin_list)
        all.append(partial)
        all.append(communityid)  # unique id
        # all.append(paint_color_com) #color of com in plot
        all.append(paint_list)

        rows = zip(all[0], all[1], all[2], all[3])
        headers = ['box', 'Value', 'id', 'paint_com']

        df = pd.DataFrame(rows, columns=headers)
        # print(df.head())
        # check('./dataset/'+self.dataset_name+'/df_color_plot/')
        df.to_csv('./dataset/'+self.dataset_name+'/plots/df' + '_' + str(self.name) + '_' + datetime.datetime.now().strftime("%d%m%Y_%H%M%S") + '.csv',
                  index=False)
        # print(df.head())

        df = df.sort_values(by=['box', 'paint_com'])

        df.pivot_table(index='box', columns='paint_com', values='Value',
                       aggfunc='sum', fill_value=0).plot.bar(stacked=True, legend=True, colormap=map)

        print(df.head())
        plt.tight_layout()
        plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        plt.xlabel('Community Size', fontsize='large')
        # plt.show()
        check("./dataset/" + str(self.dataset_name) + "/plots/transitions/")
        plt.savefig("./dataset/" + str(self.dataset_name) + "/plots/transitions/colors_" + str(self.name) + "_" +
                    datetime.datetime.now().strftime("%d%m%Y_%H%M%S") + ".png", bbox_inches='tight')
        # plt.close()

        return

    def plot_changes_per_year(self):

        change = dict()
        for person, years in self.per_year_paint.items():
            od = collections.OrderedDict(sorted(years.items()))
            nl, cl = [], []
            for year, painted_com in od.items():
                cl.append(painted_com)
                nl.append(year)
            for i in range(len(cl) - 1):
                if cl[i] != cl[i + 1]:
                    change[nl[i+1]] = change.get(nl[i+1], 0) + 1
        year_count_painted = {year: len(self.year_nodes[year]) for year in self.year_nodes}

        y, percent = [], []
        for year in change:
            y.append(year)
            percent.append((change[year] / year_count_painted[year])*100)

        plt.figure()
        plt.bar(y, percent, align='center')
        plt.xlabel('year')
        plt.ylabel('percent from test changed community')
        plt.tight_layout()
        # plt.show()
        check("./dataset/" + str(self.dataset_name) + "/plots/transitions/")
        plt.savefig("./dataset/" + str(self.dataset_name) +
                    "/plots/transitions/changes_per_year_" + str(self.name) + "_" +
                    datetime.datetime.now().strftime("%d%m%Y_%H%M%S") + ".png", bbox_inches='tight')
        plt.close()

        return

    def prediction_pos_neg(self, tag):
        positive, negative = set(), set()
        person_time_paint = dict()
        for person, tuples in self.person_tup.items():
            num_of_years = [0] * len(tuples)
            sorted_by_year = sorted(tuples)
            for i in range(len(self.person_tup[person])):
                num_of_years[i] = sorted_by_year[i][1]
            person_time_paint[person] = num_of_years
            c = 0
            if len(set(person_time_paint[person])) == 1:  # if all the tags are the same or only one tag
                negative.add(person)
            elif str(tag) not in person_time_paint[person]:  # if tag does not appear
                negative.add(person)
            elif len(person_time_paint[person]) > 1:
                if person_time_paint[person][0] != str(tag):
                    if any(person_time_paint[person][i] == str(tag) for i in range(1, len(person_time_paint[person]))):
                        positive.add(person)
                    else:
                        negative.add(person)
                elif person_time_paint[person][0] == str(tag):
                    counter = 0
                    for previous, current in zip(person_time_paint[person], person_time_paint[person][1:]):
                        counter += 1

                        if current != previous and current != str(tag):
                            c += 1
                            if counter == (len(person_time_paint[person]) - 1):
                                negative.add(person)

                        elif c > 0 and current == str(tag):
                            positive.add(person)
                            break
                        elif c == 0 and counter == (len(person_time_paint[person]) - 1):
                            negative.add(person)
                            break
                        elif c > 0 and current != str(tag):
                            if counter == (len(person_time_paint[person]) - 1):
                                negative.add(person)
        return negative, positive

    def real_tag_changes(self, real_changes_pkl):
        new_to_old_id = pickle.load(open("./dataset/"+self.dataset_name+"/pkl/new_to_old_nid.pkl", "rb"))
        real_changes_pkl.tolist()
        pos, neg = set(), set()
        for i in range(len(real_changes_pkl)):
            if new_to_old_id[i] in self.test_person_painted:
                if real_changes_pkl[i] == 0:
                    neg.add(str(new_to_old_id[i]))
                elif real_changes_pkl[i] == 1:
                    pos.add(str(new_to_old_id[i]))
        return neg, pos

    def create_missing_folder(self,file_path,current_path=False):
        _=check(file_path,current_path)
        return


    def main_transitions(self):

        self.labels = self.nodes_labels() #labels(node,(label, amount))
        self.train_person, self.test_person, self.train, self.test = self.train_test_split()

        # self.train_person = pickle.load(open("./dataset/"+self.dataset_name+"/pkl/train_person.pkl","rb"))
        # self.test_person = pickle.load(open("./dataset/" + self.dataset_name + "/pkl/test_person.pkl", "rb"))
        # self.train = pickle.load(open("./dataset/" + self.dataset_name + "/pkl/train_per_year.pkl", "rb"))
        # self.test = pickle.load(open("./dataset/" + self.dataset_name + "/pkl/test_per_year.pkl", "rb"))
        # self.labels = pickle.load(open("./dataset/" + self.dataset_name + "/pkl/labels.pkl", "rb"))

        #creating the network, connecting the neighbors edges and the timed edges
        mg_dict = self.sort_by_years()
        mg = self.create_multigraph(mg_dict)
        self.community_gnx, total_timed_edges = self.create_gnx(mg)

        # creating input for graphs and labels for the GCN (graph per timestamp, labels, features matrices)
        preparations.main_prep(self.dataset_name, self.edges_path, self.nodes_path,self.number_of_unique_labels, self.dataset_time_range[0])
        create_features(self.dataset_name, self.time_inds) # creating feature matrices for the GCN

        params_ = {"data_name": self.dataset_name,  # parameters of the GCN model
                   "net": Net,
                   "epochs": Epochs,
                   "activation": "relu",
                   "dropout_rate": Dropout_Rate,
                   "hidden_sizes": Hidden_Sizes,
                   "learning_rate": Learning_Rate,
                   "weight_decay": Weight_Decay,
                   "time_ins": Time_Inds,
                   "num_of_classes": self.number_of_unique_labels
                   }
        self.similarity_edges = run_trial(params_) #runs the GCN model

        t = time.time()
        for u, v in self.similarity_edges: #adding the similarity edges to the graph with similarity factor weight
            self.community_gnx.add_edge(u, v, weight=self.similarity_factor)
        print("adding similarity edges time: ", time.time() - t)
        self.cd = self.com_det()
        self.com_nodes = self.cd_com_nodes()
        print("number of communities: ", len(self.com_nodes))

        # self.cd = pickle.load(open("./dataset/" + self.dataset_name + "/pkl/cd_" + str(self.name) + ".pkl", "rb"))
        # self.com_nodes = pickle.load(open("./dataset/" + self.dataset_name + "/pkl/com_nodes_" + str(self.name) + ".pkl", "rb"))
        # print("number of communities: ", len(self.com_nodes))

        # paint communities
        t = time.time()
        train_com_nodes = self.com_nodes_t(self.train)
        train_com_labels = self.com_label(train_com_nodes)
        self.top_label = self.paint_com(train_com_labels)
        print("number of communities painted: ", len(self.top_label))
        print("paint communities time: {:.4f}".format(time.time() - t))

        self.test_com_nodes = self.com_nodes_t(self.test)
        test_com_label = self.com_label(self.test_com_nodes)
        # top_label_test = self.paint_com(test_com_label)

        self.com_size_test = {com: len(test_com_label[com]) for com in test_com_label}
        com_size_train = {com: len(train_com_labels[com]) for com in train_com_labels}

        # Accuracy
        self.node_com_top_label = self.node_comlabel()
        # accuracy train
        t = time.time()
        self.check_communal_accuracy_t(self.train, com_size_train, "train")
        print("accuracy train time: {:.5f}".format(time.time() - t))
        # accuracy test
        t = time.time()
        self.check_communal_accuracy_t(self.test, self.com_size_test, "test")
        print("accuracy test time: {:.5f}".format(time.time() - t))
        # total accuracy
        total_accuracy = self.total_painting_accuracy()
        print("total accuracy in painted communities: {:.5f}".format(total_accuracy))

        # Entropy
        # entroy train
        t = time.time()
        self.check_communal_entropy(self.train, com_size_train, "train", train_com_labels)
        print("entropy train time: {:.5f}".format(time.time() - t))
        # entropy test
        t = time.time()
        self.check_communal_entropy(self.test, self.com_size_test, "test", test_com_label)
        print("entropy test time: {:.5f}".format(time.time() - t))

        # transitions
        self.transitions_results()
        self.plot_compaint()
        self.plot_changes_per_year()

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
    pr = cProfile.Profile()
    pr.enable()

    input_nodes_path = "./dataset/" + str(Dataset_Name) + "/input_files/nodes.csv"  # loction of input
    input_edges_path = "./dataset/" + str(Dataset_Name) + "/input_files/edges.csv"
    nodes_path = check(input_nodes_path, current_nodes_path)
    edges_path = check(input_edges_path, current_edges_path)

    total_time = time.time()
    trans = Transitions(str(nodes_path),str(edges_path))
    trans.main_transitions()
    print("total time", time.time() - total_time)

    pr.disable()
    pr.print_stats(sort="time")
