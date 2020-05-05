import os
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import sys
# sys.path.insert(1, "./transitions_main")
from transitions_main import check
def nodes_func(nodes_file,data_name):
    nodes = set()
    year_nodes = dict()
    nodes_year_labels = dict()

    for i, row in enumerate(nodes_file):
        node, year, label, count, percent_year = row.split(",")
        nodes.add(node)  # nodes old id
        year_nodes[year] = year_nodes.get(year, set()) | {node}  # {year: nodes_old_id}
        node_year_id = str(node) + "_" + str(year)
        nodes_year_labels[node_year_id] = nodes_year_labels.get(node_year_id, []) + [label] * int(count)

    old_to_new_nid = {old_id: i for i, old_id in enumerate(sorted(nodes,key=int))}  # {old_id: new_id}
    new_to_old_nid = {new_id: old_id for old_id, new_id in old_to_new_nid.items()}  # {new_id: old_id}

    pickle.dump(old_to_new_nid, open("./dataset/" + data_name + "/pkl/old_to_new_nid.pkl", "wb"))
    pickle.dump(new_to_old_nid, open("./dataset/" + data_name + "/pkl/new_to_old_nid.pkl", "wb"))


    nodes_id = set(k for k in new_to_old_nid.keys())  # set of new id

    year_nodeids = dict()  # {year: new node id}
    for year, l_nodes in year_nodes.items():
        for n in l_nodes:
            year_nodeids[year] = year_nodeids.get(year, set()) | set([old_to_new_nid[n]])

    year_new_nodeid_labels = dict()
    for key, val in nodes_year_labels.items():
        old = key.split("_")[0]
        n = old_to_new_nid[old]
        y = int(key.split("_")[1])
        if y not in year_new_nodeid_labels:
            year_new_nodeid_labels[y] = {}
        year_new_nodeid_labels[y][n] = val

    return year_nodeids, old_to_new_nid, nodes_id, year_new_nodeid_labels


def year_id_label_freq(year_new_nodeid_labels, num_of_labels):
    count_label = dict()
    for year in year_new_nodeid_labels.keys():
        if year not in count_label:
            count_label[year] = dict()
        for node, labels in year_new_nodeid_labels[year].items():
            l = [0] * num_of_labels
            value, counts = np.unique(labels, return_counts=True)
            for val, c in zip(value, counts):
                norm_counts = c / counts.sum()
                l[int(val)] = norm_counts
            count_label[year][node] = l
    return count_label


def create_tag_list_by_year(count_label, nodes_id):
    l = []
    years = sorted(list(count_label.keys()))
    for year in years:
        y = []
        for id in nodes_id:
            if id not in count_label[year]:
                y.append(-1)
            else:
                y.append(count_label[year][id])
        l.append(y)
    return l


def build_graphs(nodes_id, old_to_new_nid, edges_file, years_count,start_year):
    initial_g = nx.Graph()
    initial_g.add_nodes_from(nodes_id)
    g = [initial_g.copy() for _ in range(years_count)]
    all_edges_count = 0
    for line in edges_file:
        spline = line.split(',')  # Count right now not as weights, can be added if necessary
        year_idx = int(spline[2]) - start_year
        if spline[0] not in old_to_new_nid or spline[1] not in old_to_new_nid:
            continue
        else:
            all_edges_count += 1
            g[year_idx].add_edge(old_to_new_nid[spline[0]], old_to_new_nid[spline[1]])
    return g


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

def main_prep(dataset_name, e_path, n_path,num_of_labels,start_year, all_labeles_are_used='no'):
    # n_path = check("../dataset/" + str(dataset_name) + "/input_files/nodes.csv")
    # e_path = check("../dataset/" + str(dataset_name) + "/input_files/edges.csv")
    nodes_file = open(n_path)
    edges_file = open(e_path)
    next(nodes_file)
    next(edges_file)

    year_nodeids, old_to_new_nid, nodes_id, year_new_nodeid_labels = nodes_func(nodes_file, str(dataset_name))
    y_id_tag_dist = year_id_label_freq(year_new_nodeid_labels,num_of_labels)


    graphs = build_graphs(nodes_id, old_to_new_nid, edges_file, len(year_nodeids), start_year)
    labels = create_tag_list_by_year(y_id_tag_dist, nodes_id)
    check("./dataset/" + str(dataset_name) + "/pkl/gcn_input/")
    for i in range(len(graphs)):

        pickle.dump(graphs[i], open("./dataset/" + dataset_name +"/pkl/gcn_input/graph_" + str(i) + ".pkl", "wb"))
        pickle.dump(labels[i], open("./dataset/" + dataset_name + "/pkl/gcn_input/labels_" + str(i) + ".pkl", "wb"))

    return


if __name__ == "__main__":
    fnodes = open("nodes_little.csv", "rt")
    fedges = open("edges_little.csv", "rt")
    data_name = 'DBLP'
    main_prep(fnodes,fedges, data_name)
