import sys
import pickle
import datetime
import time
import os
from collections import Counter
import networkx as nx
import numpy as np
import pandas as pd


# sys.path.insert(1,"/home/dsi/aviv/networks/graph-measures")
# sys.path.insert(1,"/home/dsi/aviv/networks/graphs-package/multi_graph")
# from multi_graph import MultiGraph

def sort_gnx(name):
    s_l = [str(v) for v in range(1990, 2011)]
    s_d = {name: i for i, name in enumerate(s_l)}
    return s_d[name]


def sort_by_years(src_file):
    mg_dict = {}
    for i, row in enumerate(src_file):
        if i % 10000 == 0:
            print("iteration number ", i)
        # print("\r" * 10 + str(int(i / 2000)) + "%", end="")
        src, dst, time, num = row.split(",")
        mg_dict[time] = mg_dict.get(time, []) + [(src, dst)]
    return mg_dict


def create_multigraph(mg_dict):
    mg = MultiGraph("dblp", graphs_source=mg_dict)
    mg.sort_by(sort_gnx)

    t = time.time()
    community_gnx, total_blue_edges = mg.community_graph()
    print(total_blue_edges)
    print("build gnx", time.time() - t)
    with open(os.path.join("..", "pkl", "community_gnx" + datetime.datetime.now().strftime("%d%m%Y_%H%M%S")),
              "wb") as community_gnx_f:
        pickle.dump(community_gnx, community_gnx_f)
    return mg, total_blue_edges


def nodes_labels(f):
    labels = {}  # dict(nodeid : labelid)
    year2nodes_dict = dict()
    for i, row in enumerate(f):
        node, year, label, count, percent_year = row.split(",")
        year2nodes_dict[year] = year2nodes_dict.get(year, []) + [node]
        node_id = str(node) + "_" + str(sort_gnx(year))
        # labels[node_id] = labels.get(node_id, set()) | set([label])
        labels[node_id] = labels.get(node_id, []) + [label]
    with open(os.path.join("..", "pkl", "nodes2labels" + datetime.datetime.now().strftime("%d%m%Y_%H%M%S")),
              "wb") as nodes2labels_f:
        pickle.dump(labels, nodes2labels_f)
    return labels


def nodes_func(nodes_file):
    nodes = set()
    year_nodes = dict()
    #year_nodes = {num: set() for num in [str(v) for v in range(1990, 2011)]}
    node_years = dict()
    nodes_year_labels = dict()

    # print("start iterate rows")
    for i, row in enumerate(nodes_file):
        # if i % 300000 == 0:
        #     print("-----------iteration", i)
        node, year, label, count, percent_year = row.split(",")

        nodes.add(node)  # nodes old id
        year_nodes[year] = year_nodes.get(year, set()) | set([node])  # {year: nodes_old_id}
        node_years[node] = node_years.get(node, set()) | set([year])  # {nodes_old_id: year}
        node_year_id = str(node) + "_" + str(year)
        nodes_year_labels[node_year_id] = nodes_year_labels.get(node_year_id, []) + [label] * int(count)

    old_to_new_nid = {old_id: i for i, old_id in enumerate(sorted(nodes))}  # {old_id: new_id}
    new_to_old_nid = {new_id: old_id for old_id, new_id in old_to_new_nid.items()}  # {new_id: old_id}

    nodes_id = set(k for k in new_to_old_nid.keys())  # set of new id

    year_nodeids = dict()  # {year: new node id}
    for year, l_nodes in year_nodes.items():
        for n in l_nodes:
            year_nodeids[year] = year_nodeids.get(year, set()) | set([old_to_new_nid[n]])

    nodeid_years = dict()  # {new_id: years}
    for n, years in node_years.items():
        nodeid_years[old_to_new_nid[n]] = years
    year_new_nodeid_labels = dict()
    for key, val in nodes_year_labels.items():
        old = key.split("_")[0]
        n = old_to_new_nid[old]
        y = int(key.split("_")[1])

        if y not in year_new_nodeid_labels:
            year_new_nodeid_labels[y] = {}
        year_new_nodeid_labels[y][n] = val

    return nodeid_years, year_nodeids, old_to_new_nid, new_to_old_nid, nodes_id, year_new_nodeid_labels


def year_id_label_freq(year_new_nodeid_labels):
    count_label = dict()
    label_freq = dict()
    l = []
    for year in year_new_nodeid_labels.keys():
        if year not in count_label:
            count_label[year] = dict()
        for node, labels in year_new_nodeid_labels[year].items():
            # most_common = Counter(labels).most_common()
            # count_label[year][node] = most_common
            l = [0] * 15
            value, counts = np.unique(labels, return_counts=True)
            for val, c in zip(value, counts):
                norm_counts = c / counts.sum()
                l[int(val)] = norm_counts
            count_label[year][node] = l

    e = 0
    # l=[0]*15
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


def build_graphs(nodes_id, old_to_new_nid, edges_file, years_count):
    initial_g = nx.Graph()
    initial_g.add_nodes_from(nodes_id)
    g = [initial_g.copy() for _ in range(years_count)]
    all_edges_count = 0
    for line in edges_file:
        spline = line.split(',')  # Count right now not as weights, can be added if necessary
        year_idx = int(spline[2]) - 1990
        if spline[0] not in old_to_new_nid or spline[1] not in old_to_new_nid:
            continue
        else:
            all_edges_count += 1
            g[year_idx].add_edge(old_to_new_nid[spline[0]], old_to_new_nid[spline[1]])
    t = -1
    return g


def main():
    nodes_file = open("nodes_little.csv", "rt")

    next(nodes_file)
    nodeid_years, year_nodeids, old_to_new_nid, new_to_old_nid, nodes_id, year_new_nodeid_labels = nodes_func(
        nodes_file)
    y_id_tag_dist = year_id_label_freq(year_new_nodeid_labels)
    edges_file = open("edges_little.csv", "rt")
    next(edges_file)
    graphs = build_graphs(nodes_id, old_to_new_nid, edges_file, len(year_nodeids))
    labels = create_tag_list_by_year(y_id_tag_dist, nodes_id)
    nodes_file.close()
    edges_file.close()

    # for i in range(len(graphs)):
    #     with open(os.path.join("graphs_by_years", "graph_" + str(i) + ".pkl"), 'wb') as handle:
    #         pickle.dump(graphs[i], handle, protocol=pickle.HIGHEST_PROTOCOL)
    #     with open(os.path.join("graphs_by_years", "labels_" + str(i) + ".pkl"), 'wb') as handle:
    #         pickle.dump(labels[i], handle, protocol=pickle.HIGHEST_PROTOCOL)
    return graphs, labels


if __name__ == "__main__":
    main()
