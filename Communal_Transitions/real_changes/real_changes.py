import os
import pickle
import networkx as nx
import numpy as np
from numpy.core.defchararray import find
from pathlib import Path



def load_data(data_name, time_inds):
    graphs = []
    labels = []
    for i in range(time_inds):
        with open(os.path.join('.', 'dataset', data_name,'pkl','gcn_input', 'graph_' + str(i) + '.pkl'), 'rb') as f:
            g = pickle.load(f)
        with open(os.path.join('.', 'dataset',data_name,'pkl','gcn_input', 'labels_' + str(i) + '.pkl'), 'rb') as f:
            l = pickle.load(f)
        graphs.append(g)
        labels.append(l)
    print("loaded data")
    return graphs,labels

def build_matrices(labels,data_name,num_of_labels):
    labels_writers = np.zeros((len(labels[0]), len(labels), num_of_labels))
    for i in range(labels_writers.shape[0]): #num of writers
        for year_index, labels_year in enumerate(labels):
            if labels_year[i] == -1:
                labels_writers[i][year_index] = np.array([-1]*num_of_labels)
            else:
                labels_writers[i][year_index] = np.array(labels_year[i])

    check("./dataset/" + data_name + "/pkl/real_changes_per_label/")
    with open("./dataset/" + data_name + "/pkl/real_changes_per_label/matrix_labels_per_person.pkl", 'wb') as handle:
        pickle.dump(labels_writers, handle, protocol=pickle.HIGHEST_PROTOCOL)



def recognize(arr, func):
    ret = []
    ind = 0
    while ind < len(arr):
        if func(arr[ind]):
            start = ind
            while ind<len(arr) and func(arr[ind]):
                ind += 1
            end = ind-1
            ret.append((start, end))
        ind += 1
    return ret


def clear_negs(persons_7_vector,idx):
    label_1 = 0.177
    negs = recognize(persons_7_vector[idx], lambda x:x<0)
    for neg_tuple in negs:
        if neg_tuple[0] == 0 and neg_tuple[1] == len(persons_7_vector[idx]) - 1:
            print("error")
            exit(0)
        if neg_tuple[0] == 0:
            if persons_7_vector[idx][neg_tuple[1] + 1] == 0:
                persons_7_vector[idx][neg_tuple[0]:neg_tuple[1] + 1] = 0
            else:
                persons_7_vector[idx][neg_tuple[0]:neg_tuple[1] + 1] = label_1
        elif neg_tuple[1] == len(persons_7_vector[idx]) - 1:
            if persons_7_vector[idx][neg_tuple[0] - 1] == 0:
                persons_7_vector[idx][neg_tuple[0]:neg_tuple[1] + 1] = 0
            else:
                persons_7_vector[idx][neg_tuple[0]:neg_tuple[1] + 1] = label_1
        else:
            if persons_7_vector[idx][neg_tuple[1] + 1] > 0 and persons_7_vector[idx][neg_tuple[0] - 1] > 0:
                persons_7_vector[idx][neg_tuple[0]:neg_tuple[1] + 1] = label_1
            else:
                persons_7_vector[idx][neg_tuple[0]:neg_tuple[1] + 1] = 0


def build_labels_7_tag(tag,data_name,time_inds):
    with open("./dataset/" + data_name + "/pkl/real_changes_per_label/matrix_labels_per_person.pkl", 'rb') as f:
        g = pickle.load(f)
    print(len(g))
    'for each person, take the 7th column'
    persons_7_vector = np.zeros((len(g), time_inds))
    persons_7_labeled = np.zeros(len(g))
    label_all_0 = 0
    label_all_1 = 0
    label_0_to_1 = 0
    label_1_to_0 = 0
    K=3

    for idx, person in enumerate(g):
        persons_7_vector[idx] = person.T[tag] # if we want to save this to file
        vec = persons_7_vector[idx]
        last=vec.copy()
        clear_negs(persons_7_vector,idx)
        #if the person is all 0
        if np.all(vec==0):
            label_all_0 +=1
            persons_7_labeled[idx] = 0
        #if it's all 1
        elif  not np.any(vec==0): #if it's all not zero
            label_all_1 += 1
            persons_7_labeled[idx] = 0 # 0 = not moved.
        #check for movement
        else:
            positives = recognize(vec,lambda x:x>0) #get the indexes of positives
            persons_7_labeled[idx] = 0
            for pos in positives:
                if (pos[1]-pos[0]+1) < K:  #less than k positive values, continue for next pos
                    continue
                else:
                    if len(vec[0:pos[0]])<K:  #if not enough elemnts from the start
                        continue
                    elif np.all(vec[pos[0]-K:pos[0]]==0):
                        label_0_to_1+=1
                        persons_7_labeled[idx] = 1
                        break

    print(label_all_0)
    print(label_all_1)
    print(label_0_to_1)

    print(len(persons_7_labeled[persons_7_labeled==1]))
    # with open("labels_per_writer_"+str(tag)+"_tag.pkl", 'wb') as handle:
    #     pickle.dump(persons_7_labeled, handle, protocol=pickle.HIGHEST_PROTOCOL)
    check("./dataset/"+data_name+"/pkl/real_changes_per_label/")
    with open("./dataset/"+data_name+"/pkl/real_changes_per_label/real_changes_label_"+str(tag)+"_tag.pkl", 'wb') as handle:
        pickle.dump(persons_7_labeled, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return persons_7_labeled

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


def main_real_changes(num_of_labels, data_name, time_inds):
    # distibution of tages: ([1118271, 13916, 18088, 20961, 2044, 10461, 1687, 1540, 5650, 4651, 9367, 2938, 6477, 4060, 10889])
    changes_by_label = []

    graphs, labels = load_data(data_name, time_inds)
    build_matrices(labels,data_name,num_of_labels)

    # check('./dataset/'+str(data_name)+'/pkl/real_changes_per_label/')
    for i in range(num_of_labels):
        x = build_labels_7_tag(i,data_name, time_inds)

        # with open('labels_per_writer_7_tag.pkl', 'rb') as f:
        #     labels = pickle.load(f)
        with open('./dataset/'+str(data_name)+'/pkl/real_changes_per_label/real_changes_label_' + str(i) + '_tag.pkl', 'rb') as f:
            labels = pickle.load(f)
        print(np.all(labels == 0))
        print(len(labels[labels == 1]))
        changes_by_label.append(x)
    return changes_by_label


if __name__ == "__main__":

    main_real_changes()