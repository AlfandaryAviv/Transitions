import os
import pickle
from random import shuffle
import networkx as nx
import numpy as np
from numpy.core.defchararray import find

GRAPHS_NUM = 21

'''
Building the labels for testing the model (the GCN's outputs). These labels are to define
whether a person has changed his writing topic, according to our decision what is defined as a change.
In our case, transition is when the vector (the 21 years vector each person) is in the form of including 
k times 0 and then at least k times 1. 
For handling the -1 cases (so we don't miss cases), we used the following rule:
if a time point is surrounded by 1 it's 1, if it's surrounded with 0 it's 1, and if it's surrounded one side by 1
 and in the other one by 0 - it's 0. 
'''

def load_data():
    graphs = []
    labels = []
    for i in range(GRAPHS_NUM):
        with open(os.path.join('graphs_by_years', 'graph_' + str(i) + '.pkl'), 'rb') as f:
            g = pickle.load(f)
        with open(os.path.join('graphs_by_years', 'labels_' + str(i) + '.pkl'), 'rb') as f:
            l = pickle.load(f)
        graphs.append(g)
        labels.append(l)
    print("loaded data")
    return graphs,labels

def build_matrices(labels):
    labels_writers = np.zeros((len(labels[0]), len(labels), 15))
    for i in range(labels_writers.shape[0]): #num of writers
        for year_index, labels_year in enumerate(labels):
            if labels_year[i] == -1:
                labels_writers[i][year_index] = np.array([-1]*15)
            else:
                labels_writers[i][year_index] = np.array(labels_year[i])

    with open("matrix_labels_per_writer.pkl", 'wb') as handle:
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
    negs = recognize(persons_7_vector[idx], lambda x:x<0)
    for neg_tuple in negs:
        if neg_tuple[0] == 0 and neg_tuple[1] == len(persons_7_vector[idx]) - 1:
            print("error")
            exit(0)
        if neg_tuple[0] == 0:
            if persons_7_vector[idx][neg_tuple[1] + 1] == 0:
                persons_7_vector[idx][neg_tuple[0]:neg_tuple[1] + 1] = 0
            else:
                persons_7_vector[idx][neg_tuple[0]:neg_tuple[1] + 1] = 0.177
        elif neg_tuple[1] == len(persons_7_vector[idx]) - 1:
            if persons_7_vector[idx][neg_tuple[0] - 1] == 0:
                persons_7_vector[idx][neg_tuple[0]:neg_tuple[1] + 1] = 0
            else:
                persons_7_vector[idx][neg_tuple[0]:neg_tuple[1] + 1] = 0.177
        else:
            if persons_7_vector[idx][neg_tuple[1] + 1] > 0 and persons_7_vector[idx][neg_tuple[0] - 1] > 0:
                persons_7_vector[idx][neg_tuple[0]:neg_tuple[1] + 1] = 0.177
            else:
                persons_7_vector[idx][neg_tuple[0]:neg_tuple[1] + 1] = 0


def build_labels_7_tag():
    with open('matrix_labels_per_writer.pkl', 'rb') as f:
        g = pickle.load(f)
    print(len(g))
    'for each person, take the 7th column'
    persons_7_vector = np.zeros((len(g), GRAPHS_NUM))
    persons_7_labeled = np.zeros(len(g))
    label_all_0 = 0
    label_all_1 = 0
    label_0_to_1 = 0
    label_1_to_0 = 0
    K=3

    for idx, person in enumerate(g):
        persons_7_vector[idx] = person.T[7] # if we want to save this to file
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
            positives = recognize(vec, lambda x:x>0) #get the indexes of positives
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
    with open("labels_per_writer_7_tag.pkl", 'wb') as handle:
        pickle.dump(persons_7_labeled, handle, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == "__main__":


    graphs, labels = load_data()
    build_matrices(labels)
    build_labels_7_tag()

    '''some checks:
    
    with open('labels_per_writer_7_tag.pkl', 'rb') as f:
        labels = pickle.load(f)
        with open('input_7_tag.pkl', 'rb') as f:
            input_data = pickle.load(f)


    train_indices = []
    test_indices = []
    labeled_moved_data_indices = [b for b, item in enumerate(labels) if item == 1]
    labeled_not_moved = [c for c, item in enumerate(labels) if item != 1]
    print("check")
    errors=0
    for i in (labeled_moved_data_indices):
        a= input_data[i]
        for j in (labeled_not_moved):
            if i == j:
                continue
            b = input_data[j]
            if (a==b).all():
                print(i, j , "indexex")
                #print("a", a)
                #print("b", b)
                print("error")
                errors+=1
                break
    print("finish")
    shuffle(labeled_moved_data_indices)
    shuffle(labeled_not_moved)

    train_indices += labeled_moved_data_indices[:int(len(labeled_moved_data_indices) * 0.2)]
    train_indices += labeled_not_moved[:int(len(labeled_not_moved) * 0.2)]
    test_indices += labeled_moved_data_indices[int(len(labeled_moved_data_indices) * 0.2):]
    test_indices += labeled_not_moved[int(len(labeled_not_moved) * 0.2):]

    shuffle(train_indices)
    shuffle(test_indices)
    train = input_data[train_indices]
    test = input_data[test_indices]

    training_labels = np.array([labels[k] for k in train_indices])
    test_labels = [labels[k] for k in test_indices]

    print("Vf")
    print(len(train_indices),len(test_indices))
    print(np.argwhere[training_labels==1])
    # print(min([1118271, 13916, 18088, 20961, 2044, 10461, 1687, 1540, 5650, 4651, 9367, 2938, 6477, 4060, 10889]))

'''
