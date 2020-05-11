import os
import pickle
import numpy as np

GRAPHS_NUM = 21

def load_data():
    all_labels_from_model = []
    for i in range(GRAPHS_NUM):
        with open(os.path.join("gcn_outputs", "gcn_" + str(i) + '.pkl'), 'rb') as f:
            labels_from_model = pickle.load(f)
            all_labels_from_model.append(labels_from_model)
    return all_labels_from_model


def build_matrices(labels):  # labels = gcn's putput for each year - 21 length, each one is range of num of people
    labels_writers = np.zeros((len(labels[0]), len(labels), 15))  # Num of writers, 21, 15
    # for each writer
    for i in range(labels_writers.shape[0]):  # num of writers
        # for each year - labels_year is the labels from the gcn of the i'th year
        for year_index, labels_year in enumerate(labels):
            # in the writer matrix, in the line of the year, put an array of the corresponding labels of the gcn's output
            labels_writers[i][year_index] = np.array(labels_year[i])

    with open("matrix_labels_per_writer_from_gcn_output.pkl", 'wb') as handle:
        pickle.dump(labels_writers, handle, protocol=pickle.HIGHEST_PROTOCOL)


def build_input_7_tag():
    with open('matrix_labels_per_writer_from_gcn_output.pkl', 'rb') as f:
        # matrix for each person in the form of 21x15 labels
        all_writers = pickle.load(f)

    # now we are interested for each person only in it's 7'th column - the 7'th tag
    persons_7_vector = np.zeros((len(all_writers), GRAPHS_NUM))
    for idx, person in enumerate(all_writers):
        persons_7_vector[idx] = person.T[7]  # if we want to save this to file
    with open("input_7_tag.pkl", 'wb') as handle:
        pickle.dump(persons_7_vector, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    labels = load_data()
    build_matrices(labels)
    build_input_7_tag()

    ''' 
    some checks
    with open('input_7_tag_MSE.pkl', 'rb') as f:
        all_writers_7_tag = pickle.load(f)
    with open('labels_per_writer_7_tag.pkl', 'rb') as f:
        labels = pickle.load(f)

    with open('matrix_labels_per_writer.pkl', 'rb') as f:
        g = pickle.load(f)

    #check if it's logic values
    # for idx,label in enumerate(labels):
    #     if label!=0:
    #         print(idx)
    c=0
    t=0
    pl=7
    person = all_writers_7_tag[0][pl]
    for per in all_writers_7_tag:
        if per[pl]!=person:
            c+=1
            if t<500:
                print(per[pl])
                t+=1
    print(c)'''
