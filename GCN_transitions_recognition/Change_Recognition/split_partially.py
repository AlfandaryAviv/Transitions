import pickle
from random import shuffle

import numpy as np


if __name__ == "__main__":

    partial_legth = 1000

    with open('input_7_tag_MSE.pkl', 'rb') as f:
        input_data = pickle.load(f)
    with open('input_7_tag_real_tags.pkl', 'rb') as f:
        real = pickle.load(f)
    with open('labels_per_writer_7_tag.pkl', 'rb') as f:
        labels = pickle.load(f)

    partial_data = np.zeros((partial_legth, 21))
    partial_real = np.zeros((partial_legth, 21))
    partial_labels = np.zeros(partial_legth)
    count = 0

    labeled_moved_data_indices = [b for b, item in enumerate(labels) if item == 1]
    labeled_not_moved_indices = [c for c, item in enumerate(labels) if item != 1]
    shuffle(labeled_not_moved_indices)

    print(len(labeled_moved_data_indices))
    print(len(labeled_not_moved_indices))

    for index in labeled_moved_data_indices:
        partial_data[count] = input_data[index]
        partial_labels[count] = 1
        partial_real[count] = real[index]
        count += 1

    for index in labeled_not_moved_indices:
        partial_data[count] = input_data[index]
        partial_real[count] = real[count]
        partial_labels[count] = 0
        count += 1
        if count==1000:
            break

    with open("input_partial_MSE.pkl", 'wb') as handle:
        pickle.dump(partial_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open("labels_partial_MSE.pkl", 'wb') as handle:
        pickle.dump(partial_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open("input_real_partial.pkl", 'wb') as handle:
        pickle.dump(partial_real, handle, protocol=pickle.HIGHEST_PROTOCOL)