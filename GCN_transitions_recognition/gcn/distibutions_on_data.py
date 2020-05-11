import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)

import seaborn as sns
sns.set()




with open("matrix_labels_per_writer_from_gcn_output.pkl", 'rb') as f:
    matrix_per_person = pickle.load(f)

with open("matrix_labels_per_writer.pkl", 'rb') as f:
    matrix_per_person_real = pickle.load(f)


moved = [15325,18519,18998,23876,24362,24365,82107,136890,137282,204778,1074017]
not_moved = [a-1 for a in moved]

moved_vectors = [[matrix_per_person[person][i][7] for i in range(21)] for person in moved]
moved_real_vectors = [[matrix_per_person_real[person][i][7] for i in range(21)] for person in moved]
not_moved_vectors = [[matrix_per_person[person][i][7] for i in range(21)] for person in not_moved]
not_moved_real_vectors = [[matrix_per_person_real[person][i][7] for i in range(21)] for person in not_moved]

print(moved_vectors[1])
print(moved_real_vectors[1])
#print(not_moved_vectors)
#print(not_moved_real_vectors)




not_moved_vectors[1] = [not_moved_vectors[1][j]*100000 for j in range(21)]
t = np.array([a for a in range(21)])
plt.scatter(t, not_moved_vectors[1],c="r")
#plt.scatter(t, not_moved_real_vectors[1])
plt.suptitle("Not Moved vector gcn")
plt.savefig("exploring_data_figs/not_moved_vector_gcn.png")
plt.show()













#seaborn stuff
'''
ax=sns.heatmap(matrix_per_person[0])
plt.suptitle("not moved matrice example",fontsize=10)
plt.savefig("exploring_data_figs/moved_7_tag.png")
plt.show()

ax=sns.heatmap(matrix_per_person[15325])
print(matrix_per_person[15326])
plt.show()
plt.savefig("exploring_data_figs/not_moved_7_tag.png")


with open(os.path.join("gcn_outputs", "gcn_1" + '.pkl'), 'rb') as f:
    labels_from_model = pickle.load(f)

u_d= np.random.rand(10,12)
ax=sns.heatmap(labels_from_model)
plt.savefig("exploring_data_figs/1m_x_15_first_year.png")
plt.show()
'''

