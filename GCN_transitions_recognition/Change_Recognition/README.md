### Change Recognition: Testing GCN's output

In order to test gcn's outputs on the graphs, here are some instructions and files needed for this code to run:

Data needed: 
- A directory named <graphs_by_years> which includes GRAPHS_NUM (in our case: 21) files named as <labels_i> , i from 0 to GRAPHS_NUM
- A directory named <gcn_outputs> which includes GRAPHS_NUM matrices, each of the is the gcn's output for each year.

#### Running instructions:

Build labels:
Run Build_Labels.py to create labels_per_writer_7_tag.pkl file. (This file will define for each writer if he has moved topic, 
according to our drfinition for movement, as explained above.

Build input:
Run Build_Input.py to create input_7_tag.pkl file. This will be the input to the model we build to check the accuracy of the gcn.





