## Change Recognition: Testing GCN's output

In order to test gcn's outputs on the graphs, here are some instructions and files needed for this code to run:

#### Backgroud:
Our goal is to determine whether a writer is going to change his articles topics. For this, we trained a gcn model,
and our goal now is to test the it's output, see if we recognise transitions with them, and (F.) define whether a writer is going to move in future.

The way we define transition:
The labels we build are to define whether a person has changed his writing topic, according to our definition change/transition.
In our case, transition is when the vector (the 21 years vector each person) includes k times 0 and then at least k times 1. 
For handling the -1 cases (so we don't miss cases), we used the following rule:
if a time point is surrounded by 1 it's 1, if it's surrounded with 0 it's 1, and if it's surrounded one side by 1 and in the other one by 0 - it's 0. 

steps for recognise transitions in past:
Building labels, building input from the gc'n output, building and running a model to find transitions using them.

Data needed: 
- A directory named <graphs_by_years> which includes GRAPHS_NUM (in our case: 21) files named as <labels_i> , i from 0 to GRAPHS_NUM
- A directory named <gcn_outputs> which includes GRAPHS_NUM matrices, each of the is the gcn's output for each year.

### Running instructions:

#### Build labels:

Run Build_Labels.py to create labels_per_writer_7_tag.pkl file. (This file will define for each writer if he has moved topic, 
according to our drfinition for movement, as explained above.

#### Build input:

Run Build_Input.py to create input_7_tag.pkl file. This will be the input to the model we build to check the accuracy of the gcn.

#### Running the model:
Only after the former two levels, run the transitions recognition model with the following command:
python Graph_Chnages_Recognition.py





