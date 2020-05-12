## Transitions Recognition: GCN model

### Data: DBLP data set
The DBLP is an open source dataset available for download at: https://dblp.uni-trier.de/xml/
The DBLP Computer Science bibliography provides comprehensive data about research papers in the Computer Science field.
Our data covers 20 years (1990-2010).
We constructed a network graph so that each researcher in the dataset represents a node in the graph and every two authors are connected with an edge if they publish at least one paper together on the same year. 

The main idea is to create a temporal graph and monitor changes over time. We aimed for a generally stable graph presenting small changes over time.

We aspired to learn from this graph which was generated from a small amount of known information. 
In order to do so, we used 14 different labels (tags) for the data. The tags were determined by the names of the journals that published the articles. 

Two input files were created from the data using xml2txt.py.
As for the data is way to bug to upload, a small dummy is attached, named <edges_little.csv>,  <nodes_little.csv>.

### Backgroud:
Our main goal is to determine whether a writer is going to change his articles topics. For example, at 2000 he published 200 articles at the field of network, at 2001 he published 70 articles at the same field, but from 2002 and on, he started writing only at the field of operating systems.

In order to achieve this goal, we needed a model to label the data. Means, given 10% labeled data (writers we know their publishing history), and given a graph, the model will label the rest un-labeled publishers with their tags.
For this purpose, we used a GCN model, created using the pytorch geometric.

For this purpose, we used a GCN model, created using the pytorch geometric.

### GCN
We first separated the data by years, and created 21 graphs (for each year). Each writer now is having a label for each year. Our labels will be distribution vectors. For example, if a writer at 2000 wrote 20 articles at 1 topic, and 30 at 10 topic, it’s vector-label will be look as the following:

[ 0, 20/50, 0, 0, 0, 0, 0, 0, 0, 30/50, 0, 0, 0, 0, 0 ]

Each writer will have 21 vectors - for each year. 
If a writer didn’t write anything in a specific year, we labeled him with -1.
For each of the graphs, we created features matrix using the following features:
- page rank
- general
- Average_Neighbor_Degree
- k core

The code we used for these calculation lies here: https://github.com/louzounlab/graph-measures 

Input:
- A1, ….. An   -   Adjacency matrix for each year
- X1, ….. Xn   - Features matrix for each graph (for each year)
- L1 ….. Ln     - Labels for each year, described above.

### Running instructions:

set GRAPHS_NUM to number of years, according to the data
run with the following command:
python GCN_temporal_communities.py

for nni see the attached overview.











