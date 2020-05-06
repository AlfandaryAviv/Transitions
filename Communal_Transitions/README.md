# Communal Transactions


1.	Install requirements.txt file

2.	Create input files:

    There are two input files: node.csv and edges.csv
    
    They should be located at “./dataset/data_name/input_files/” 
    
    (If they will be located at any other folder they will be transferred to the correct folder)
    
    A.	nodes.csv
        
        format: (node, year, label, count, percent)
          
          •	 node: node id - int
          
          •	year: int
          
          •	label: int number, tag 0 represents an unknown label
          
          •	count: number events same year with the same label
          
          •	percent: partial part of the label on the same year


        
        Input example:
        
        node,year,label,count,percent
        
        1,2010,0,4,100%
        
        2,1998,4,2,100%

    
    B. edges.csv 
      
        format: (n1, n2, year, count)
      
        •	n1: node_id1 - int
      
        •	n2: node_id2 - int
      
        •	year: int
      
        •	Count: counter of collaborations same year -int


      
        Input example:
      
        n1,n2,year,count
      
        1,2,1993,3
      
        1,3,1998,9


3.  download graphs-measures repository from the labs git: 
    
    https://github.com/louzounlab/graph-measures

4.	Change the parameters:

#### dataset parameters

    Dataset_Name is any data with the input format and data types

    current_nodes_path is the location of the nodes input: "./dataset/" + str(Dataset_Name) + "/nodes.csv"

    current_edges_path is the location of the edges input: "./dataset/" + str(Dataset_Name) + "/edges.csv"

    Neighbors_Factor is the weight parameter for neighbors edges (represents a relationship at the same timestamp)

    Timed_Factor is the weight parameter for timed edges (a connection between a vertex to himself on the different time lines)

    Similarity_Factor is the weights parameter for the similarity edges (generated from the GCN)

    Communal_Resolution effects the size of the community 

    Dataset_time_range in array format: [min_year, max year]

    All_Dataset_Is_Labeled: 'yes'/'no'

    Number_Of_Classes 



#### GCN Parameters

    Net is either GCNNet3 (3 conv layers) or GCNNet2 (2 conv layers)

    Epochs

    Dropout_Rate

    Hidden_Sizes in the format of: [10, 10] (if GCNNet2 was chosen the network will take the first element as the size of the hidden layer) 

    Learning_Rate

    Weight_Decay 

    Time_Inds = int(Dataset_time_range[1])+1 - int(Dataset_time_range[0])
