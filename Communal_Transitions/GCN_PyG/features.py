import os
from loggers import PrintLogger
import numpy as np
from features_infra.graph_features import GraphFeatures
import pickle
from features_infra.feature_calculators import FeatureMeta
from features_algorithms.vertices.average_neighbor_degree import AverageNeighborDegreeCalculator
from features_algorithms.vertices.general import GeneralCalculator
from features_algorithms.vertices.k_core import KCoreCalculator
from features_algorithms.vertices.page_rank import PageRankCalculator
# import networkx as nx
# from features_algorithms.vertices.attractor_basin import AttractorBasinCalculator
# from features_algorithms.vertices.louvain import LouvainCalculator
# from features_algorithms.vertices.betweenness_centrality import BetweennessCentralityCalculator

def create_features(data_name, time_range):
    for i in range(time_range):
        gnx = pickle.load(open("./dataset/"+data_name+"/pkl/gcn_input/"+"graph_"+str(i)+".pkl","rb"))

        # with open(os.path.join('data',str(data_name),'gcn_input', 'graph_'+str(i)+'.pkl'), 'rb') as f:
        #     gnx = pickle.load(f)
        logger = PrintLogger("MyLogger")
        features_meta = {
            "page_rank": FeatureMeta(PageRankCalculator, {"pr"}),
            "general": FeatureMeta(GeneralCalculator, {"gen"}),
            "Average_Neighbor_Degree": FeatureMeta(AverageNeighborDegreeCalculator, {"avg_nd"}),
            "k_core": FeatureMeta(KCoreCalculator, {"kc"})}
    
        features = GraphFeatures(gnx, features_meta, "./dataset/"+str(data_name)+"/pkl/feature", logger=logger)
        features.build()
        mx = features.to_matrix(mtype=np.matrix)

        pickle.dump(mx, open("./dataset/"+data_name+"/pkl/gcn_input/"+"mx_"+str(i)+".pkl", "wb"))
        # with open(os.path.join('data',str(data_name),'gcn_input','mx_'+str(i)+'.pkl'), 'wb') as f:
        #     pickle.dump(mx, f, protocol=pickle.HIGHEST_PROTOCOL)
            
    return

# with open(os.path.join('data',str(data_name),'pkl', 'mx_1.pkl'), 'rb') as f:
#     l = pickle.load(f)
#
# print (l[0])