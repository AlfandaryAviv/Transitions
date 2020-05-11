#this file should be in the same level of geaph-measures project

import os
import pickle
import networkx as nx

from features_algorithms.vertices.attractor_basin import AttractorBasinCalculator
from loggers import PrintLogger
import numpy as np
from features_infra.graph_features import GraphFeatures
import pickle
from features_infra.feature_calculators import FeatureMeta
from features_algorithms.vertices.louvain import LouvainCalculator
from features_algorithms.vertices.betweenness_centrality import BetweennessCentralityCalculator
from features_algorithms.vertices.average_neighbor_degree import AverageNeighborDegreeCalculator
from features_algorithms.vertices.general import GeneralCalculator
from features_algorithms.vertices.k_core import KCoreCalculator
from features_algorithms.vertices.page_rank import PageRankCalculator

'''creating the features'''
def create_features():
    for i in range(21):
        with open(os.path.join('graphs_by_years', 'graph_'+str(i)+'.pkl'), 'rb') as f:
            gnx = pickle.load(f)
    
        logger = PrintLogger("MyLogger")
    
    
    
        features_meta = {
            "page_rank": FeatureMeta(PageRankCalculator, {"pr"}),
            "general": FeatureMeta(GeneralCalculator, {"gen"}),
            "Average_Neighbor_Degree": FeatureMeta(AverageNeighborDegreeCalculator, {"avg_nd"}),
            "k_core": FeatureMeta(KCoreCalculator, {"kc"}),
        }
    
        features = GraphFeatures(gnx, features_meta, "/home/dsi/racheli/graph_calculations", logger=logger)
        features.build()
    
        mx = features.to_matrix(mtype=np.matrix)
    
    
        with open(os.path.join('graphs_by_years','mx_'+str(i)+'.pkl'), 'wb') as f:
            pickle.dump(mx, f, protocol=pickle.HIGHEST_PROTOCOL)
            
