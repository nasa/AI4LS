import networkx as nx
import numpy as np

import sys
import os
import pandas as pd
# sys.path.append(os.getcwd())

from .new_get_ensemble_results import get_ensemble_results


def get_average_voting_ensemble(list_jsons):
    n_methods = len(list_jsons)
    n_seeds = len(list_jsons[0])
    
    print('n methods', n_methods)
    print('n seeds', n_seeds)
    
    
    # let's get the mean of the methods and seeds
    list_voted = []
    for i in range(n_seeds):
        # let's append the different methods
        methods_same_seed = []
        for j in range(n_methods):
            methods_same_seed.append(list_jsons[j].get(str(i)))
        
        voting = get_ensemble_results(methods_same_seed)
        if i == 0:
            list_features = voting.index.values.tolist()
        list_voted.append(voting['weighted_coefficient'].to_numpy())
        
    list_voted = np.array(list_voted)
    print(list_voted.shape)
    list_voted_average = np.mean(list_voted, axis=0)
    
    # let's get the mean per model
    list_methods = []
    list_methods_avg = []
    for i in range(n_methods):
        list_methods.append(list_jsons[i].get(str(0))["method"]+" avg")
        seeds_same_method = []
        for j in range(n_seeds):
            temp = list_jsons[i].get(str(j))
            temp["method"] = temp["method"] + str(j)
            seeds_same_method.append(temp)
        
        voting = get_ensemble_results(seeds_same_method)
        list_methods_avg.append(voting["weighted_coefficient"].to_numpy())
        
    list_methods_avg = np.array(list_methods_avg)
    print('### feature sahep', len(list_features), list_features[0])
    print('### listmethods', list_methods)
    print('### avg methods shape', list_methods_avg.shape)
    df = pd.DataFrame(data=list_methods_avg.T, columns=list_methods, index=list_features)
    
    df["Voting Average"] = list_voted_average
    
    return df
        