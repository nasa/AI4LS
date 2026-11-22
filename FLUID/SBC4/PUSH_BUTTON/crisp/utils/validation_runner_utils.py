import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from google.cloud import storage
storage_client = storage.Client()
bucket = storage_client.get_bucket('ah21_data')

def get_seed_blobs(seed_subset, experiment_path, verbose=False):
        
    potential_blobs = [str(blob).split()[2][:-1] for blob in storage_client.list_blobs(bucket, prefix=experiment_path)]
    seed_blobs = [blob for blob in potential_blobs if blob[-13:] == 'save/logs.txt'] 

    if verbose: print(">", len(seed_blobs), seed_blobs)
    sorted_seed_blobs = []
    for i in range(seed_subset[0], seed_subset[1]):
        for s in seed_blobs:
            if int(s.split("/")[-3].replace("seed_","")) == i:
                sorted_seed_blobs.append(s)
    if verbose: print("number of sorted blobs = ", len(sorted_seed_blobs))
    
    return sorted_seed_blobs

def get_acc_loss(seed, n_models=7):
    seed_dictionaries = {}
    blob = bucket.get_blob(seed)
    s = int(seed.split("/")[-3].replace("seed_",""))
    content = blob.download_as_string().decode('UTF-8').splitlines()
    tasks = [content[i-1] for i, c in enumerate(content) if '<openfl.component.aggregation_functions.weighted_average.WeightedAverage object at' in c][1::n_models]
    content = [c.split(">")[1][1:] for c in content if '<openfl.component.aggregation_functions.weighted_average.WeightedAverage object at' in c][1:]

    ## create a list of n_models*n_tasks tasks in the correct order as they appear in the tasks heading list (differs per seed)
    cnt = 0
    openfl_tasks = []
    for j in range(len(tasks)):
        t = [tasks[cnt] for model in range(n_models)]
        openfl_tasks.extend(t)
        cnt += 1    

    value_dictionary = {}
    for i, word in enumerate(content):
        task = openfl_tasks[i]
        key = task.replace(" task metrics...","")+"_"+word.split(":")[0]
        value = float(word.split(":")[1].replace("\t",""))

        if key in value_dictionary.keys():
            value_dictionary[key].append(value)
        else:
            value_dictionary[key] = [value]

    seed_dictionaries[s] = value_dictionary
    
    return seed_dictionaries[s]
