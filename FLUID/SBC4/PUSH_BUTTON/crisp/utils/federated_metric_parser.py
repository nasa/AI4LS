import pandas as pd
import numpy as np
from google.cloud import storage
from tqdm import tqdm

def get_model_performance_accross_seeds(experiment_path, return_array=False, seed_subset = None, bucket = 'ah21_data'):
    
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket)
    potential_blobs = [str(blob).split()[2][:-1] for blob in storage_client.list_blobs(bucket,prefix=experiment_path)]
    seed_blobs = [blob for blob in potential_blobs if blob[-8:] == 'logs.txt'] #potentially fragile
    
    if seed_subset:
        seed_blobs = seed_blobs[seed_subset[0]:seed_subset[1]]
        
    seed_dictionaries = {}

    for seed in tqdm(range(len(seed_blobs))):
        blob = bucket.get_blob(seed_blobs[seed])
        content = blob.download_as_string().decode('UTF-8').splitlines()
        content = [c for c in content if 'Saved the best model' not in c]

        value_dictionary = {}
        
        for i, x in enumerate(content):
            if '<openfl.component' == x[:17] and '<openfl.component' == content[i+1][:17]:
                alpha_key = content[i-1].split()[0] +'_'+ content[i].split()[4]
                beta_key = content[i-1].split()[0] +'_'+ content[i+1].split()[4]

                if alpha_key not in value_dictionary.keys():
                    value_dictionary[alpha_key] = [float(content[i].split()[5])]
                else:
                    value_dictionary[alpha_key] = value_dictionary[alpha_key] + [float(content[i].split()[5])]

                if beta_key not in value_dictionary.keys():
                    value_dictionary[beta_key] = [float(content[i+1].split()[5])]
                else:
                    value_dictionary[beta_key] = value_dictionary[beta_key] + [float(content[i+1].split()[5])]
        
        seed_dictionaries[seed] = value_dictionary
    
    keys = list(seed_dictionaries[0].keys())
    
    mean_dictionary = {}
    array_dictionary = {}

    for key in keys:
        mean_dictionary[key] = np.mean(np.array([seed_dictionaries[x][key] for x in range(len(seed_blobs))]), axis = 0)
        array_dictionary[key] = np.array([seed_dictionaries[x][key] for x in range(len(seed_blobs))])

    if return_array:
        return (mean_dictionary, array_dictionary)
    else:
        return (mean_dictionary)