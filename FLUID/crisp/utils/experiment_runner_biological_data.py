"""
This function is meant to be used to run different biological data on the
different methods so to be able to compare them on the same settings.
"""

import sys
import os
import pandas as pd

sys.path.append(os.path.join(os.getcwd(),"crisp"))

import torch
import numpy as np

import json
from dataio.datasets import get_datasets
from vm_helpers import save_dict_to_json

from dataio.DataFrameDataset import DataFrameDataset

from models.CausalNex import CausalNexClass, CausalNexClassEnv
from models.EmpericalRiskMinimization import EmpericalRiskMinimization
from models.NonLinearInvariantRiskMinimization import NonLinearInvariantRiskMinimization
from models.LinearInvariantRiskMinimization import LinearInvariantRiskMinimization
from models.SimpleNonLinearInvariantRiskMinimization import SimpleNonLinearInvariantRiskMinimization

import io
from google.cloud import storage
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--example', default='elijah_mouse_experiment_reduced', help='Example to run the validation on', type=str)
parser.add_argument('--method', default='ERM', help='Causal method that we want to implement', type=str)
#parser.add_argument('--dim_reduction', default=False, help='Implement dimensionality reduction', type=bool)
parser.add_argument('--vebose', default=True, help='vebose flag of method', type=bool)
parser.add_argument('--n_iterations', default=1000, help='number of iterations to train method', type=int)
parser.add_argument('--save_dir', default='results', help='directory where the experiments are saved', type=str)
parser.add_argument('--data_dir', default='data', help='directory where the data are saved', type=str)
args = parser.parse_args()

# Read the data
bucket_file_name = args.example

# This is to load the federated model weights from buckets
storage_client = storage.Client()
bucket = storage_client.get_bucket('ah21_data')

# This is to mount the directory where the json file 
#   results are stored for non-federated models
mount_dir = args.example    # bucket directory
mount_point = os.path.join(args.data_dir, args.example)  # local save directory

if not os.path.isdir(mount_point):
    os.makedirs(mount_point)
try:
    os.system("umount %s" %mount_point)
except:
    pass
#dir = os.path.join(mount_dir, experiment_name)
print("\nmount dir: ", mount_dir)

print("gcsfuse --implicit-dirs -only-dir %s ah21_data %s" %(mount_dir, mount_point))
try:
    os.system("gcsfuse --implicit-dirs -only-dir %s ah21_data %s" %(mount_dir, mount_point))
except:
    pass

# mount the savind firectory




list_in_exp_dir = os.listdir(os.path.join(args.data_dir,mount_dir))
n_seeds = 0
for elem in list_in_exp_dir:
    if "seed" in elem:
        n_seeds +=1
        
list_in_seed_dir = os.listdir(os.path.join(args.data_dir,mount_dir,'seed_0'))
n_env = 0
list_in_seed_dir_col = []
for elem in list_in_seed_dir:
    if "col_" in elem:
        n_env += 1
        list_in_seed_dir_col.append(elem)

# Generate the methods configurations
if args.method == 'IRM':
    method_config = {"use_icp_initialization": False,
            "verbose": args.vebose,
            "n_iterations": args.n_iterations,
            "seed": 0,
            "lr": 0.005, # 0.001
            "cuda": True}
elif args.method == 'NLIRM':
    method_config = {
            "NN_method": "NN",
            "verbose": args.vebose,
            "n_iterations": args.n_iterations,
            "seed":  0,
            "l2_regularizer_weight": 0.001,
            "lr": 0.001,
            "penalty_anneal_iters": 100,
            "penalty_weight": 10000.0,
            "cuda": False,
            "hidden_dim":  256
        }
elif args.method == 'ERM':
    method_config = {
            "verbose": args.vebose,
            "method": 'Linear',
            "cuda": False,
            "seed": 12,
            "epochs": 200,
            "hidden_dim": 256
        }
elif args.method == 'ERM_MLP':
    method_config = {
            "verbose": args.vebose,
            "method": 'NN',
            "cuda": False,
            "seed": 12,
            "epochs": 200,
            "hidden_dim": 256
        }
elif args.method == 'CSNX':
    method_config = {"output_data_regime":"real-valued"}
elif args.method == 'CSNX_ENV':
    method_config = {}
elif args.method == 'SNLIRM':
    method_config = {
            "verbose": args.vebose,
            "n_iterations": args.n_iterations,
            "seed":  0,
            "l2_regularizer_weight": 0.001,
            "lr": 0.001,
            "penalty_anneal_iters": 100,
            "penalty_weight": 10000.0,
            "cuda": False,
        }

# Run the experiments on the seeds 

results = {}

for seed in range(n_seeds):

    print('#### seed', seed)
    experiment_name = '%s/seed_%i'%(bucket_file_name,seed)
    
    environment_datasets = []
    val_dataset = []
    test_dataset = []
    
    train_list = []
    test_list = []
    
    for i, env in enumerate(list_in_seed_dir_col):
        print('## elem ',i)
        test_data_dir = "%s/%s/test/data.csv"%(experiment_name, env)
        train_data_dir = "%s/%s/train/data.csv"%(experiment_name, env)
        df_test = pd.read_csv(os.path.join(args.data_dir,test_data_dir), index_col=0)
        df_train = pd.read_csv(os.path.join(args.data_dir,train_data_dir), index_col=0)
        env_df = DataFrameDataset(dataframe=df_test,
                                 predictor_columns = df_test.keys().drop('Target'),
                                 target_columns=['Target'], exclude_columns=[''])
        train_list.append(env_df)
        test_list.append(df_test)
        
    environment_datasets = train_list
    test_dataset = DataFrameDataset(dataframe=pd.concat(test_list),
                                    predictor_columns = df_test.keys().drop('Target'),
                                    target_columns=['Target'], exclude_columns=[''])
    
    method_config["seed"]=seed
    
    method_config["target"] = ['Target']
    if "elijah" in args.example:
        method_config["output_data_regime"] = "real-valued"
    else:
        method_config["output_data_regime"] = "binary"
        
    method_config["columns"] = df_test.keys().drop('Target')
    method_config["output_dim"] = 1

    if args.method == 'IRM':
        model = LinearInvariantRiskMinimization(environment_datasets, val_dataset, test_dataset, method_config)
    elif args.method == 'NLIRM':
        model = NonLinearInvariantRiskMinimization(environment_datasets, val_dataset, test_dataset, method_config)
    elif 'ERM' in args.method:
        model = EmpericalRiskMinimization(environment_datasets, val_dataset, test_dataset, method_config)
    elif args.method == 'CSNX':
        print('Starting CSNX')
        model = CausalNexClass(environment_datasets, val_dataset, test_dataset, method_config)
    elif args.method == 'CSNX_ENV':
        model = CausalNexClassEnv(environment_datasets, val_dataset, test_dataset, method_config)
    elif args.method == 'SNLIRM':
        model = SimpleNonLinearInvariantRiskMinimization(environment_datasets, val_dataset, test_dataset, method_config)

    results[seed] = model.results()
    results[seed]["seed"] = seed

    try:
        results.get(seed).get("to_bucket")["test_acc"] = results.get(seed).get("to_bucket")["test_acc"].item()
    except:
        pass

    try:
        results.get(seed)["test_acc"] = results.get(seed)["test_acc"].item()
    except:
        pass

    try:
        results.get(seed)["test_probs"] = np.float64(results.get(seed)["test_probs"]).tolist()
    except:
        pass

    try:
        results.get(seed)["test_labels"] = np.float64(results.get(seed)["test_labels"]).tolist()
    except:
        pass

    try:
        results.get(seed)["feature_coeffients"] = np.float64(results.get(seed)["feature_coeffients"]).tolist()
    except:
        pass
    try:
        results.get(seed)["to_bucket"]["coefficients"] = np.float64(results.get(seed).get("to_bucket")["coefficients"].squeeze()).tolist()
    except:
        pass

    path = os.path.join(os.getcwd(),args.data_dir, args.example)
    #path = os.path.join(os.getcwd(),args.save_dir)
    if not os.path.exists(path):
        os.makedirs(path)
    print(path)

    save_dict_to_json(results, path+'/'+args.method+'.json')

