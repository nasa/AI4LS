"""
This function is meant to be used to run different experiments on the
different methods so to be able to compare them on the same settings.
"""
import sys
import os
import pandas as pd
sys.path.append(os.getcwd())

import torch
import numpy as np

import json
from get_synthetic_data_for_validation import get_datasets
from vm_helpers import save_dict_to_json

from models.CausalNex import CausalNexClass, CausalNexClassEnv
from models.EmpericalRiskMinimization import EmpericalRiskMinimization
from models.NonLinearInvariantRiskMinimization import NonLinearInvariantRiskMinimization
from models.LinearInvariantRiskMinimization import LinearInvariantRiskMinimization
from models.SimpleNonLinearInvariantRiskMinimization import SimpleNonLinearInvariantRiskMinimization

import io
from google.cloud import storage
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--example', default='Example1', help='Example to run the validation on', type=str)
parser.add_argument('--method', default='CSNX', help='Causal method that we want to implement', type=str)
parser.add_argument('--n_seeds', default=100, help='number of seeds that should run', type=int)
parser.add_argument('--dim_inv', default=6, help='dimension of the invariant variables', type=int)
parser.add_argument('--dim_spu', default=6, help='dimension of the spurius variables', type=int)
parser.add_argument('--dim_unc', default=500, help='dimension of the uncorrelated variables', type=int)
parser.add_argument('--n_samp', default=900, help='number of samples that we are going to consider', type=int)
parser.add_argument('--n_env', default=5, help='number of environments that we consider', type=int)
parser.add_argument('--bucket', default=False, help='using buckets or not', type=bool)
parser.add_argument('--data_dir', default='data/synthetic', help='directory where the data are saved', type=str)
parser.add_argument('--save_dir', default='results/validation_bucket', help='directory where the experiments are saved', type=str)
parser.add_argument('--model_weights', default=None, help='.pt file with model weights trained in federation', type=str)

parser.add_argument('--vebose', default=False, help='vebose flag of method', type=bool)
parser.add_argument('--n_iterations', default=6000, help='number of iterations to train method', type=int)

args = parser.parse_args()

# Define the configuration of the data
config_file="experiment_configs/synthetic_experiments_config.json"
config = json.load(open(config_file, "r"))

if args.example == "Example4":
    config[args.example]["n_inv"] = (int(args.dim_inv//2),int(args.dim_inv//2))
    config[args.example]["n_spu"] = (int(args.dim_spu//2),int(args.dim_spu//2))
else:
    config[args.example]["n_inv"] = args.dim_inv
    config[args.example]["n_spu"] = args.dim_spu
config[args.example]["n_unc"] = args.dim_unc
config[args.example]["n_ex_train"] = args.n_samp
config[args.example]["n_ex_test"] = args.n_samp
config[args.example]["n_env"] = args.n_env


# Save the config, so that it can be used

modified_config_file = "experiment_configs/synthetic_experiments_config_modified.json"
with open(modified_config_file, 'w') as f:
    json.dump(config, f)

bucket_file_name = args.example +'_dim_inv_'+str(args.dim_inv)+\
                '_dim_spu_'+str(args.dim_spu)+'_dim_unc_'+ str(args.dim_unc) + \
                '_n_exp_'+str(args.n_samp)+\
                '_n_env_'+str(args.n_env)
experiment_name = bucket_file_name + '/'#'_n_seeds_' + str(args.n_seeds) +'/'


# Generate the methods configurations
if args.method == 'IRM':
    method_config = {"use_icp_initialization": False,
            "verbose": args.vebose,
            "n_iterations": args.n_iterations,
            "seed": 0,
            "lr": 0.001, # 0.001
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
            "epochs": args.n_iterations,
            "hidden_dim": 256
        }
elif args.method == 'CSNX':
    method_config = {
            "output_data_regime": config[args.example]["output_data_regime"]
    }
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

for seed in range(0, args.n_seeds):


    method_config["seed"]=seed
    print('####### seed', str(seed))
    environment_datasets, val_dataset, test_dataset, config =  get_datasets(args.example,
                                                        config_file=modified_config_file, seed=seed,
                                                        save_dir = args.data_dir)
    method_config["target"] = config["data_options"]["targets"]
    method_config["output_data_regime"] = config["data_options"]["output_data_regime"]
    method_config["columns"] = config["data_options"]["predictors"]
    method_config["output_dim"] = config["output_dim"]

    if args.method == 'IRM':
        model = LinearInvariantRiskMinimization(environment_datasets, val_dataset, test_dataset, method_config)
    elif args.method == 'NLIRM':
        model = NonLinearInvariantRiskMinimization(environment_datasets, val_dataset, test_dataset, method_config)
    elif args.method == 'ERM':
        model = EmpericalRiskMinimization(environment_datasets, val_dataset, test_dataset, method_config)
    elif args.method == 'CSNX':
        model = CausalNexClass(environment_datasets, val_dataset, test_dataset, method_config)
    elif args.method == 'CSNX_ENV':
        model = CausalNexClassEnv(environment_datasets, val_dataset, test_dataset, method_config)
    elif args.method == 'SNLIRM':
        model = SimpleNonLinearInvariantRiskMinimization(environment_datasets, val_dataset, test_dataset, method_config)

    results[seed] = model.results()
    results[seed]["seed"] = seed
#     print()
#     print(sorted(zip(results[seed]["to_bucket"]["features"], results[seed]["to_bucket"]["sensitivities"]), key=lambda x: x[1], reverse=True))

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

    path = os.path.join(os.getcwd(),args.save_dir, experiment_name)
    #path = os.path.join(os.getcwd(),args.save_dir)
    if not os.path.exists(path):
        os.makedirs(path)
    print(path)

    save_dict_to_json(results, path+'/'+args.method+'.json')

