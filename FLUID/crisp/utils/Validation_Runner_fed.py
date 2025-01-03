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

from fl_src.irm_module import IRMModule, NLIRMModule
from fl_src.erm_module import ERMModule
from utils.validation_runner_utils import get_seed_blobs, get_acc_loss

import io
from google.cloud import storage
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_name', default='Example0a', help='Example index of experiment to load', type=str)
parser.add_argument('--output_data_regime', default="real-valued", help="output type of the dataset being evaluated", type=str)
parser.add_argument('--n_seeds_end', default=20, help='number of seeds that can be looped over', type=int)
parser.add_argument('--n_seeds_start', default=0, help='start of seeds that can be looped over', type=int)
parser.add_argument("--index_col", default=False, help="Does the datasets csv files contain an index column that needs removing", type=bool)
# parser.add_argument('--dim_inv', default=6, help='dimension of the invariant variables', type=int)
# parser.add_argument('--dim_spu', default=6, help='dimension of the spurius variables', type=int)
# parser.add_argument('--dim_unc', default=500, help='dimension of the uncorrelated variables', type=int)
# parser.add_argument('--n_samp', default=900, help='number of samples that we are going to consider', type=int)
# parser.add_argument('--n_env', default=5, help='number of environments that we consider', type=int)
parser.add_argument('--bucket', default=True, help='using buckets or not', type=bool)
parser.add_argument('--data_dir', default='data/synthetic', help='directory where the data are saved', type=str)
parser.add_argument('--save_dir', default='results/validation_bucket', help='directory where the experiments are saved', type=str)
parser.add_argument('--user', default='user', help='/home subdirectory used for local mountpoint', type=str)
parser.add_argument('--best', default='best', help='best or last model .pt that is saved during training', type=str)
# parser.add_argument('--model_weights', default=None, help='.pt file with model weights trained in federation', type=str)
#parser.add_argument('--vebose', default=False, help='vebose flag of method', type=bool)
#parser.add_argument('--n_iterations', default=1000, help='number of iterations to train method', type=int)


args = parser.parse_args()

# Define the configuration of the data
# config_file="experiment_configs/synthetic_experiments_config.json"
# config = json.load(open(config_file, "r"))

# if args.example == "Example4":
#     config[args.example]["n_inv"] = (int(args.dim_inv//2),int(args.dim_inv//2))
#     config[args.example]["n_spu"] = (int(args.dim_spu//2),int(args.dim_spu//2))
# else:
#     config[args.example]["n_inv"] = args.dim_inv
#     config[args.example]["n_spu"] = args.dim_spu
# config[args.example]["n_unc"] = args.dim_unc
# config[args.example]["n_ex_train"] = args.n_samp
# config[args.example]["n_ex_test"] = args.n_samp
# config[args.example]["n_env"] = args.n_env

# modified_config_file = "experiment_configs/synthetic_experiments_config_modified.json"
# with open(modified_config_file, 'w') as f:
#     json.dump(config, f)

client = storage.Client()
blobs = client.list_blobs('ah21_data', prefix='validation/Example')

# get the list of Experiments in the GCP Bucket
#experiments = set([])
#for blob in blobs:
#    exp = blob.name.split("/")[1]
#    experiments.add(exp)
#experiments = list(experiments)
#experiments.sort()
#print(">>", experiments)

#for i, exp in enumerate(experiments):
#    if args.example in exp:
#        experiment_name = exp
#        dir_ind = i
#print(">", dir_ind, experiment_name)

# experiment_name = "%s_dim_inv_%s_dim_spu_%s_dim_unc_%s_n_exp_%s_n_env_%s" % (args.example, args.dim_inv, args.dim_spu, args.dim_unc, args.n_samp, args.n_env)
experiment_name = args.experiment_name

## Sort out the bucket and storage: 
if args.bucket:
    # This is to load the federated model weights from buckets
    storage_client = storage.Client()
    bucket = storage_client.get_bucket('ah21_data')

    # This is to mount the directory where the json file 
    #   results are stored for non-federated models

    mount_bucket = os.path.join(experiment_name)
    print("\nmount dir: ", mount_bucket)

    mount_point = "/home/" + args.user + "/data"  # local save directory
    if not os.path.isdir(mount_point):
        os.mkdir(mount_point)

    try:
        os.system("umount %s" %mount_point)
    except:
        pass

    print("gcsfuse --implicit-dirs -only-dir %s ah21_data %s" %(mount_bucket, mount_point))
    try:
        os.system("gcsfuse --implicit-dirs -only-dir %s ah21_data %s" %(mount_bucket, mount_point))
    except:
        pass


# Generate the methods configurations
#method_config = {"output_regime" : config[args.example]["output_data_regime"]}
method_config = {"output_regime" : args.output_data_regime}

seed_subset = (args.n_seeds_start, args.n_seeds_end)
seed_blobs = get_seed_blobs(seed_subset, experiment_name)

dataset_path = os.path.join(mount_point, "seed_%s" %seed_subset[0], "col_0", "test", "data.csv")
column_names = pd.read_csv(dataset_path, index_col=None, nrows=0).columns.tolist()
column_names.remove("Target")
if args.index_col:
    column_names = column_names[1:]
print("NAMES = ", column_names)

# Load the experiments of the seed
results = {"irm0":{}, "irm1":{}, "irm2":{},"nlirm0":{},"nlirm1":{},"nlirm2":{},"erm0":{}}
n_skipped = 0
for seed in range(*seed_subset):
    if n_skipped > 5*len(list(results.keys())): continue

    if args.output_data_regime == "real-valued":
        num_classes = 1
        metric = "MSE"
    elif args.output_data_regime == "binary":
        num_classes = 2
        metric = "CCR"

    for method in ['IRM', 'ERM', 'NLIRM']:
        suffixes = [0,1,2]
        if method == 'ERM':
            suffixes = [0]

        for n in suffixes:
            model_name = method.lower()+str(n)
            print("\nmodel: ", model_name)
            dir = os.path.join(mount_bucket, 'seed_%s/save/crisp_%s_%s.pt' % ( seed, args.best, model_name))
            print(dir)
            try:
                en_model = bucket.get_blob(dir).download_as_string()

                weights = torch.load(io.BytesIO(en_model), map_location=torch.device('cpu'))["model_state_dict"]

                for key, value in weights.items():
                    prefix = key.split("phi")[0]
                    print(">>", prefix, value.shape)
                    in_size = value.shape[1]
                    break

                #print("Model weight info:", prefix, in_size, out_size)

                if method == 'IRM':
                    model = IRMModule(logger=None, prefix=prefix)
                if method == 'NLIRM':
                    model = NLIRMModule(logger=None, prefix=prefix)
                if method == 'ERM':
                    model = ERMModule(logger=None, prefix=prefix)
                model.init_network(in_size, num_classes, output_data_regime=args.output_data_regime, seed=seed)
                model.load_state_dict(weights)
                # model.set_test_dataset(test_dataset)
                model.predictor_columns = column_names

                results[model_name][seed] = model.results()
                results[model_name][seed]["seed"] = seed

                # read accuracy and loss info from logs
                seed_blob = [x for x in seed_blobs if "seed_" + str(seed) + "/" in x]
                # print(">>",seed_blob)
                if len(seed_blob) == 1:
                    seed_blob = seed_blob[0]
                else:
                    raise Exception("multiple blobs for seed:", seed, seed_blob)

                acc_loss_results = get_acc_loss(seed_blob)

                suffix = prefix[:-1] # remove trailing underscore
                loss_key = "train_IRM_" + method.lower() + "_loss_" + model_name
                acc_key = 'aggregated_model_validation_' + metric + '_' + model_name
                #print(">", results[model_name][seed])
                print(">>", acc_loss_results.keys())
                print(loss_key, acc_key)
                results[model_name][seed]["to_bucket"]["test_acc"] = acc_loss_results.get(acc_key)[-1] # grab accuracy from last round
                results[model_name][seed]["to_bucket"]["test_loss"] = acc_loss_results.get(loss_key) # use the losses from all rounds

                results[model_name][seed]["to_bucket"]["coefficients"] =  results[model_name][seed]["to_bucket"]["coefficients"].squeeze().detach().numpy().tolist()

            except Exception as e:
                print("No directory found on bucket for: %s" % dir)
                print("Exception:", e)
                n_skipped += 1


for model_name, value in results.items():
    path = os.path.join(os.getcwd(),args.save_dir, experiment_name)
    #path = os.path.join(os.getcwd(),args.save_dir)
    if not os.path.exists(path):
        os.makedirs(path)
    #print(path)
    print("saved to bucket loc:", os.path.join(mount_point, model_name+"_fed.json"))

    save_dict_to_json(value, path+'/'+model_name+'_fed.json')
    save_dict_to_json(value, os.path.join(mount_point, model_name+'_fed.json'))
