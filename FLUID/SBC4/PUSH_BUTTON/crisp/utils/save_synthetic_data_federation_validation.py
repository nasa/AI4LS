import sys
import os
sys.path.append(os.getcwd())

import numpy as np

from utils.get_synthetic_data_for_validation import get_datasets
from utils.vm_helpers import save_dict_to_json
from utils.gcp_helpers import save_json_to_bucket
from synthetic.facebook_synthetic_data_generator import generator_example
from synthetic.synthetic_generator_conf import synthetic_generator_counfounder
import pandas as pd

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n_seeds', default=100, help='number of seeds that should run', type=int)
parser.add_argument('--dim_inv', default=10, help='dimension of the invariant variables', type=int)
parser.add_argument('--dim_spu', default=50, help='dimension of the spurius variables', type=int)
parser.add_argument('--dim_unc', default=0, help='dimension of the uncorrelated variables', type=int)
parser.add_argument('--n_env', default=5, help='number of environments that we consider', type=int)
parser.add_argument('--data_dir', default='data_validation/', help='directory where the data are saved', type=str)
parser.add_argument('--n_ex_train', default=300, help='dimension of the training sample', type=int)
parser.add_argument('--n_ex_test', default=300, help='dimension of the testing samples', type=int)


args = parser.parse_args()



# TO DO: Include example4

if __name__ == '__main__':
    
    mount_point = os.path.join(os.getcwd(),args.data_dir)
    bucket_dir = "validation"
    # unmount and mount the directroy
    try:
        os.system("umount %s" %(mount_point))
    except:
        pass
    
    try:
        os.system("gcsfuse -only-dir %s ah21_data %s  " %(bucket_dir, mount_point))
    except:
        pass
    
    
    # Generate the datasets
    examples = ["Example1", "Example2"]#, "Example1", "Example2", "Example3", "Example4", "Example5", "Example_Confounder", "Example_Nonlinear"]
    n_seeds = args.n_seeds
    
    for example in examples:      
        
        if "Example0" in example:
            args.dim_inv = int(5)
            args.dim_spu = int(5)
            args.dim_unc = int(0)
            args.n_ex_train = int(3000)
            args.n_env = int(3)
            print(args.dim_inv)
            print(args.dim_spu)
            print(args.dim_inv + args.dim_spu)
            print(type(args.dim_spu))
            
        
        experiment_name = example +'_dim_inv_'+str(args.dim_inv)+\
                        '_dim_spu_'+str(args.dim_spu)+'_dim_unc_'+str(args.dim_unc) + '_n_exp_'+str(args.n_ex_train)+\
                        '_n_env_'+str(args.n_env) + '/'
        
        if os.path.exists(os.path.join(os.getcwd(),args.data_dir, experiment_name)):
            continue
        
        for seed in range(n_seeds):
            
            print(example, ' seed ', int(seed))
                    
            if example == "Example_Nonlinear":
                df = synthetic_generator_nonlinear_environments(
                        n=args.n_ex_train+args.n_ex_test,
                        n_layer=[args.dim_inv,args.dim_spu//2,args.dim_spu-args.dim_spu//2],
                        n_causal=args.dim_inv, 
                        n_env=args.dim_env)
                df_train = df[:args.n_ex_train]
                df_test = df[-args.n_ex_test:]
            elif example == "Example_Confounder":
                df = synthetic_generator_counfounder(
                        n=args.n_ex_train+args.n_ex_test,
                        n_layer=[args.dim_inv+1,args.dim_spu//2,args.dim_spu-args.dim_spu//2],
                        n_causal=args.dim_inv, 
                        n_env=args.n_env)
                df_train = df[:args.n_ex_train]
                df_test = df[-args.n_ex_test:]
            else:
                if example=="Example4":
                    df_train = generator_example(int(example[-1]), (args.dim_inv//2,args.dim_inv//2),
                                           (args.dim_spu//2,args.dim_spu//2),
                                           args.n_ex_train, args.n_env, save=False, test=False, seed=seed,
                                           dim_unc=args.dim_unc)
                    df_test = generator_example(int(example[-1]), (args.dim_inv//2,args.dim_inv//2),
                                           (args.dim_spu//2,args.dim_spu//2),
                                           args.n_ex_train, args.n_env, save=False, test=True, seed=seed,
                                           dim_unc=args.dim_unc)
                if "Example0" in example:
                    df_train = generator_example(example[-2:], args.dim_inv, args.dim_spu,
                                           args.n_ex_train, args.n_env, save=False, test=False, seed=seed,
                                           dim_unc=args.dim_unc)
                    df_test = generator_example(example[-2:],  args.dim_inv, args.dim_spu,
                                           args.n_ex_train, args.n_env, save=False, test=True, seed=seed,
                                           dim_unc=args.dim_unc)
                else:
                    df_train = generator_example(int(example[-1]), args.dim_inv, args.dim_spu,
                                           args.n_ex_train, args.n_env, save=False, test=False, seed=seed,
                                           dim_unc=args.dim_unc)
                    df_test = generator_example(int(example[-1]), args.dim_inv, args.dim_spu,
                                           args.n_ex_train, args.n_env, save=False, test=True, seed=seed,
                                           dim_unc=args.dim_unc)


            
                
                
            # saving the training data first
            
            for i in range(args.n_env):
                print(os.getcwd())
                path = os.path.join(os.getcwd(),args.data_dir, experiment_name, "seed_%s"%seed, "col_%s" %i, "train")
                print(path)
                if not os.path.exists(path): 
                    os.makedirs(path)
                    
                    
                df = df_train[df_train["env_split"] == i]
                df = df.drop(columns=["env_split", "Subj_ID"])
                df.to_csv(os.path.join(path, "data.csv"), index=False)
                
                
            # saving the testing data after
            
            for i in range(args.n_env):
                path = os.path.join(os.getcwd(),args.data_dir, experiment_name, "seed_%s"%seed, "col_%s" %i, "test")
                if not os.path.exists(path): 
                    os.mkdir(path)  
                df = df_test[df_test["env_split"] == i]
                df = df.drop(columns=["env_split", "Subj_ID"])
                df.to_csv(os.path.join(path, "data.csv"), index=False)
                
                
    try:
        os.system("umount %s" %(mount_point))
    except:
        pass
    