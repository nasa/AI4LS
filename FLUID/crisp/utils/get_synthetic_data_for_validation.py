import json
import numpy as np
from dataio.datasets import get_datasets_for_experiment
from synthetic.facebook_synthetic_data_generator import generator_example
from synthetic.synthetic_generator_conf import synthetic_generator_counfounder, synthetic_generator_nonlinear_environments

def get_datasets(example="Example1",
                 config_file="experiment_configs/synthetic_experiments_config.json",
                 save_dir = "data/synthetic", seed=0):
    
    config = json.load(open(config_file, "r"))
    config = config.get(example)
    
    exp_config = json.load(open("experiment_configs/template.json", "r"))

    if example == "Example_Nonlinear":
        df = synthetic_generator_nonlinear_environments(
                n=config["n_ex_train"]+config["n_ex_test"],
                n_layer=[config["n_inv"],config["n_spu"]//2,config["n_spu"]-config["n_spu"]//2],
                n_causal=config["n_inv"], 
                n_env=config["n_env"],
                n_unc=config["n_unc"])
        synthetic_name = 'nonlinear_'+'_dim_inv_'+str(config["n_inv"])+\
                '_dim_spu_'+str(config["n_spu"])+'_n_exp_'+str(config["n_ex_train"]+config["n_ex_test"])+\
                '_n_env_'+str(config["n_env"])+'.pickle'
        data_path = save_dir+synthetic_name
        df.to_pickle(data_path)
        columns_train = df.columns
        exp_config["data_options"]["dataset_fp"] = data_path
        exp_config["data_options"]["dataset_fp_train"] = None
        exp_config["data_options"]["dataset_fp_test"] = None
        exp_config["data_options"]['synthetic_train_test_split'] = False 
        
    elif example == "Example_Confounder":
        df = synthetic_generator_counfounder(
                n=int(config["n_ex_train"]+config["n_ex_test"]),
                n_layer=[config["n_inv"]+1,config["n_spu"]//2,config["n_spu"]-config["n_spu"]//2],
                n_causal=config["n_inv"], 
                n_env=config["n_env"],
                n_unc=config["n_unc"])
        synthetic_name = 'counfunder_'+'_dim_inv_'+str(config["n_inv"])+\
                '_dim_spu_'+str(config["n_spu"])+'_n_exp_'+str(config["n_ex_train"]+config["n_ex_test"])+\
                '_n_env_'+str(config["n_env"])+'.pickle'
        data_path = save_dir+synthetic_name
        df.to_pickle(data_path)
        columns_train = df.columns
        exp_config["data_options"]["dataset_fp"] = data_path
        exp_config["data_options"]["dataset_fp_train"] = None
        exp_config["data_options"]["dataset_fp_test"] = None
        exp_config["data_options"]['synthetic_train_test_split'] = False
    elif "Example0" in example:
        train_data_path, columns_train = generator_example(example[-2:], config["n_inv"], config["n_spu"], config["n_ex_train"], config["n_env"], save_dir, False, seed, dim_unc=config["n_unc"])
        test_data_path, columns_test = generator_example(example[-2:], config["n_inv"], config["n_spu"], config["n_ex_test"], config["n_env"], save_dir, True, seed, dim_unc=config["n_unc"])
        
        exp_config["data_options"]["dataset_fp_train"] = train_data_path
        exp_config["data_options"]["dataset_fp_test"] = test_data_path
        exp_config["data_options"]["dataset_fp"] = None
        exp_config["data_options"]['synthetic_train_test_split'] = True
    else:
        train_data_path, columns_train = generator_example(int(example[-1]), config["n_inv"], config["n_spu"], config["n_ex_train"], config["n_env"], save_dir, False, seed, dim_unc=config["n_unc"])
        test_data_path, columns_test = generator_example(int(example[-1]), config["n_inv"], config["n_spu"], config["n_ex_test"], config["n_env"], save_dir, True, seed, dim_unc=config["n_unc"])
        
        exp_config["data_options"]["dataset_fp_train"] = train_data_path
        exp_config["data_options"]["dataset_fp_test"] = test_data_path
        exp_config["data_options"]["dataset_fp"] = None
        exp_config["data_options"]['synthetic_train_test_split'] = True
        
        
    exp_config["data_options"]["output_data_regime"] = config["output_data_regime"]
    exp_config["data_options"]["subject_keys"] = 'Subj_ID'
    exp_config["data_options"]['environments'] = ['env_split']
    exp_config['data_options']["targets"] = ["Target"]
    
    if example == "Example4":
        if config["n_unc"]>0:
            exp_config['data_options']["predictors"] = list(columns_train[0:np.sum(config["n_inv"]) + np.sum(config["n_spu"])]) + list(columns_train[-config["n_unc"]:])
        else:
            exp_config['data_options']["predictors"] = list(columns_train[0:np.sum(config["n_inv"]) + np.sum(config["n_spu"])])
    elif example == "Example_Confounder":
        if config["n_unc"]>0:
            exp_config['data_options']["predictors"] = list(columns_train[1:config["n_inv"] + config["n_spu"]+1]) + list(columns_train[-config["n_unc"]:])
        else:
            exp_config['data_options']["predictors"] = list(columns_train[1:config["n_inv"] + config["n_spu"]+1])
    else:
        if config["n_unc"]>0:
            exp_config['data_options']["predictors"] = list(columns_train[0:config["n_inv"] + config["n_spu"]]) + list(columns_train[-config["n_unc"]:])
        else: 
            exp_config['data_options']["predictors"] = list(columns_train[0:config["n_inv"] + config["n_spu"]])
            
    del exp_config["data_options"]["remove_keys"]
    del exp_config["data_options"]["merge_keys"]
    
    if example == "Example_Confounder":
        exp_config["data_options"]['exclude'] = ['Subj_ID', 'Confounder']
    else:
        exp_config["data_options"]['exclude'] = ['Subj_ID']
        
    #return exp_config
    if config["output_data_regime"] == 'binary':
        exp_config["output_dim"] = 2
    elif config["output_data_regime"] == 'multi-class':
        exp_config["output_dim"] = 3
    else:
        exp_config["output_dim"] = 1
        
    environment_datasets, val_dataset, test_dataset = get_datasets_for_experiment(exp_config)

    return environment_datasets, val_dataset, test_dataset, exp_config