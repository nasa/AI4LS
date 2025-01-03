import pandas as pd
import synthetic.facebook_synthetic_data as datasets
import numpy as np
import os
import torch


def generator_example(n_example, dim_inv=5, dim_spu=5, n_exp=int(1e4), n_env=2,
                      save_dir="", test=False, seed=0, save=True, dim_unc=0):
    """
    This function will generate data according to example n_example and it will save
    them in the save_dir with the following format title 
        
        save_dir/data_fb_example_1_n_inv_X_n_spu_X_n_exp_X_n_env_X_test_X.pickle
    
    as a pandas dataframe with 
        
        n_example : number of the example that has to be selected
        dim_inv : dimension of the invariant features with the target
        dim_spu : dimension of the spurius variables
        n_exp : number of samples that are taken 
        n_env : number of environments that are going to be considered
        save_dir : directory where to save the generated data
  
    This is based on the work in 
    
            https://github.com/facebookresearch/InvarianceUnitTests/
        
    """
    assert n_exp%n_env == 0
    
    # Let's set the seed to 0 for reproducability (the test and train might be taking different elements otherwise)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic=True
    #torch.set_rng_state(seed)
    
    # Let's upload the class example
    Exmpl = datasets.DATASETS['Example'+str(n_example)](dim_inv, dim_spu, n_env)
    
    # Let's update the flag for the training or testing variable
    split = 'test' if test else 'train'
    
    # Let's upload all the experiments for each environment  (else condition for multi-target output)
    count_env = 0
    
    if n_example == 4: 
        if len(dim_inv) == 2 or len(dim_spu) == 2:
            dim_inv = dim_inv[0]+dim_inv[1]
            dim_spu = dim_spu[0]+dim_spu[1]

    X = np.zeros((n_exp, dim_inv+dim_spu+3))
    
    X[:,-2] = np.arange(n_exp)
    
    # generate data per environment from the sample method
    for env in Exmpl.envs:
        X_env, y_env = Exmpl.sample(
                        n=n_exp//n_env,
                        env=env,
                        split=split)

        X[count_env*n_exp//n_env:(count_env+1)*n_exp//n_env,:][:,:-3] = X_env
        X[count_env*n_exp//n_env:(count_env+1)*n_exp//n_env,:][:,-3] = count_env* np.ones(n_exp//n_env)

        X[count_env*n_exp//n_env:(count_env+1)*n_exp//n_env,:][:,-1] = y_env.resize_(y_env.size()[0])      
        count_env += 1


    # Let's define the 
    columns = []
    for i in range(dim_inv):
        columns.append("Causal_" + str(i))
    for i in range(dim_spu):
        columns.append("Non_causal_" + str(i))

    columns.append("env_split")
    columns.append("Subj_ID")
    columns.append("Target")
    
    df = pd.DataFrame(data=X, columns=columns)
    
    # add the uncorrelated features
    mean_std = np.std(X[:,:dim_inv+dim_spu])
    
    for i in range(dim_unc):
        df["Uncorrelated_"+str(i)] = np.random.randn(n_exp)*np.random.rand(1)*2*mean_std
    
    
    name = 'data_fb_example_'+str(n_example)+'_dim_inv_'+str(dim_inv)+\
                '_dim_spu_'+str(dim_spu)+'_dim_unc_'+str(dim_unc)+'_n_exp_'+str(n_exp)+\
                '_n_env_'+str(n_env)+'_test_'+str(test)
    save_name = (name +'.pickle')
    

    if save:
        # Let's save the df
        output_file = os.path.join(save_dir, save_name)
        df.to_pickle(output_file)
        print('Generated Synthetic Data according to the Facebook setup Example: '  +str(n_example))
        print('     df with ', str(len(df.keys())-3),' columns')
        return output_file, df.columns
    else:
#         print('Generated Synthetic Data according to the Facebook setup Example: '  +str(n_example))
        return df
    
