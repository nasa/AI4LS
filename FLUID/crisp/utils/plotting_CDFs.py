import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--example', default='Example1', help='Example to run the validation on', type=str)
parser.add_argument('--method', default='IRM', help='Causal method that we want to implement', type=str)
parser.add_argument('--n_seeds', default=100, help='number of seeds that should run', type=int)
parser.add_argument('--dim_inv', default=6, help='dimension of the invariant variables', type=int)
parser.add_argument('--dim_spu', default=6, help='dimension of the spurius variables', type=int)
parser.add_argument('--dim_unc', default=500, help='dimension of the uncorrelated variables', type=int)
parser.add_argument('--n_samp', default=900, help='number of samples that we are going to consider', type=int)
parser.add_argument('--n_env', default=5, help='number of environments that we consider', type=int)
parser.add_argument('--confidence_int', default=True, help='confidence intervals', type=bool)
parser.add_argument('--data_dir', default='results/validation_bucket', help='directory where the plots are saved', type=str)
parser.add_argument('--bucket_dir', default='validation', help='directory where the plots are saved', type=str)

args = parser.parse_args()



def counting_causal(results, n_inv, multiplicator=1, confounder=False):
    """
    This function counts the number of causal features that a method has
    retrieved with a maximum number of mutliplicator times the actual number of
    causal variables to be returned.

    args:
        - results: dictionary with the ordered elements per causality intensity
        - n_inv: number of invariant variables
        - multiplicator: how many times the actual number of causal variables
                we allow the methods to return us
        - confounder: boolean on a confounder being present in the dataset    
    """
    found = 0
    count = 0

    # Set the number of variables that we are going the
    # methods to return us

    if confounder:
        n_considered = np.ceil(multiplicator*n_inv+1)
    else:
        n_considered = np.ceil(multiplicator*n_inv)

    # Let's now go throuh what the methods have returned
    for j in range(len(results)):
        feature = results[j][0]
        if ('Causal' in feature) or ('Confounder' == feature):
#             print(feature)
            found += 1
        if confounder:
            if ('Z' in feature):
                found += 1
        count +=1
        if count == n_considered:
            break

    return found/n_considered


def statistic_on_results(results):
    keys = results.keys()
    fraction_found = []
    for key in keys:
        list_features = sorted(zip(results[key]["to_bucket"]["features"], results[key]["to_bucket"]["coefficients"]), key=lambda x: abs(x[1]), reverse=True)
        fraction_found.append(counting_causal(
                    list_features,
                    4,
                    1))
    return fraction_found

def CDF_successful_causals_links(results):
    CDF = [0]
    for j in range(len(results)):
        if 'Causal' in results[j][0]:
            CDF.append(CDF[-1]+1)
        else:
            CDF.append(CDF[-1])
    return CDF

def CDF_related_links(results, causalnex=False):
    CDF = [0]
    for j in range(len(results)):
        if 'Causal' in results[j][0] or 'Non' in results[j][0]:
            CDF.append(CDF[-1]+1)
        else:
            CDF.append(CDF[-1])
    return CDF


prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

def plot_variance(xx,results, index, label, percentiles = [5,25,50,75,95], CI=True):

    rsDist = np.zeros((len(percentiles), results.shape[1]))

    for i in range(len(percentiles)):
        rsDist[i,:]=np.percentile(results,percentiles[i], axis=0)

    half = int((len(percentiles)-1)/2)

    plt.semilogx(xx, rsDist[half,:],color=colors[index], label=label)
    if CI:
        for i in range(half):
            plt.fill_between(xx, rsDist[i,:],rsDist[-(i+1),:],color=colors[index], alpha=(i/half))

            
# Let's mount the results directory
mount_dir = args.bucket_dir
mount_point = os.getcwd() + "/results/"
print(mount_point)
try:
    os.system("fusermount -u %s" %(mount_point))
except:
    pass
print('unmounted')
try:
    os.system("gcsfuse -o nonempty --implicit-dirs -only-dir %s ah21_data %s  " %(mount_dir,mount_point))
except:
    pass
print("gcsfuses  -o allow_other --implicit-dirs -only-dir %s ah21_data %s" %(mount_dir,mount_point))

list_examples = ['Example6']#,'Example2','Example3','Example4',
#                 'Example5','Example_Confounder', 'Example_Nonlinear']

list_methods = ['IRM', 'ERM', 'CSNX', 'SNLIRM', 'CSNX_ENV', 'IRM_fed']

cwd = os. getcwd()


for example in list_examples: 
    
    list_results = []
    list_results_related = []
    list_methods_used = []

    experiment_name = example +'_dim_inv_'+str(args.dim_inv)+\
                '_dim_spu_'+str(args.dim_spu)+'_dim_unc_'+ str(args.dim_unc) + \
                '_n_exp_'+str(args.n_samp)+ \
                '_n_env_'+str(args.n_env) + '/'
    if mount_dir == "results_validation":
        experiment_name = experiment_name[:-1] + '_n_seeds_' + str(args.n_seeds) + '/'
    
    # Let's read the data and compute the CDFs

    path = os.path.join(mount_point, experiment_name)

    for method in list_methods:
        # search for the result
        save_dir = path +method+'.json'

        print(save_dir)
        
        if os.path.isfile(save_dir):
            print('file', save_dir)
            list_methods_used.append(method)
            
            results = json.load(open(save_dir,'r'))

            print(len(results))
            keys = results.keys()
            CDFs = []
            CDFs_linked = []
            for key in keys:
                if method == "CSNX":
                    coefs = results[key]["to_bucket"]["pvals"]
                else:
                    coefs = results[key]["to_bucket"]["coefficients"]
                if len(coefs) >3:
                    sorted_results = sorted(zip(results[key]["to_bucket"]["features"], coefs), key=lambda x: abs(x[1]), reverse=True)
                else:
                    coefs = np.array(coefs)
                    coefs = np.sum(coefs, axis=0)
                    sorted_results = sorted(zip(results[key]["to_bucket"]["features"], coefs), key=lambda x: abs(x[1]), reverse=True)
                CDFs.append(CDF_successful_causals_links(sorted_results))
                CDFs_linked.append(CDF_related_links(sorted_results))

            list_results.append(CDFs)
            list_results_related.append(CDFs_linked)

            
    # Let's plot the CDFs
    
    if len(list_results)>0:
        print('saving the results')
        fig = plt.figure()
        xx = np.arange(len(list_results[0][0]))
        for j in range(len(list_methods_used)):

            CDFs = np.array(list_results[j])
            plot_variance(xx, CDFs, j, list_methods_used[j])

        plt.legend()
        plt.xlabel(r'$\alpha$ = $\#$ features considered')
        plt.ylabel(r'CDF($\alpha$)')
        name_fig = path + 'plots/Comparison_on_the_invaraint_CDFs.pdf'
        if not os.path.exists(os.path.join(path, 'plots')): 
            os.makedirs(os.path.join(path, 'plots'))
        fig.savefig(name_fig)

        fig = plt.figure()
        xx = np.arange(len(list_results_related[0][0]))
        for j in range(len(list_methods_used)):

            CDFs = np.array(list_results_related[j])
            plot_variance(xx, CDFs, j, list_methods_used[j], CI=args.confidence_int)

        plt.legend()
        plt.xlabel(r'$\alpha$ = $\#$ features considered')
        plt.ylabel(r'CDF($\alpha$)')
        name_fig = path + 'plots/Comparison_on_the_related_CDFs_CI_'+str(args.confidence_int)+'.pdf'
        print(name_fig)
        if not os.path.exists(os.path.join(path, 'plots')): 
            os.makedirs(os.path.join(path, 'plots'))
        fig.savefig(name_fig)

