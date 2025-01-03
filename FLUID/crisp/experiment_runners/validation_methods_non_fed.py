import os
import yaml
import argparse
from google.cloud import storage

#gs://ah21_data/validation/Example1_dim_inv_6_dim_spu_6_dim_unc_500_n_exp_300_n_env_5/seed_parser = argparse.ArgumentPars>

parser = argparse.ArgumentParser(description='To Parse Experiment directories')

parser.add_argument('--example', default='Example1', help='Example to run the validation on', type=str)
parser.add_argument('--method', default='CSNX', help='Causal method that we want to implement', type=str)
parser.add_argument('--n_seeds', default=100, help='number of seeds that should run', type=int)
parser.add_argument('--dim_inv', default=6, help='dimension of the invariant variables', type=int)
parser.add_argument('--dim_spu', default=6, help='dimension of the spurius variables', type=int)
parser.add_argument('--dim_unc', default=1, help='dimension of the uncorrelated variables', type=int)
parser.add_argument('--n_samp', default=900, help='number of samples that we are going to consider', type=int)
parser.add_argument('--n_env', default=5, help='number of environments that we consider', type=int)
parser.add_argument('--bucket', default=False, help='using buckets or not', type=bool)
parser.add_argument('--data_dir', default='data/synthetic', help='directory where the data are saved', type=str)
parser.add_argument('--save_dir', default='results/validation_bucket', help='directory where the experiments are saved', type=str)

args = parser.parse_args()



experiment_name = args.example +'_dim_inv_'+str(args.dim_inv)+\
                '_dim_spu_'+str(args.dim_spu)+'_dim_unc_'+ str(args.dim_unc) + \
                '_n_exp_'+str(args.n_samp)+\
                '_n_env_'+str(args.n_env) +'/'

# Unmount any preceeding mount
try:
    os.system("umount %s" %("/home/user/crisp/results/"+experiment_name))
except:
    pass

if not os.path.exists("/home/user/crisp/data/synthetic"): 
    os.makedirs("/home/user/crisp/data/synthetic")

client = storage.Client()
blobs = client.list_blobs('ah21_data', prefix='validation/Example')

# get the list of Experiments in the GCP Bucket
try:
    os.system("umount %s" %("/home/user/crisp/results/"))
except:
    pass

experiment_already_run = False
for blob in blobs:
    exp = blob.name.split("/")[1]
    if exp == experiment_name:
        experiment_already_run = True



if not experiment_already_run:
    # we need to generate the experiment folder in the bucket
    print('Experiment non Existing')
    mount_point = "/home/user/crisp/results"
    direct = "validation"
    try:
        os.system("gcsfuses  -o allow_other --implicit-dirs -only-dir %s ah21_data %s" %(direct,mount_point))
    except:
        pass
    # generate the new directory
    exp_dir = os.path.join(mount_point, experiment_name)
    print('Generating folder in the bucket')
    if not os.path.exists(exp_dir):
        #os.system("mkdir %s"%(exp_dir))
        oldmask = os.umask(000)
        os.makedirs(exp_dir)
        os.umask(oldmask)
    print('Generationfolder successfull')
    # unmount the directory


    try:
        os.system("umount %s" %(mount_point))
    except:
        pass



# cehcking that the saving directory is present in the local machine
mount_point = "/home/user/crisp/results"
path = os.path.join(mount_point, experiment_name)
print(path)

if not os.path.exists(path): 
    os.system("mkdir %s"%(path)) #makedirs(path)

direct = os.path.join("validation", experiment_name)
path_to_results = mount_point
mount_point = os.path.join(mount_point, experiment_name)
print("\nmount dir: ", direct)

try:
    os.system("gcsfuse -only-dir %s ah21_data %s  " %(direct, mount_point))
except:
    pass

print("gcsfuse --implicit-dirs -only-dir %s ah21_data %s" %(direct, mount_point))

print('###### mounting at ', mount_point)


## run the tmux scrip:
os.chdir("/home/user/crisp")
os.system("python utils/Validation_Runner_non_fed.py --example=%s --method=%s \
--n_seeds=%i --dim_unc=%i --save_dir=%s --data_dir=%s \
--dim_inv=%i --dim_spu=%i --n_samp=%i --n_env=%s" %
             (args.example, args.method, args.n_seeds, args.dim_unc,
              path_to_results, os.path.join(os.getcwd(),args.data_dir),
              args.dim_inv, args.dim_spu, args.n_samp, args.n_env))

print("\nDone")
