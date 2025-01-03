import os
import yaml
import argparse
from google.cloud import storage

#gs://ah21_data/validation/Example1_dim_inv_6_dim_spu_6_dim_unc_500_n_exp_300_n_env_5/seed_1

parser = argparse.ArgumentParser(description='To Parse Experiment directories')
parser.add_argument("dir_ind", help="Experiment Directory index to mount", type=int)
parser.add_argument("seed", help="seed number", type=int)

args = parser.parse_args()
print("dir_ind: ", args.dir_ind)
print("seed: ", args.seed)

client = storage.Client()
blobs = client.list_blobs('ah21_data', prefix='validation/Example')

experiments = set([])
for blob in blobs:
    exp = blob.name.split("/")[1]
    experiments.add(exp)

experiments = list(experiments)
experiments.sort()

dir = os.path.join("validation", experiments[args.dir_ind], "seed_%s" % args.seed)
print("Mount dir:", dir)

mount_point= "/home/user/data"

try:
    os.system("gcsfuse --implicit-dirs -only-dir %s ah21_data %s" %(dir, mount_point))
except:
    pass
print("gcsfuse --implicit-dirs -only-dir %s ah21_data %s" %(dir, mount_point))

# update the seed:
os.chdir("/home/user/crisp")
os.system("chmod 777 fl_plan/plan.yaml")
with open('fl_plan/plan.yaml') as f:
    plan = yaml.safe_load(f)
plan['task_runner']['settings']['seed'] = int(args.seed)

with open('fl_plan/plan.yaml', 'w') as outfile:
    yaml.dump(plan, outfile, default_flow_style=False)

## run the federation script:
os.system("./federation.sh")

## run the tmux scrip:
os.system("./fl_tmux.sh %s" % mount_point)

print("\nDone")
