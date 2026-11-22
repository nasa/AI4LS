
import os
import yaml
import argparse
from google.cloud import storage

#gs://ah21_data/validation/Example1_dim_inv_6_dim_spu_6_dim_unc_500_n_exp_300_n_env_5/seed_1

parser = argparse.ArgumentParser(description='To Parse Experiment directories')
parser.add_argument("--dir_ind", default=0, help="Experiment Directory index to mount", type=int)
parser.add_argument("--seed", default=0, help="seed number", type=int)
parser.add_argument("--repeats", default=1, help="number of seed runs: seed+i", type=int)
parser.add_argument("--n_colabs", default=5, help="number of collaborators", type=int)
parser.add_argument("--rounds_to_train", default=100, help="number of aggregator training rounds ", type=int)
parser.add_argument("--num_epochs", default=30, help="number of training epochs per round ", type=int)
parser.add_argument("--batch_size", default=50, help="number of training samples per batch ", type=int)
parser.add_argument("--output_data_regime", default="real-valued", help="output type of the dataset being evaluated", type=str)
parser.add_argument("--num_classes", default=1, help="number of output classes. regression=1, binary=2 etc.", type=int)
parser.add_argument("--index_col", default=False, help="Does the datasets csv files contain an index column that needs removing", type=bool)
parser.add_argument("--bucket_blob_prefix", default='validation/Example', help="ah21_data bucket blob prefix", type=str)
parser.add_argument("--num_features", default=None, help='desired number of featurs after dimensionality reduction', type=int)
parser.add_argument("--user", default="user", help="user name for data directory mounting", type=str)

args = parser.parse_args()
print("dir_ind %s: type: %s" % (args.dir_ind, args.output_data_regime))
print("seed:repeats", args.seed, args.repeats)
print("n_collabs", args.n_colabs)

# assume this file is called from crisp base-dir
crisp_dir = os.getcwd()

client = storage.Client()
blobs = client.list_blobs('ah21_data', prefix= args.bucket_blob_prefix)

# get the list of Experiments in the GCP Bucket
experiments = set([])
for blob in blobs:
    if "validation" in blob.name:
        exp = blob.name.split("/")[1]
    else:
        exp = blob.name.split("/")[0]
    experiments.add(exp)
experiments = list(experiments)
experiments.sort()
print(experiments)

mount_point= "/home/" + args.user + "/data/"
for i in range(args.repeats):
    print("repeat  = ", i+1)
    seed = args.seed+i
    try:
        os.system("fusermount -u %s" %mount_point)
    except:
        pass

    dir = os.path.join(experiments[args.dir_ind], "seed_%s" % seed)
    if "validation" in args.bucket_blob_prefix:
        dir = os.path.join("validation", experiments[args.dir_ind], "seed_%s" % seed)
    print("\nmount dir: ", dir)
    print("seed: i", args.seed, i)

    try:
        os.system("gcsfuse --implicit-dirs -only-dir %s ah21_data %s" %(dir, mount_point))
    except:
        pass
    print("gcsfuse --implicit-dirs -only-dir %s ah21_data %s" %(dir, mount_point))

    # update the PLAN:
    #os.chdir("/home/user/crisp")
    #os.system("chmod 755 fl_plan/plan.yaml")
    #os.system("chmod 755 fl_plan/data.yaml")
    with open('fl_plan/plan_bk.yaml') as f:
        plan = yaml.safe_load(f)

    # input the random seed into the plan.yaml file
    plan['aggregator']['settings']['rounds_to_train'] = args.rounds_to_train

    plan['data_loader']['settings']['seed'] = int(seed)
    plan['data_loader']['settings']['collaborator_count'] = args.n_colabs
    plan['data_loader']['settings']['batch_size'] = args.batch_size
    plan['data_loader']['settings']['index'] = args.index_col

    plan['task_runner']['settings']['seed'] = int(seed)
    plan['task_runner']['settings']['output_data_regime'] = args.output_data_regime
    plan['task_runner']['settings']['num_classes'] = args.num_classes

    plan['tasks']['train_IRM']['kwargs']['num_epochs'] = args.num_epochs

    # You can request the dimentionality reduction task, here:
    if args.num_features is not None:
        plan['task_runner']['settings']['num_features'] = args.num_features
        plan["assigner"]["settings"]["task_groups"][0]['tasks'] = ['dimensionality_reduction', 'aggregated_model_validation', 'train_IRM', 'locally_tuned_model_validation']
        plan["task_runner"]["template"] = "src.dim_red_crisp_task_runner.CRISPTaskRunner"
    else:
        plan["task_runner"]["template"] = "src.crisp_task_runner.CRISPTaskRunner"

    with open('fl_plan/plan.yaml', 'w') as outfile:
        yaml.dump(plan, outfile, default_flow_style=False)

    col_list = []
    for i in range(args.n_colabs):
        col_list.append("col_%s" %i)

    dataF = open("fl_plan/data.yaml", "w")
    for i in range(args.n_colabs):
        dataF.write( "col_%s,%s" % (i, os.path.join(mount_point, "col_%s" % i)  ))
        dataF.write("\n")
    dataF.close()

    with open('fl_plan/cols.yaml', 'w') as outfile:
        yaml.dump({"collaborators" : col_list}, outfile, default_flow_style=False)

    # build the federated workspace environments:
    os.system("./federation.sh %s" % args.n_colabs)

    ## run the tmux scrip:
    #tmux_launcher = "fl_tmux.sh"
    #if args.n_colabs > 6:
    tmux_launcher = "fl_tmux_launcher.sh"
    os.system("./%s %s true %s fed_%s" % (tmux_launcher, args.n_colabs, mount_point, seed))

print("\nDone")

