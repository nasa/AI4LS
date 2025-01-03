import os
examples = ['Example3',"Example4","Example5"]
methods = ['ERM', 'CSNX', 'IRM', 'SNLIRM']
n_seeds = int(101)
experiments_fb = {
#     'case0b' : {"n_unc" : 0,
#            "n_samp" : 300,
#           "n_env": 5,
#           "n_spu": 50,
#           "n_inv":10},
#     'case1b' : {
#         "n_unc": 0,
#         "n_samp": 900,
#           "n_env":5,
#           "n_spu":500,
#           "n_inv":10
#         },
    'case0' : {"n_unc" : 50,
               "n_samp" : 300,
              "n_env":5,
              "n_spu":6,
              "n_inv":6},
    'case1' : {
    "n_unc": 50,
    "n_samp": 900,
              "n_env":5,
              "n_spu":6,
              "n_inv":6
        },
    'case2' : {
            "n_unc": 500,
            "n_samp": 300,
              "n_env":5,
              "n_spu":6,
              "n_inv":6
        },
    'case3' : {
            "n_unc": 500,
            "n_samp": 900,
              "n_env":5,
              "n_spu":6,
              "n_inv":6
        }
}

experiments_0 = {
    'case0' : {"n_unc" : 0,
               "n_samp" : 3000,
              "n_env":3,
              "n_spu":5,
              "n_inv":5}}

for example in examples:
    if '0' in example:
        experiments_var = experiments_0
    else:
        experiments_var = experiments_fb
    for method in methods:
        for case in experiments_var.keys():
            print(case)
            os.system("gcloud beta compute --project=fdl-us-astronaut-health instances create-with-container giuseppe-test-%s-%s-%s --zone=us-central1-a --machine-type=e2-standard-4 --subnet=default --network-tier=PREMIUM --metadata=google-logging-enabled=true --maintenance-policy=MIGRATE --service-account=694584192919-compute@developer.gserviceaccount.com --scopes=https://www.googleapis.com/auth/cloud-platform --image=cos-stable-89-16108-470-11 --image-project=cos-cloud --boot-disk-size=50GB --boot-disk-type=pd-balanced --boot-disk-device-name=giuseppe-test-%s-%s-%s --no-shielded-secure-boot --shielded-vtpm --shielded-integrity-monitoring --container-image=gcr.io/fdl-us-astronaut-health/ah-crisp-validate --container-restart-policy=on-failure --container-privileged --container-stdin --container-tty --container-command=python --container-arg=/home/user/crisp/experiment_runners/validation_methods_non_fed.py --container-arg=--method=%s --container-arg=--example=%s --container-arg=--n_samp=%i  --container-arg=--dim_unc=%i --container-arg=--n_seeds=%i --container-arg=--dim_inv=%i --container-arg=--dim_spu=%i --container-arg=--n_env=%i --labels=container-vm=cos-stable-89-16108-470-11"%(method.lower(), case, example.lower(), method.lower(), case, example.lower(), method, example, experiments_var[case]["n_samp"], experiments_var[case]["n_unc"], n_seeds, experiments_var[case]["n_inv"], experiments_var[case]["n_spu"], experiments_var[case]["n_env"]))
