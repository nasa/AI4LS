import os

experiments_dim_red = {
        'final_elijah_mouse_experiment' : {
            "dir_ind" : 0,
            "seed" : 20,
            "repeats" : 10,
            "n_colabs" : 6,
            "rounds_to_train" : 300,
            "num_epochs" : 1,
            "batch_size" : 1000,
            "output_data_regime" : "real-valued",
            "num_classes" : 1,
            "index_col" : "True",
            "bucket_blob_prefix" : "final_elijah_mouse_experiment",
            "num_features" : 1000,
            "tmux_launcher" : "fl_tmux.sh",
            },
        'final_gamma_ray_experiment' : {
            "dir_ind" : 0,
            "seed" : 20,
            "repeats" : 10,
            "n_colabs" : 7,
            "rounds_to_train" : 300,
            "num_epochs" : 1,
            "batch_size" : 1000,
            "output_data_regime" : "binary",
            "num_classes" : 2,
            "index_col" : "True",
            "bucket_blob_prefix" : "final_gamma_ray_experiment",
            "num_features" : 1000,
            "tmux_launcher" : "fl_tmux_launcher.sh",
            },
        'final_heavy_ion_experiment': {
            "dir_ind" : 0,
            "seed" : 10,
            "repeats": 10,
            "n_colabs" : 3,
            "rounds_to_train": 300,
            "num_epochs" : 1,
            "batch_size" : 1000,
            "output_data_regime" : "binary",
            "num_classes" : 2,
            "index_col" :"true",
            "bucket_blob_prefix" : "final_heavy_ion_experiment",
            "num_features" : 1000,
            "tmux_launcher" : "fl_tmux.sh",
            }
        }

instance_name = "fdr-1-epoch-20-30"
for experiment, case in experiments_dim_red.items():
    print("\n",experiment)

    #os.system(
    print(
    "gcloud beta compute --project=fdl-us-astronaut-health instances create-with-container linus-%s-%s \
    --zone=us-central1-a --machine-type=e2-standard-4 --subnet=default --network-tier=PREMIUM --metadata=google-logging-enabled=true \
    --maintenance-policy=MIGRATE --service-account=694584192919-compute@developer.gserviceaccount.com --scopes=https://www.googleapis.com/auth/cloud-platform \
    --image=cos-stable-89-16108-470-11 --image-project=cos-cloud --boot-disk-size=100GB --boot-disk-type=pd-balanced --boot-disk-device-name=linus-multi-runner-%s \
    --no-shielded-secure-boot --shielded-vtpm --shielded-integrity-monitoring --container-image=gcr.io/fdl-us-astronaut-health/ah-crisp-validate --container-restart-policy=on-failure \
    --container-privileged --container-stdin --container-tty --container-command=python --container-arg=/home/user/crisp/experiment_runners/multi_run.py --container-arg=--dir_ind=%i \
    --container-arg=--seed=%i \
    --container-arg=--repeats=%i \
    --container-arg=--n_colabs=%i \
    --container-arg=--rounds_to_train=%i \
    --container-arg=--num_epochs=%i \
    --container-arg=--batch_size=%i \
    --container-arg=--output_data_regime=%s \
    --container-arg=--num_classes=%i \
    --container-arg=--index_col=%s \
    --container-arg=--bucket_blob_prefix=%s \
    --container-arg=--num_features=%s \
    --container-arg=--tmux_launcher=%s \
    --labels=container-vm=cos-stable-89-16108-470-11" \
    % (instance_name, experiment.lower().replace("_","-"), experiment.lower(), case["dir_ind"], case["seed"], case["repeats"], case["n_colabs"], case["rounds_to_train"], case["num_epochs"], case["batch_size"], case["output_data_regime"], case["num_classes"], case["index_col"], case["bucket_blob_prefix"], case["num_features"], case["tmux_launcher"])
    )
