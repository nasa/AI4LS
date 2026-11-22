import os

instance_name = "fed-runner-1epoch-wed"

experiments = {
#    'Example0a': {"dir_ind" : 0,
#                   "seed" : 90,
#                   "repeats" : 10,
#                   "n_colabs" : 3,
#                   "rounds_to_train": 3000,
#                   "num_epochs" : 1,
#                   "batch_size" : 1000,
#                   "output_data_regime" : "real-valued",
#                   "num_classes" : 1,
#                   "index_col" :"false",
#                   "bucket_blob_prefix" : "validation/Example0a_dim_inv_5_dim_spu_5_dim_unc_0_n_exp_3000_n_env_3"
#                   },

#     'Example0b': {"dir_ind" : 0,
#                   "seed" : 90,
#                   "repeats" : 10,
#                   "n_colabs" : 3,
#                   "rounds_to_train" :3000,
#                   "num_epochs" : 1,
#                   "batch_size" : 1000,
#                   "output_data_regime" : "binary",
#                   "num_classes" : 1,
#                   "index_col" :"false",
#                   "bucket_blob_prefix" : "validation/Example0b_dim_inv_5_dim_spu_5_dim_unc_0_n_exp_3000_n_env_3"
#                   },

    # 'Example1d': {"dir_ind" : 0,
    #               "seed" : 90,
    #               "repeats": 10,
    #               "n_colabs" : 5,
    #               "rounds_to_train" :200,
    #               "num_epochs" : 1,
    #               "batch_size" : 200,
    #               "output_data_regime" : "real-valued",
    #               "num_classes" : 1,
    #               "index_col" :"false",
    #               "bucket_blob_prefix" : "validation/Example1_dim_inv_10_dim_spu_50_dim_unc_0_n_exp_900_n_env_5/"
    #               },

    'Example2d': {"dir_ind" : 0,
                  "seed" : 90,
                  "repeats" : 10,
                  "n_colabs" : 5,
                  "rounds_to_train" :200,
                  "num_epochs" : 1,
                  "batch_size" : 60,
                  "output_data_regime" : "binary",
                  "num_classes" : 2,
                  "index_col" :"false",
                  "bucket_blob_prefix" : "validation/Example2_dim_inv_10_dim_spu_50_dim_unc_0_n_exp_900_n_env_5/"
                  },

    'ExampleCon': {"dir_ind" : 0,
                  "seed" : 90,
                  "repeats" : 10,
                  "n_colabs" : 5,
                  "rounds_to_train" :200,
                  "num_epochs" : 1,
                  "batch_size" : 1000,
                  "output_data_regime" : "real-valued",
                  "num_classes" : 1,
                  "index_col" :"false",
                  "bucket_blob_prefix" : "validation/Example_Confounder_dim_inv_10_dim_spu_50_dim_unc_0_n_exp_900_n_env_5/"
                  },


    # 'Example0a': {"dir_ind" : 0,
    #               "seed" : 0,
    #               "repeats" : 30,
    #               "n_colabs" : 3,
    #               "rounds_to_train": 200,
    #               "num_epochs" : 30,
    #               "batch_size" : 60,
    #               "output_data_regime" : "real-valued",
    #               "num_classes" : 1,
    #               "index_col" :"false",
    #               "bucket_blob_prefix" : "validation/Example0a_dim_inv_5_dim_spu_5_dim_unc_0_n_exp_3000_n_env_3"
    #               },

    # 'Example0b': {"dir_ind" : 0,
    #               "seed" : 0,
    #               "repeats" : 30,
    #               "n_colabs" : 3,
    #               "rounds_to_train" :200,
    #               "num_epochs" : 30,
    #               "batch_size" : 60,
    #               "output_data_regime" : "binary",
    #               "num_classes" : 1,
    #               "index_col" :"false",
    #               "bucket_blob_prefix" : "validation/Example0b_dim_inv_5_dim_spu_5_dim_unc_0_n_exp_3000_n_env_3"
    #               },

    # 'Example1a': {"dir_ind" : 0,
    #               "seed" : 0,
    #               "repeats" : 30,
    #               "n_colabs" : 5,
    #               "rounds_to_train" : 200,
    #               "num_epochs" : 30,
    #               "batch_size" : 60,
    #               "output_data_regime" : "real-valued",
    #               "num_classes" : 1,
    #               "index_col" :"false",
    #               "bucket_blob_prefix" : "validation/Example1_dim_inv_10_dim_spu_500_dim_unc_0_n_exp_300_n_env_5/"
    #               },

    # 'Example1b': {"dir_ind" : 0,
    #               "seed" : 0,
    #               "repeats" : 30,
    #               "n_colabs" : 5,
    #               "rounds_to_train" :200,
    #               "num_epochs" : 30,
    #               "batch_size" : 60,
    #               "output_data_regime" : "real-valued",
    #               "num_classes" : 1,
    #               "index_col" :"false",
    #               "bucket_blob_prefix" : "validation/Example1_dim_inv_10_dim_spu_500_dim_unc_0_n_exp_900_n_env_5/"
    #               },

    # 'Example1c': {"dir_ind" : 0,
    #               "seed" : 0,
    #               "repeats" : 30,
    #               "n_colabs" : 5,
    #               "rounds_to_train" : 200,
    #               "num_epochs" : 30,
    #               "batch_size" : 60,
    #               "output_data_regime" : "real-valued",
    #               "num_classes" : 1,
    #               "index_col" :"false",
    #               "bucket_blob_prefix" : "validation/Example1_dim_inv_10_dim_spu_50_dim_unc_0_n_exp_300_n_env_5/"
    #               },

    # 'Example1d': {"dir_ind" : 0,
    #               "seed" : 0,
    #               "repeats": 30,
    #               "n_colabs" : 5,
    #               "rounds_to_train" :200,
    #               "num_epochs" : 30,
    #               "batch_size" : 60,
    #               "output_data_regime" : "real-valued",
    #               "num_classes" : 1,
    #               "index_col" :"false",
    #               "bucket_blob_prefix" : "validation/Example1_dim_inv_10_dim_spu_50_dim_unc_0_n_exp_900_n_env_5/"
    #               },

    # 'Example1e': {"dir_ind" : 0,
    #               "seed" : 0,
    #               "repeats": 30,
    #               "n_colabs" : 5,
    #               "rounds_to_train" :200,
    #               "num_epochs" : 30,
    #               "batch_size" : 100,
    #               "output_data_regime" : "real-valued",
    #               "num_classes" : 1,
    #               "index_col" :"false",
    #               "bucket_blob_prefix" : "validation/Example1_dim_inv_6_dim_spu_6_dim_unc_500_n_exp_300_n_env_5"
    #               },

    # 'Example1f': {"dir_ind" : 0,
    #               "seed" : 0,
    #               "repeats": 30,
    #               "n_colabs" : 5,
    #               "rounds_to_train" :200,
    #               "num_epochs" : 30,
    #               "batch_size" : 100,
    #               "output_data_regime" : "real-valued",
    #               "num_classes" : 1,
    #               "index_col" :"false",
    #               "bucket_blob_prefix" : "validation/Example1_dim_inv_6_dim_spu_6_dim_unc_500_n_exp_900_n_env_5"
    #               },

    # 'Example1g': {"dir_ind" : 0,
    #               "seed" : 0,
    #               "repeats": 30,
    #               "n_colabs" : 5,
    #               "rounds_to_train" :200,
    #               "num_epochs" : 30,
    #               "batch_size" : 100,
    #               "output_data_regime" : "real-valued",
    #               "num_classes" : 1,
    #               "index_col" :"false",
    #               "bucket_blob_prefix" : "validation/Example1_dim_inv_6_dim_spu_6_dim_unc_50_n_exp_300_n_env_5"
    #               },

    # 'Example1h': {"dir_ind" : 0,
    #               "seed" : 0,
    #               "repeats": 30,
    #               "n_colabs" : 5,
    #               "rounds_to_train" :200,
    #               "num_epochs" : 30,
    #               "batch_size" : 100,
    #               "output_data_regime" : "real-valued",
    #               "num_classes" : 1,
    #               "index_col" :"false",
    #               "bucket_blob_prefix" : "validation/Example1_dim_inv_6_dim_spu_6_dim_unc_50_n_exp_900_n_env_5"
    #               },

    # 'Example2a': {"dir_ind" : 0,
    #               "seed" : 0,
    #               "repeats": 30,
    #               "n_colabs" : 5,
    #               "rounds_to_train": 200,
    #               "num_epochs" : 30,
    #               "batch_size" : 60,
    #               "output_data_regime" : "binary",
    #               "num_classes" : 2,
    #               "index_col" :"false",
    #               "bucket_blob_prefix" : "validation/Example2_dim_inv_10_dim_spu_500_dim_unc_0_n_exp_300_n_env_5/"
    #               },

    # 'Example2b': {"dir_ind" : 0,
    #               "seed" : 0,
    #               "repeats" :30,
    #               "n_colabs" : 5,
    #               "rounds_to_train" :200,
    #               "num_epochs" : 30,
    #               "batch_size" : 60,
    #               "output_data_regime" : "binary",
    #               "num_classes" : 2,
    #               "index_col" :"false",
    #               "bucket_blob_prefix" : "validation/Example2_dim_inv_10_dim_spu_500_dim_unc_0_n_exp_900_n_env_5/"
    #               },

    # 'Example2c': {"dir_ind" : 0,
    #               "seed" : 0,
    #               "repeats" :30,
    #               "n_colabs" : 5,
    #               "rounds_to_train":200,
    #               "num_epochs" : 30,
    #               "batch_size" : 60,
    #               "output_data_regime" : "binary",
    #               "num_classes" : 2,
    #               "index_col" :"false",
    #               "bucket_blob_prefix" : "validation/Example2_dim_inv_10_dim_spu_50_dim_unc_0_n_exp_300_n_env_5/"
    #               },

    # 'Example2d': {"dir_ind" : 0,
    #               "seed" : 0,
    #               "repeats" : 30,
    #               "n_colabs" : 5,
    #               "rounds_to_train" :200,
    #               "num_epochs" : 30,
    #               "batch_size" : 60,
    #               "output_data_regime" : "binary",
    #               "num_classes" : 2,
    #               "index_col" :"false",
    #               "bucket_blob_prefix" : "validation/Example2_dim_inv_10_dim_spu_50_dim_unc_0_n_exp_900_n_env_5/"
    #               },

    # 'Example2e': {"dir_ind" : 0,
    #               "seed" : 0,
    #               "repeats": 30,
    #               "n_colabs" : 5,
    #               "rounds_to_train": 200,
    #               "num_epochs" : 30,
    #               "batch_size" : 60,
    #               "output_data_regime" : "binary",
    #               "num_classes" : 2,
    #               "index_col" :"false",
    #               "bucket_blob_prefix" : "validation/Example2_dim_inv_6_dim_spu_6_dim_unc_500_n_exp_300_n_env_5/"
    #               },

    # 'Example2f': {"dir_ind" : 0,
    #               "seed" : 0,
    #               "repeats" :30,
    #               "n_colabs" : 5,
    #               "rounds_to_train" :200,
    #               "num_epochs" : 30,
    #               "batch_size" : 60,
    #               "output_data_regime" : "binary",
    #               "num_classes" : 2,
    #               "index_col" :"false",
    #               "bucket_blob_prefix" : "validation/Example2_dim_inv_6_dim_spu_6_dim_unc_500_n_exp_900_n_env_5/"
    #               },

    # 'Example2g': {"dir_ind" : 0,
    #               "seed" : 0,
    #               "repeats" :30,
    #               "n_colabs" : 5,
    #               "rounds_to_train":200,
    #               "num_epochs" : 30,
    #               "batch_size" : 60,
    #               "output_data_regime" : "binary",
    #               "num_classes" : 2,
    #               "index_col" :"false",
    #               "bucket_blob_prefix" : "validation/Example2_dim_inv_6_dim_spu_6_dim_unc_50_n_exp_300_n_env_5/"
    #               },

    # 'Example2h': {"dir_ind" : 0,
    #               "seed" : 0,
    #               "repeats" : 30,
    #               "n_colabs" : 5,
    #               "rounds_to_train" :200,
    #               "num_epochs" : 30,
    #               "batch_size" : 60,
    #               "output_data_regime" : "binary",
    #               "num_classes" : 2,
    #               "index_col" :"false",
    #               "bucket_blob_prefix" : "validation/Example2_dim_inv_6_dim_spu_6_dim_unc_50_n_exp_300_n_env_5/"
    #               },



    'heavy-ion-reduced': {"dir_ind" : 0,
                          "seed" : 0,
                          "repeats": 10 ,
                          "n_colabs" : 3,
                          "rounds_to_train": 200,
                          "num_epochs" : 1,
                          "batch_size" : 50,
                          "output_data_regime" : "binary",
                          "num_classes" : 2,
                          "index_col" :"true",
                          "bucket_blob_prefix" : "heavy_ion_experiment_reduced_new_code"
                          },

    # 'elijah6': {"dir_ind" : 0,
    #             "seed" : 0,
    #             "repeats": 20 ,
    #             "n_colabs" : 6,
    #             "rounds_to_train": 100,
    #             "num_epochs" : 30,
    #              "batch_size" : 1000,
    #              "output_data_regime" : "real-valued",
    #              "num_classes" : 1,
    #              "index_col" :"true",
    #              "bucket_blob_prefix" : "elijah_mouse_experiment_reduced_new_code_v5"
    #              },

    # 'elijah12': {"dir_ind" : 0,
    #             "seed" : 0,
    #             "repeats": 20 ,
    #             "n_colabs" : 12,
    #             "rounds_to_train": 100,
    #             "num_epochs" : 30,
    #              "batch_size" : 1000,
    #              "output_data_regime" : "real-valued",
    #              "num_classes" : 1,
    #              "index_col" :"true",
    #              "bucket_blob_prefix" : "Final_elijah_mouse_experiment_reduced_12_env_mode"
    #              }

    'gamma-ray': {"dir_ind" : 0,
                          "seed" : 0,
                          "repeats": 10,
                          "n_colabs" : 7,
                          "rounds_to_train": 200,
                          "num_epochs" : 1,
                          "batch_size" : 1000,
                          "output_data_regime" : "binary",
                          "num_classes" : 2,
                          "index_col" :"true",
                          "bucket_blob_prefix" : "final_gamma_ray_experiment_reduced"
                          },

    'elijah6': {"dir_ind" : 0,
                "seed" : 0,
                "repeats": 10,
                "n_colabs" : 6,
                "rounds_to_train": 200,
                "num_epochs" : 1,
                 "batch_size" : 1000,
                 "output_data_regime" : "real-valued",
                 "num_classes" : 1,
                 "index_col" :"true",
                 "bucket_blob_prefix" : "elijah_mouse_experiment_reduced_new_code_v5"
                 },

    'elijah12': {"dir_ind" : 0,
                "seed" : 0,
                "repeats": 10 ,
                "n_colabs" : 12,
                "rounds_to_train": 200,
                "num_epochs" : 1,
                 "batch_size" : 1000,
                 "output_data_regime" : "real-valued",
                 "num_classes" : 1,
                 "index_col" :"true",
                 "bucket_blob_prefix" : "Final_elijah_mouse_experiment_reduced_12_env_mode"
                 }

}

for experiment, case in experiments.items():
    print("\n",experiment)

    os.system("gcloud beta compute --project=fdl-us-astronaut-health instances create-with-container paul-%s-%s \
    --zone=us-central1-a --machine-type=e2-standard-4 --subnet=default --network-tier=PREMIUM --metadata=google-logging-enabled=true \
    --maintenance-policy=MIGRATE --service-account=694584192919-compute@developer.gserviceaccount.com --scopes=https://www.googleapis.com/auth/cloud-platform \
    --image=cos-stable-89-16108-470-11 --image-project=cos-cloud --boot-disk-size=100GB --boot-disk-type=pd-balanced --boot-disk-device-name=paul-multi-runner-%s \
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
    --labels=container-vm=cos-stable-89-16108-470-11" \
    % (instance_name, experiment.lower(), experiment.lower(), case["dir_ind"], case["seed"], case["repeats"], case["n_colabs"], case["rounds_to_train"], case["num_epochs"], case["batch_size"], case["output_data_regime"], case["num_classes"], case["index_col"], case["bucket_blob_prefix"]) 
    )
