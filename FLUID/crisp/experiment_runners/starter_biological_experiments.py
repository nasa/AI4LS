import os
examples = ['final_gamma_ray_experiment_reduced','final_elijah_mouse_experiment_reduced','final_heavy_ion_experiment_reduced']
methods = ['ERM', 'CSNX', 'IRM', 'SNLIRM']
data_dir = '/home/user/crisp/data'
for example in examples:
    
    if 'elijah' in example:
        n_iterations = 4500
    else:
        n_iterations = 3000
        
    
    for method in methods:
        os.system("gcloud beta compute --project=fdl-us-astronaut-health instances create-with-container giuseppe-bio-val-%s-%s --zone=us-central1-a --machine-type=e2-standard-4 --subnet=default --network-tier=PREMIUM --metadata=google-logging-enabled=true --maintenance-policy=MIGRATE --service-account=694584192919-compute@developer.gserviceaccount.com --scopes=https://www.googleapis.com/auth/cloud-platform --image=cos-stable-89-16108-470-11 --image-project=cos-cloud --boot-disk-size=50GB --boot-disk-type=pd-balanced --boot-disk-device-name=giuseppe-bio-val-%s-%s --no-shielded-secure-boot --shielded-vtpm --shielded-integrity-monitoring --container-image=gcr.io/fdl-us-astronaut-health/ah-crisp-validate --container-restart-policy=on-failure --container-privileged --container-stdin --container-tty --container-command=python --container-arg=/home/user/crisp/utils/experiment_runner_biological_data.py --container-arg=--method=%s --container-arg=--example=%s --container-arg=--n_iterations=%i  --container-arg=--data_dir=%s --labels=container-vm=cos-stable-89-16108-470-11"%(method.lower(),example.lower().replace("_", "-")[:8], method.lower(), example.lower().replace("_", "-")[:8], method, example, n_iterations, data_dir))
