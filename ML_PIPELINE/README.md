
# Summary 
This software represents a dockerized microservice implementation of a complete pipeline that may be used to leverage machine learning for analyzing transcriptomic data from OSDR.

## Clone the repo.
1. Clone this github repository to your local system. 

```console
git clone https://github.com/nasa/AI4LS
```

2. Change directory to the `AI4LS` directory. 

```console
cd AI4LS 
```

3. Checkout the `mlpipe` branch.

```console
git checkout mlpipe
```

4. Create a conda environment with Python version 3.10.

```console
conda create -n mlpipe python=3.10 -c conda-forge --override-channels
```

5. Activate the environment. 

```console
conda activate mlpipe 
```

6. Install the Python requirements. 

```console
pip install -r requirements.txt
```

## Install docker (if not already installed)

For Mac users, follow [these steps](https://docs.docker.com/desktop/setup/install/mac-install/).

For Windows users, follow [these steps](https://docs.docker.com/desktop/setup/install/windows-install/)

For Linux users, follow [these steps](https://docs.docker.com/desktop/setup/install/linux/)

Make sure the Docker Desktop service is running 

```console
docker image list
```

## Create the microservice docker containers

1. Change directory to the `ML_PIPELINE` directory.

```console
cd ML_PIPELINE
```

2. Build the microservice containers

```console
docker-compose build
```

3. Verify that the images are built.

```console
docker image list
```

4. Start the microservice containers in daemon mode.

```console
docker-compose up -d
```

5. Verify the containers are running.

```console
docker container list
```

6. View the log files from the containers.

```console
docker-compose logs -f
```

## Run a classification algorithm against OSD-48 

1. Open another terminal window.

2. Change directory to the ML_PIPELINE directory

```console
cd AI4LS/ML_PIPELINE
```
3. Run the `run_docker_pipeline.py` script against the data in OSD-48 to do classification using the random_forest algorithm.

```console
python run_docker_pipeline.py --operation=download --osd_id=48 --target_column='Factor Value[Spaceflight]'  --task_type=classification --algorithm=random_forest --test_size=0.2 --trans_list=t,s --fi_methods=built_in,rfe -pv 0.05 -qv 0.05 -fc 1 --dgea=True
```
4. Check the results directory for the CSV and PNG files 
 
```console
ls -R results/
```

5. Run the `run_docker_pipeline.py` script against the data in OSD-137 to do classification using the logistic_regression algorithm.

```console
python run_docker_pipeline.py --operation=download --osd_id=137 --target_column='Factor Value[Spaceflight]'  --task_type=classification --algorithm=logistic_regression --test_size=0.2 --trans_list=l,t --fi_methods=pfi,rfe --patterns=unnormalized -pv 0.05 -qv 0.05 --dgea=True

6. Check the results directory for the CSV and PNG files 
 
```console
ls -R results/
```


## Run a classification algorithm against an uploaded data file. 

1. Open another terminal window.

2. Change directory to the ML_PIPELINE directory

```console
cd AI4LS/ML_PIPELINE
```

3. Run the `run_docker_pipeline.py` script to upload a CSV file and do classification using the neural_network algorithm.

```console
python run_docker_pipeline.py   -op upload   -tt classification   -al neural_network -if DATA/X_hne_class_nosample_100.csv -tc "Factor Value[Spaceflight]" -ec sample -sc sample -pv 0.05 -qv 0.05 -fi rfe --dgea=True 
```

4. Check the results directory for the CSV and PNG files 

```console
ls -R results/
```

# Manage artifacts

## List all artifacts

```console
python utils/cleanup.py list experiments
python utils/cleanup.py list models
python utils/cleanup.py list datasets
```

## Preview deletions (dry-run)

```console
python utils/cleanup.py delete-experiment exp_abc123 --dry-run
python utils/cleanup.py delete-model model_f4d97fe38159 --dry-run
python utils/cleanup.py delete-dataset 65a3ddc6-b4e8 --dry-run
```

## Actually delete

```console
python utils/cleanup.py delete-experiment exp_abc123
python utils/cleanup.py delete-model model_f4d97fe38159
python utils/cleanup.py delete-dataset 65a3ddc6-b4e8
python utils/cleanup.py delete-importance model_f4d97fe38159
python utils/cleanup.py delete-kegg --analysis-id model_61abe71dee68
```

# multi-dataset experiments
## Combine all muscle tissue datasets
```console
python multi_dataset_pipeline.py --tissue muscle
```

## Other available tissues: bone, liver, kidney, heart, brain, blood, skin, intestine, lung
```console
python multi_dataset_pipeline.py --tissue brain
```

## Combine specific datasets
```console
python multi_dataset_pipeline.py --osd-ids "OSD-48,OSD-51,OSD-71"
```

## Single dataset still works
```console
python multi_dataset_pipeline.py --osd-ids "OSD-48"
```

## Customize the pipeline
```console
python multi_dataset_pipeline.py \
  --tissue muscle \
  --task_type classification \
  --algorithm random_forest \
  --test_size 0.2 \
  --cv_step 0.25 \
  --min_features 1000 \
  --fi_methods "built_in,permutation"
```

## Custom factor values
```console
python multi_dataset_pipeline.py \
  --osd-ids "OSD-48,OSD-51" \
  --factor_name "Factor Value[Spaceflight]" \
  --factor_values "Ground Control,Space Flight" \
  --algorithm logistic_regression
```

# Full pipeline with all steps
python multi_dataset_pipeline_full.py --tissue liver

# With custom transformations
python multi_dataset_pipeline_full.py \
  --tissue liver \
  --trans_list "t,l,s" \
  --cv_step 0.25 \
  --min_features 1000

# Skip transformations (raw filtered data)
python multi_dataset_pipeline_full.py \
  --tissue liver \
  --trans_list ""

# Skip specific steps
python multi_dataset_pipeline_full.py \
  --tissue liver \
  --no-ensemble \
  --no-kegg
