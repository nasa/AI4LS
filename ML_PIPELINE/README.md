
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
python run_docker_pipeline.py --operation=download --osd_id=48 --target_column='Factor Value[Spaceflight]'  --task_type=classification --algorithm=random_forest --test_size=0.2 --trans_list=n,s,l,t --fi_methods=built_in,rfe -pv 0.9 -qv 0.9 
```
4. Check the results directory for the CSV and PNG files 
 
```console
ls -R results/
```

5. Run the `run_docker_pipeline.py` script against the data in OSD-137 to do classification using the random_forest algorithm.

```console
python run_docker_pipeline.py --operation=download --osd_id=137 --target_column='Factor Value[Spaceflight]'  --task_type=classification --algorithm=random_forest --test_size=0.2 --trans_list=l,t --fi_methods=built_in,pfi,rfe --patterns=unnormalized -pv 0.9 -qv 0.9

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

3. Run the `run_docker_pipeline.py` script to upload a CSV file and do classification using the random_forest algorithm.

```console
python run_docker_pipeline.py   -op upload   -tt classification   -al random_forest -if DATA/X_hne_class_nosample_100.csv -tc "Factor Value[Spaceflight]" -ec sample -sc sample -pv 0.9 -qv 0.9 -fi built_in 
```

4. Check the results directory for the CSV and PNG files 

```console
ls -R results/
```
