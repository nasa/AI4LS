<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://gitlab.com/frontierdevelopmentlab/astronaut-health/crisp">
    <img src="streamlit_frontend/ah_streamlit_banner.png" alt="Logo">
  </a>
</p>

# CRISP: Causal Relation and Inference Search Platform

NASA FDL 2020 Astronaut Health Team 

This tool is a prototype preview for generation of synthetic causal SEM data with properties common to cancer and biology data, and data agnostic causal discovery with heterogenous data using CRISP: The Causal Relation and Inference Search Platform.

CRISP is an ensemble of causal inference methods and interactive results visualisation dashboard for discovery of causal relationships in complex biological data.

At present CRISP is suitable for binary classification of tabulated data, with continous, binary or categorical (one-hot encoded) variables.


## NOTE
This repository was forked from the NASA FDL 2020 Astronaut Health Team code repository.  The changes made to this code base are described at the end of this document.

## Installation

Ensure you have a working version of Python 3.7.  CRISP v1.1 is not currently compatible with 3.8+). We recommend using Python from the [Anaconda Python distribution](https://www.continuum.io/downloads) for a quicker and more reliable experience. However, if you have Python already installed that will probably work fine too.

```sh
git clone https://github.com/nasa/AI4LS 
```
 
```sh
pip install -r requirements.txt
```

## Usage

### 1. Generate Synthetic Data

We provide tooling to generate synthetic SEM datasets that mirror properties of complex biological data.

```sh
python setup_synthetic.py
```

### 2. Create experiment configuration JSON file

We provide a template configuration JSON in /experiment_configs/ that should be updated to provide a unique experiment name, and path to the data file (in python pickle format) to be used. Specify the target variable column name, custom choice of predictor variables (or 'All' to use everything), choice of 'enviroment' split variable. Please provide a 'subject_keys' variable that will be used to split data into train/val/test sets on with samples with matching 'subject_keys' variable only appearing in one of train/val/test.

```json
{
    "name": "Example synthetic",
    "short_name": "example",
    "verbose": 1,
    "test_val_split": [0.1, 0.1],
    "data_options": {
        "dataset_fp": "data/synthetic/full_fw_synthetic_sem_n_causal_5_0.pickle",
        "subject_keys": "Subj_ID",
        "targets": ["Target"],
        "predictors": "All",
        "environments": ["env_split"],
        "exclude": ["Subj_ID"],
	"output_data_regime": "binary"
    },
    "feature_selection_options": {
        "max_features": 20,
        "verbose": 0,
        "seed": 12
    },
    "ensemble_options": {
        "models": ["ERM", "RF", "ICP", "IRM", "LIRM", "NLICP"]
    }
}
```

Custom model parameters can be overriden in the config file -- see template configuration json.

### 3. Train CRISP ensemble of models

```sh
python main.py --experiment_config experiment_configs/synthetic_example.json
```

### 4. Visualise results using streamlit frontend

1. Add path to experiment configuration file to 'streamlit_main.py'

2. Run streamlit

```sh
streamlit run streamlit_main.py
```


## Custom Data
Users are free to use CRISP on their own data but should take care to perform suitable data processing as described in the paper. We have performed validation on synthetic datasets that may or may not be a suitable fit for custom data, and as such results are not guaranteed with custom data.

Custom data should be normalised for continuous data, one-hot encoded for categorical data and target variables should be binarised. The variable chosen for 'environment' splitting is best suited to binary but does support categorical data, however from emperical results we have found that two environments is sufficient.

## Resource Requirements
CRISP is CPU-bound and supports running some of the tensorflow routines on either GPU or CPU.  RAM and disk storage requirements are minimal.  The synthetic example can easily run on a laptop with 8GB of RAM and 1 CPU core.  Running all CRISP ensemble models on a transcriptomic dataset of 100 samples and 10,000 genes takes roughly 3 hours on a system with 8 GPUs.  On a system with 64 CPU cores, the same workload takes roughly 16 hours to finish.

## Model Zoo
Several methods (IRM, Linear IRM, ICP, Non-Linear ICP) rely on PyTorch models defined in 'models/TorchModelZoo.py', and are selected within each method by a field 'method' passed in the config to each method's configuration object. At present this defaults to the MLP defined in TorchModelZoo, but can be replaced with any model specified by user, and the user is free to replace the MLP class with whichever architecutre they wish, or add additional options themselves (this will require adding to the init function of each method that uses models defined in the TorchModelZoo.py)

## References
We would like to thank the following for their implementations that guided ours:

- Deconfounder: https://github.com/blei-lab/deconfounder_tutorial 
- Invariant Causal Prediction: https://github.com/facebookresearch/InvariantRiskMinimization/blob/master/code/experiment_synthetic/models.py
- Non-Linear Invariant Causal Prediction: https://github.com/christinaheinze/nonlinearICP-and-CondIndTests
- Invariant Risk Minimisation: https://github.com/facebookresearch/InvariantRiskMinimization
- Linear Invariant Risk Minimisation:


## Updates included in this release:
1. Change the weights of the coefficients of the features of each ensemble model to be on the same scale.  Here we use the [MinMaxScaler method from scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html) package to scale weights. 

2. Change the feature coefficients of the Linear IRM module to be on the unit interval [0, 1].

3. Add the [CausalNex](https://causalnex.readthedocs.io/en/latest/) module to the learning ensemble and use it for feature reduction.

