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

## Installation

Ensure you have a working version of 3.7 (Not currently compatible with 3.8+). We recommend using Python from the [Anaconda Python distribution](https://www.continuum.io/downloads) for a quicker and more reliable experience. However, if you have Python already installed that will probably work fine too.

```sh
git clone https://gitlab.com/frontierdevelopmentlab/astronaut-health/crisp
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

### 2. Create Experiment configuration json file

We provide a template configuration json in /experiment_configs/ that should be updated to provide a unique experiment name, and path to the data file to be used (.pickle). Specify the target variable column name, custom choice of predictor variables (or 'All' to use everything), choice of 'enviroment' split variable. Please provide a 'subject_keys' variable that will be used to split data into train/val/test sets on with samples with matching 'subject_keys' variable only appearing in one of train/val/test.

```json
{
    "name": "Example Experiment for AH casual ensemble",
    "short_name": "<unique_experiment_name>",
    "bucket_project": "<bucket_project>",
    "bucket_name": "<bucket_name>",
    "bucket_path": "<bucket_path>",
    "verbose": 1,
    "test_val_split": [0.1, 0.1],
    "per_variant_experiment": false,
    "data_options": {
        "dataset_fp": "<path_to_dataset>",
        "subject_keys": "<subject_key>",
        "targets": ["<target_variable_column_name>"],
        "predictors": "All",
        "environments": ["<environment_split>"],
        "exclude": ["<keys_to_exclude>"],
        "remove_keys": ["<subject_key>", "<any_others_to_remove>"],
        "merge_keys": ["<not_used_here>"]
    },
    "feature_selection_options": {
        "max_features": 20,
        "verbose": 0,
        "seed": 12
    },
    "ensemble_options": {
        "models": ["ERM", "RF", "ICP", "IRM", "DCF", "ITE", "LIRM", "NLICP"]
    }
}
```

Custom model parameters can be overriden in the config file -- see template configuration json.

### 3. Train CRISP ensemble of models

```sh
python main.py --experiment_config experiment_configs/<path_to_config>
```

### 4. Visualise results using streamlit frontend

1. Add path to experiment configuration file to 'streamlit_main.py'

2. Run streamlit

```sh
streamlit run streamlit_main.py
```


### Custom Data
Users are free to use CRISP on their own data but should take care to perform suitable data processing as described in the paper. We have performed validation on synthetic datasets that may or may not be a suitable fit for custom data, and as such results are not guaranteed with custom data.

Custom data should be normalised for continuous data, one-hot encoded for categorical data and target variables should be binarised. The variable chosen for 'environment' splitting is best suited to binary but does support categorical data, however from emperical results we have found that two environments is sufficient.

### Model Zoo
Several methods (IRM, Linear IRM, ICP, Non-Linear ICP) rely on PyTorch models defined in 'models/TorchModelZoo.py', and are selected within each method by a field 'method' passed in the config to each method's configuration object. At present this defaults to the MLP defined in TorchModelZoo, but can be replaced with any model specified by user, and the user is free to replace the MLP class with whichever architecutre they wish, or add additional options themselves (this will require adding to the init function of each method that uses models defined in the TorchModelZoo.py)

### References
We would like to thank the following for their implementations that guided ours:

- Deconfounder: https://github.com/blei-lab/deconfounder_tutorial 
- Invariant Causal Prediction: https://github.com/facebookresearch/InvariantRiskMinimization/blob/master/code/experiment_synthetic/models.py
- Non-Linear Invariant Causal Prediction: https://github.com/christinaheinze/nonlinearICP-and-CondIndTests
- Invariant Risk Minimisation: https://github.com/facebookresearch/InvariantRiskMinimization
- Linear Invariant Risk Minimisation: