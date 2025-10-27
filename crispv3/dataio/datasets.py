import numpy as np
import pandas as pd


from dataio.DataFrameDataset import DataFrameDataset, subjID_sensitive_datasplit, subjID_sensitive_datasplit_balanced

def get_datasets(dataframe_fp, subject_keys, test_val_split, predictors, targets, environments, exclude, remove_keys):
    """
    Function to get list of training environments, validation and test split on Subject ID key
    :param dataframe_fp: filepath to dataframe to use
    :param subject_keys: key for column based train, val, test split
    :param predictors: column names for columns to be used as predictors
    :param targets: column names for columns to be used as targets
    :param environments: column name for column to split training data into 'environments'
    
    returns
    list of training environment pytorch datasets, val pytorch dataset, test pytorch dataset
    """
    # Load data frame,
    dataframe = pd.read_pickle(dataframe_fp)
    # Split into train and test by subject
    test_perc, val_perc = test_val_split[0], test_val_split[1]

    # If predicting a single binary output: use balanced test set, else random (balanced not implemented for multiclass or multilabel)
    if len(targets) == 1:
        train, test = subjID_sensitive_datasplit_balanced(dataframe, test_size=test_perc, key=subject_keys, target=targets)
        # Split train into into train and val by subject
        train, validation = subjID_sensitive_datasplit_balanced(train, test_size=val_perc, key=subject_keys, target=targets)
    else:
        train, test = subjID_sensitive_datasplit(dataframe, test_size=test_perc, key=subject_keys)
        # Split train into into train and val by subject
        train, validation = subjID_sensitive_datasplit(train, test_size=val_perc, key=subject_keys)
    # Split train into environments
    grouped = train.groupby(environments)
    
    # Check if only certain features are used, or 'All'
    if predictors == 'All':
        predictors = list(train.columns)
        for re in remove_keys:
            predictors.remove(re)
        for e in environments:
            predictors.remove(e)
        for tg in targets:
            predictors.remove(tg)
    
    # Create a DataFrameDataset for each training environment split
    train_envs = []
    for val in np.unique(train[environments].values):
        env_ds = DataFrameDataset(dataframe=grouped.get_group(val), predictor_columns=predictors, target_columns=targets, exclude_columns=exclude)
        train_envs.append(env_ds)
    
    # Create a DataFrameDataset for test and val splits
    val_dataset = DataFrameDataset(dataframe=validation, predictor_columns=predictors, target_columns=targets, exclude_columns=exclude)
    test_dataset = DataFrameDataset(dataframe=test, predictor_columns=predictors, target_columns=targets, exclude_columns=exclude)
    
    return train_envs, val_dataset, test_dataset


def get_datasets_for_experiment(config):
    """
        Function to get train/test/val pytorch datasets
        :param config: dictionary passed from main

        returns
        list of training environment pytorch datasets, val pytorch dataset, test pytorch dataset (each is a DataFrameDataset as defined in DataFrameDataset.py)

        """
    # Load Datasets
    # Per subject experiment
    if config['verbose']:
        print('Running a per sample experiment')
    data_config = config['data_options']
    environment_datasets, val_dataset, test_dataset = get_datasets(
        dataframe_fp=data_config['dataset_fp'],
        subject_keys=data_config['subject_keys'],
        test_val_split=config['test_val_split'],
        predictors=data_config['predictors'],
        targets=data_config['targets'],
        environments=data_config['environments'],
        exclude=data_config.get('exclude', []),
        remove_keys=data_config.get('remove_keys', [])
    )
    if config['verbose']:
        print('Loaded ', len(environment_datasets), ' train environments')
        for i, e in enumerate(environment_datasets):
            print('Env ', str(i), ' has ', e.__len__(), ' samples')
            print('X shape ', e.get_shape()[0], ' y shape ', e.get_shape()[1])
        print('Loaded val set, X shape:', val_dataset.get_shape()[0], ' y shape: ', val_dataset.get_shape()[1])
        print('Loaded test set, X shape:', test_dataset.get_shape()[0], ' y shape: ', test_dataset.get_shape()[1])
        
    return environment_datasets, val_dataset, test_dataset
