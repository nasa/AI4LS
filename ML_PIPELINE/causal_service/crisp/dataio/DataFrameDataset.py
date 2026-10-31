import torch
import torch.utils.data as data
import numpy as np

from itertools import cycle, islice, chain
from utils.gcp_helpers import get_dataframe_from_bucket

global ALL_COLUMNS
ALL_COLUMNS = None

class DataFrameDataset(data.Dataset):
    """
    Generic Pytorch dataset:
    :param dataframe: pandas dataframe to be used
    :param predictor_columns: column names to be used as input
    :param target_columns: column name(s) to be used as target
    """
    def __init__(self, dataframe, predictor_columns, target_columns, exclude_columns):
        super(DataFrameDataset, self).__init__()
        self.predictor_columns = [p for p in predictor_columns if p not in exclude_columns]
        self.target_columns = target_columns
        self.data = dataframe[self.predictor_columns].values.astype(float)
        self.targets = dataframe[self.target_columns].values.astype(float)
        
    def __len__(self):
        return self.data.shape[0]
    
    def get_shape(self):
        return self.data.shape, self.targets.shape
    
    def get_feature_dim(self):
        return len(self.predictor_columns)
    
    def get_output_dim(self):
        return len(self.target_columns)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx, :]).float(), torch.from_numpy(self.targets[idx, :]).float()
    
    def get_all(self):
        return torch.from_numpy(self.data).float(), torch.from_numpy(self.targets).float()

    

def subjID_sensitive_datasplit(standardised_df, test_size = 0.25, random_state=20, key = 'Subj_ID', printing = False):
    """
    Takes a dataframe and randomly splits it into train and test set under the restriction that all unique values
    of column specified in key (default: Subj_ID) are on either side of split. E.g. all samples with Subj_ID 1013 
    land in either train or test set.

    :param stadardised_df: pandas dataframe to be split.
    :param test_size: (optional) Percentage of data (in samples/rows of df, not subjID) to be assigned to test data set. 
        If not specified, default value is used. 
    :param random_state: (optional) Integer value that numpy random seed gets set to when calling the function. If not 
        specified,default value is used. 
    :param key: (optional) Column name of feature for which each unique value needs to be entirely in either train 
        or test set. Defaults to Subj_ID.
    :param printing: (optional) If True, function call prints resulting train/test split percentages on key level and
        sample level. Default False.
        
    :return: pandas dataframe containing train data, pandas dataframe containing test data
    """
    
    np.random.seed(random_state)
    n = standardised_df.shape[0]
    n_test = int(test_size * n)
    keydict = standardised_df[key].value_counts()
    m = len(keydict)
    ID_list_idcs = np.random.permutation(m)
    chosen_IDs = []
    
    count = 0
    included_samples = 0
    
    while included_samples < n_test:
        curr_ID = keydict.axes[0][ID_list_idcs[count]]
        chosen_IDs.append(curr_ID)
        curr_ID_samples = keydict[curr_ID]
    
        count += 1
        included_samples += curr_ID_samples
        
    if printing:
        print('Effective test set ratio on sample level:', included_samples / n)
        print('Resulting test set ratio on subjectID level:', len(chosen_IDs) / m)
    train = standardised_df.loc[~standardised_df[key].isin(chosen_IDs)]
    test = standardised_df.loc[standardised_df[key].isin(chosen_IDs)]
    return train, test


def subjID_sensitive_datasplit_balanced(standardised_df, test_size = 0.25, random_state=20, key = 'Subj_ID', target = 'PMMR_DMMR', printing = False):
    """
    Takes a dataframe and randomly splits it into train and test set under the restriction that all unique values
    of column specified in key (default: Subj_ID) are on either side of split. E.g. all samples with Subj_ID 1013 
    land in either train or test set.

    :param stadardised_df: pandas dataframe to be split.
    :param test_size: (optional) Percentage of data (in samples/rows of df, not subjID) to be assigned to test data set. 
        If not specified, default value is used. 
    :param random_state: (optional) Integer value that numpy random seed gets set to when calling the function. If not 
        specified,default value is used. 
    :param key: (optional) Column name of feature for which each unique value needs to be entirely in either train 
        or test set. Defaults to Subj_ID.
    :param printing: (optional) If True, function call prints resulting train/test split percentages on key level and
        sample level. Default False.
        
    :return: pandas dataframe containing train data, pandas dataframe containing test data
    """
    
    np.random.seed(random_state)
    n = standardised_df.shape[0]
    n_test = int(test_size * n)
    keydict = standardised_df[key].value_counts()
    m = len(keydict)
    ID_list_idcs = np.random.permutation(m)
    chosen_IDs = []
    
    count = 0
    included_samples = 0
    
    target_vals = np.unique(standardised_df[target].values)
    target_val_counts = {}
    for val in target_vals:
        target_val_counts[val] = 0
    
    while included_samples < n_test: 
        curr_ID = keydict.axes[0][ID_list_idcs[count]]
        
        cur_target_vals = standardised_df[standardised_df[key].isin([curr_ID])][target].values
        cur_target_val_count = len(cur_target_vals)
        cur_target_val = cur_target_vals[0][0]
        cur_total_target_val_count = target_val_counts[cur_target_val]
        skip = False
        for val in target_vals:
            if val != cur_target_val and target_val_counts[val] < cur_total_target_val_count:
                skip = True
                
        if skip:
            if count < len(ID_list_idcs)-1:
                count += 1
            else:
                count = 0
        else:
            if count < len(ID_list_idcs)-1:
                count += 1
            else:
                count = 0
            chosen_IDs.append(curr_ID)
            curr_ID_samples = keydict[curr_ID]
            
            target_val_counts[cur_target_val] += cur_target_val_count
            included_samples += curr_ID_samples
    
    
        
    if printing:
        print("Test set value split", target_val_counts)
        print('Effective test set ratio on sample level:', included_samples / n)
        print('Resulting test set ratio on subjectID level:', len(chosen_IDs) / m)
    train = standardised_df.loc[~standardised_df[key].isin(chosen_IDs)]
    test = standardised_df.loc[standardised_df[key].isin(chosen_IDs)]
    return train, test