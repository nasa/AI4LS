from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

import torch
import pandas as pd

import numpy as np

from Ensembl_converter import EnsemblConverter

from sklearn.inspection import permutation_importance

def make_importance_getter(X, y):
    def tabpfn_importance_getter(estimator):
        result = permutation_importance(estimator, X, y, n_repeats=5, random_state=42)
        return result.importances_mean
    return tabpfn_importance_getter



def get_symbol_from_id(gene_id_list):
  # Create an instance of EnsemblConverter
  converter = EnsemblConverter()

  # Convert Ensembl IDs to gene symbols
  result = converter.convert_ids(gene_id_list)

  # Print the resulting DataFrame
  gene_symbol_list = list()
  for i in range(len(result)):
    gene_symbol_list.append(result.iloc[i]['Symbol'])

  return gene_symbol_list

def full_transform(X, x_list):
  from sklearn.preprocessing import StandardScaler
  from sklearn.preprocessing import power_transform
  import numpy as np
  from rnanorm import TPM

  temp = X
  if 'tpm' in x_list:
    # TPM
    print('shape of df before tpm: ', temp.shape)
    # Provide a GTF annotation file (gene lengths will be computed)
    gtf_filename = './Mus_musculus.GRCm39.115.gtf.gz'
    tpm_calculator = TPM(gtf=gtf_filename)

    # Set output format (e.g., pandas DataFrame)
    tpm_calculator.set_output(transform="pandas")

    # Transform raw counts to TPM
    temp_tpm = tpm_calculator.fit_transform(temp)
    temp = temp_tpm

  if 'log' in x_list:
    # LOG
    # assumes genes x samples
    print('shape of df before log: ', temp.shape)
    temp_t = temp.T
    temp_log = np.log2(temp_t + 1)
    temp = temp_log.T

  if 'std' in x_list:
    # STD
    # assumes samples x genes
    print('shape of df before scaling: ', temp.shape)
    #scaler = StandardScaler()
    #scaled = scaler.fit_transform(temp)
    scaled = (temp - np.mean(temp, axis=0)) + 0.01 / (np.std(temp, axis=0) + 0.01)
    temp = scaled


  if 'power' in x_list:
    # assumes samples x genes
    temp_power = np.zeros((temp.shape[0], temp.shape[1]))
    for i in range(X.shape[1]):
      temp_power[:, i] = power_transform(temp[:, i].reshape(-1, 1), method='yeo-johnson', standardize=True).reshape(-1)
    temp = temp_power

  # drop nans
  #return temp.T.dropna(axis=1, how='any').T
  return temp



# Load data
#X, y = load_breast_cancer(return_X_y=True)

df=pd.read_csv('expr_with_condition.csv', sep=',', header=0)

X=df.drop(columns=['sample', 'condition'])
y=np.array(list(df['condition']))


X_trans = full_transform(X, ['tpm', 'log', 'std'])
X_array = np.array(X_trans)

print('y labels: ', y[:10])
X_train, X_test, y_train, y_test = train_test_split(X_array, y, test_size=0.5, random_state=42)

# Initialize a classifier
clf = TabPFNClassifier()  # Uses TabPFN 2.5 weights, finetuned on real data.

# persist model to disk
torch.save(clf, 'tabpfn-2.5.pt')

# To use TabPFN v2:
# clf = TabPFNClassifier.create_default_for_version(ModelVersion.V2)
clf.fit(X_train, y_train)


# Predict probabilities
prediction_probabilities = clf.predict_proba(X_test)
print("ROC AUC:", roc_auc_score(y_test, prediction_probabilities[:, 1]))

# Predict labels
predictions = clf.predict(X_test)
print("Accuracy", accuracy_score(y_test, predictions))

# get most predictive genes
n_genes=20

# RFE
from sklearn.feature_selection import RFE
#selector = RFE(clf, n_features_to_select=n_genes, step=0.25).fit(X_array, y)


selector = RFE(clf, n_features_to_select=n_genes, step=0.25, importance_getter=make_importance_getter(X_array, y)).fit(X_array, y)

indices = selector.get_support(indices=True)
rfe_genes = get_symbol_from_id([list(X.columns)[i] for i in indices])
print('rfe genes: ', rfe_genes)

# PFI 
pfi_genes = get_symbol_from_id(permutation_feature_importance(clf, \
            X_array, y, genes = list(X.columns), scoring=classification_metric, n=n_genes))
print('pfi genes: ', pfi_genes)




