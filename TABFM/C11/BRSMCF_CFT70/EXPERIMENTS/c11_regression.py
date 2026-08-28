import os
import sys
os.environ["SCIPY_ARRAY_API"] = "1"

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
#from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.metrics import r2_score, mean_squared_error


from tabpfn_client import TabPFNRegressor
#from tabpfn_extensions import interpretability

#import torch
import pandas as pd

import numpy as np

#from Ensembl_converter import EnsemblConverter

from sklearn.inspection import permutation_importance



# set access
from tabpfn_client import set_access_token

set_access_token("tabpfn_sk_4i2HFotq_9wpdLTdfT5UndsQPCF7QWzfchuddvYiRtA")

# Load data
#X, y = load_breast_cancer(return_X_y=True)

# open file for output
f = open(sys.argv[2], "w")

df=pd.read_csv(sys.argv[1], sep=',', header=0)

path_name=sys.argv[1].split('.csv')[0]
# name derived from filename may have leading parent dirs. remove those
colname = os.path.basename(path_name)

X=df.drop(columns=[colname])
y=np.array(list(df[colname]))

X_array = np.array(X)

X_train, X_test, y_train, y_test = train_test_split(X_array, y, test_size=0.2, random_state=42)

#X_test=X_test[:2]
#y_test=y_test[:2]


# first try to train a RF
rf = RandomForestRegressor(
    n_estimators=100,      # Number of trees in the forest
    max_depth=10,          # Limits depth to prevent massive file sizes
    min_samples_split=5,   # Minimum samples required to split a node
    random_state=42        # Ensures reproducible results
)

rf.fit(X_train, y_train)

# Predict on test data
y_pred = rf.predict(X_test)

# 6. Evaluate metrics
rf_r2 = r2_score(y_test, y_pred)

# get RF feature importance
importances = rf.feature_importances_

# Organize into a clean DataFrame
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

top_3_rf_features = list(feature_importance_df['Feature'])[:3]

# create new row of output
new_row = colname + '\t' + str(rf_r2) + '\t' + str(top_3_rf_features) + '\t'

# Initialize a classifier
reg = TabPFNRegressor(n_estimators=1)
reg.fit(X_train, y_train)  # downloads checkpoint on first use
y_pred = reg.predict(X_test)

# get r2
tab_r2 = r2_score(y_test, y_pred)

# append score to new row for output
new_row =  new_row + str(tab_r2) + '\t'

# Feature selection
'''feature_names =  list(X.columns)
sfs = interpretability.feature_selection.feature_selection(
    estimator=reg,
    X=X_array,
    y=y,
    n_features_to_select=3,  # How many features to select
    feature_names=feature_names,
    verbose=False,
)'''
from sklearn.inspection import permutation_importance
tab_pfi = permutation_importance(reg, X_test, y_test, scoring='r2',  n_repeats=1, random_state=0)
features = list(X.columns)
tab_fi_list = list()
for i in tab_pfi.importances_mean.argsort()[::-1][:3]:
    tab_fi_list.append(features[i])


# Print selected features
'''tab_fi_list = list()
for feature in sfs.selected_names:
    tab_fi_list.append(feature)'''

new_row += str(tab_fi_list)
print(new_row)
f.write(new_row + '\n')
f.flush()

f.close()
