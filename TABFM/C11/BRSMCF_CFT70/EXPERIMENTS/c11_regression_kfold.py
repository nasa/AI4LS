import os
import sys
os.environ["SCIPY_ARRAY_API"] = "1"

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.inspection import permutation_importance

from tabpfn_client import TabPFNRegressor
from tabpfn_client import set_access_token

import pandas as pd
import numpy as np
from collections import Counter

# set access
set_access_token("tabpfn_sk_4i2HFotq_9wpdLTdfT5UndsQPCF7QWzfchuddvYiRtA")

# open file for output
f = open(sys.argv[2], "w")

df = pd.read_csv(sys.argv[1], sep=',', header=0)

path_name = sys.argv[1].split('.csv')[0]
colname = os.path.basename(path_name)

X = df.drop(columns=[colname])
y = np.array(list(df[colname]))
X_array = np.array(X)

# ---- k-fold setup ----
#N_SPLITS = 5
N_SPLITS=int(sys.argv[3])
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

rf_r2_scores = []
tab_r2_scores = []
rf_top_features_per_fold = []
tab_top_features_per_fold = []

for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_array)):
    X_train, X_test = X_array[train_idx], X_array[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # ---- Random Forest ----
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    rf_r2 = r2_score(y_test, y_pred_rf)
    rf_r2_scores.append(rf_r2)

    importances = rf.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    top_3_rf_features = list(feature_importance_df['Feature'])[:3]
    rf_top_features_per_fold.append(top_3_rf_features)

    # ---- TabPFN ----
    reg = TabPFNRegressor(n_estimators=1)
    reg.fit(X_train, y_train)
    y_pred_tab = reg.predict(X_test)
    tab_r2 = r2_score(y_test, y_pred_tab)
    tab_r2_scores.append(tab_r2)

    tab_pfi = permutation_importance(reg, X_test, y_test, scoring='r2', n_repeats=1, random_state=0)
    features = list(X.columns)
    tab_fi_list = [features[i] for i in tab_pfi.importances_mean.argsort()[::-1][:3]]
    tab_top_features_per_fold.append(tab_fi_list)

    # response_variable,randomforest_r2,randomforest_features,tabpfn_r2,tabpfn_features

    fold_line = (f"Fold {fold_idx+1}/{N_SPLITS}: "
                 f"RF R2={rf_r2:.4f}, TabPFN R2={tab_r2:.4f}, "
                 f"RF top3={top_3_rf_features}, TabPFN top3={tab_fi_list}")
    print(fold_line)
    f.write(fold_line + '\n')
    f.flush()

# ---- aggregate across folds ----
rf_r2_mean, rf_r2_std = np.mean(rf_r2_scores), np.std(rf_r2_scores)
tab_r2_mean, tab_r2_std = np.mean(tab_r2_scores), np.std(tab_r2_scores)

# most frequently selected top features across folds, as a simple summary
def summarize_top_features(list_of_lists, k=3):
    counter = Counter()
    for flist in list_of_lists:
        for feat in flist:
            counter[feat] += 1
    return [feat for feat, _ in counter.most_common(k)]

rf_overall_top = summarize_top_features(rf_top_features_per_fold)
tab_overall_top = summarize_top_features(tab_top_features_per_fold)

# response_variable,randomforest_r2,randomforest_features,tabpfn_r2,tabpfn_features
# post2:WKLD_@_VO2PK,0.5084084191757556,['pre:SBPseated', 'pre:VO2PK', 'pre:WKLD_@_VO2PK'],0.5182547344292523,['pre:WKLD_@_VO2PK', 'pre:VO2PK', 'pre:VEPK']

# Fold 1/5: RF R2=0.0002, TabPFN R2=0.3567, RF top3=['PRE:2_3:LVDV', 'PRE:2_3:LVSV', 'PRE:2_3:LV_mass'], TabPFN top3=['Treatment', 'PRE:2_3:LVDV', 'PRE:2_3:LVSV']
summary_row = (colname + '\t' +
               str(rf_r2_mean)  + '\t' + \
		str(rf_overall_top) + '\t' + \
               	str(tab_r2_mean)  + '\t' + \
		str(tab_overall_top))

print(summary_row)
f.write(summary_row + '\n')
f.flush()

f.close()
