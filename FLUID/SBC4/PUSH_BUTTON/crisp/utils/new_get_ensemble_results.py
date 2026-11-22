import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


#weights feature coefficients/sensitivities by the number of models selected, returns as weighted feature coefficient dataframe feature_weight_df
def get_ensemble_results(to_bucket_results, use_feature_gradients = False):
    feat_dicts = []
    method_names = []

    all_features = set()
    for i, method_dict in enumerate(to_bucket_results):
        method = method_dict.get('method', "Non-Causal ERM")
        """
            Structure of method_dict
            "to_bucket": {
                'method': 'Method Name',
                'features': [X1, X2, X4, X32, .. Xmax],
                'coefficients': [w1, w2, w4, w32, .. wmax],
                'pvals': [p1, p2, p4, p32, .. pmax] || p_total || None
                'test_acc': 0.97 || None
            }
            """

        method_names += [method]
        # get feats sorted by highest absolute value; note for nonlinear models these are sensitivities (which should be used on normalized data)
        coefs = pd.DataFrame()
        coefs['feature'] = method_dict['features']
        if use_feature_gradients:
            coefs['coefficient'] = method_dict['feature_gradients']
        else:
            try:
                coefs['coefficient'] = method_dict['coefficients']
            except:
                coefs['coefficient'] = method_dict['coefficients'][0]
                

        print(method,method_dict['test_acc'])
        coefs['sort'] = method_dict['test_acc'] * coefs['coefficient'].abs()/max(coefs['coefficient'].abs())

        coefs = coefs.sort_values('sort', ascending=False)
        feat_dicts.append(coefs)
        # update all_features with all features seen across all models
        all_features.update(list(coefs['feature'].values))

    # create df with coefficients each method, and the 'selected' parameter which is proportional to the number of
    # models that feature intersected with
    feature_weight_df = pd.DataFrame(columns=method_names, index=all_features)
    for i, feat_df in enumerate(feat_dicts):
        method = method_names[i]
        feature_weight_df[method][feat_df['feature']] = feat_df['coefficient'].abs().values

    feature_weight_df = feature_weight_df.fillna(0)

    # count number of models feature is chosen by
    feature_weight_df['count_models'] = (feature_weight_df != 0).sum(axis=1)
    # calc proportion of models feature is chosen by
    method_count = len(method_names)
    feature_weight_df['selected'] = feature_weight_df['count_models'] / method_count
    feature_weight_df['weighted_coefficient'] = feature_weight_df[method_names].mean(axis=1) * feature_weight_df[
        'selected']
    feature_weight_df['feature'] = feature_weight_df.index
    feature_weight_df.replace([-np.inf], 0, inplace=True)

    return feature_weight_df

