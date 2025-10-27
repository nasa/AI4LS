import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import statistics
import math


NON_CAUSAL = ['RF', 'Non-Causal ERM']


# takes top_N how big of rank genes want to take (top ten, top twenty), dataframe of data, and output filename fname
# JC which fig is this?
def plot_most_predictive(coefficients, fname, top_N=50):
    coefficients['sort'] = coefficients['coefficient'].abs()
    num_nonzero = int(np.sum(coefficients['sort'] > 0))
    dfsorted = coefficients.sort_values('sort', ascending=False)
    dfsorted['pvals'] = [p if p else 1 for p in dfsorted['pvals']]

    # plot
    total_to_plot = min(top_N, num_nonzero)
    fig, ax = plt.subplots(1, figsize=(50, 30))
    ax.bar(x=dfsorted['feature'][0:total_to_plot].astype(str), height=dfsorted['coefficient'][0:total_to_plot])
    for i, v in enumerate(dfsorted['coefficient'][0:total_to_plot]):
        ax.text(i, v, s=str(dfsorted['pvals'][i]), color='black', va='center', fontweight='bold', fontsize=25)
    plt.xticks(rotation='vertical', fontsize=60)
    plt.yticks(fontsize=40)
    plt.tight_layout()
    plt.savefig(fname)

# plot to get top_N selected features for a single CRISP method
# takes top_N how big of rank genes want to take (top ten, top twenty), dataframe, and method name making plot for 
#returns fig object for streamlit
# JC this is the top 20 features (if any) per method
def get_fig_most_predictive(coefficients, method_name, top_N=50):
    coefficients['sort'] = coefficients['coefficient'].abs()
    num_nonzero = int(np.sum(coefficients['sort'] > 0))
    dfsorted = coefficients.sort_values('sort', ascending=False)
    dfsorted['pvals'] = [p if p else 1 for p in dfsorted['pvals']]

    # top 20 if any, 
    total_to_plot = min(top_N, num_nonzero)
    top_df = pd.DataFrame(
        {'coefficient': dfsorted['coefficient'][:total_to_plot], 'feature': dfsorted['feature'][:total_to_plot]})

    data = [go.Bar(
        x=top_df['feature'],
        y=top_df['coefficient']
    )]

    layout = go.Layout(xaxis=dict(type='category'))
    fig = go.Figure(data=data, layout=layout)
    fig.update_yaxes(automargin=True)
    fig.update_xaxes(automargin=True)
    fig.update_layout(title=f'Top {top_N} features: {method_name}')

    return fig


# plots overall top features for all methods based on weighted_coefficient, saves figure to fname
# JC which fig is this?
def plot_overall_most_predictive(coefficients, fname, top_N=50):
    coefficients['sort'] = coefficients['weighted_coefficient'].abs()
    num_nonzero = int(np.sum(coefficients['sort'] > 0))
    dfsorted = coefficients.sort_values('sort', ascending=False)
    colors = {0.0: 'black', 1.0: 'red', 2.0: 'blue', 3.0: 'green', 4.0: 'orange', 5.0: 'purple', 6.0: 'cyan',
              7.0: 'yellow'}
    dfsorted['color'] = [colors[ix] for ix in dfsorted['count_models']]

    # plot
    import matplotlib.patches as mpatches
    total_to_plot = min(top_N, num_nonzero)
    fig, ax = plt.subplots(1, figsize=(50, 30))
    ax.bar(x=dfsorted['feature'][0:total_to_plot].astype(str), height=dfsorted['weighted_coefficient'][0:total_to_plot],
           color=dfsorted['color'])
    plt.xticks(rotation='vertical', fontsize=60)
    plt.yticks(fontsize=40)
    handles = [mpatches.Patch(color=colors[key], label=str(key) + ' models intersected') for key in colors]
    plt.legend(handles=handles, loc="lower right", prop={'size': 50})
    plt.tight_layout()
    plt.savefig(fname)

# plots overall top features for all methods based on weighted_coefficient, saves figure to fname
#returns fig object for streamlit
# JC this is the top 20 features: ensemble
def get_fig_overall_most_predictive(coefficients, top_N=50):
    coefficients['sort'] = coefficients['weighted_coefficient'].abs()
    num_nonzero = int(np.sum(coefficients['sort'] > 0))
    dfsorted = coefficients.sort_values('sort', ascending=False)
    dfsorted['number of models'] = [str(ix) for ix in dfsorted['count_models']]

    total_to_plot = min(top_N, num_nonzero)

    top_df = pd.DataFrame({'weighted_coefficient': dfsorted['weighted_coefficient'][:total_to_plot],
                           'feature': dfsorted['feature'][:total_to_plot].astype(str),
                           'number of models': dfsorted['number of models'][:total_to_plot]})

    fig = px.bar(top_df, x='feature', y='weighted_coefficient', color='number of models',
                 title=f'Top {top_N} features: Ensemble')

    fig.update_yaxes(automargin=True)
    fig.update_xaxes(automargin=True)
    layout = go.Layout(xaxis=dict(type='category'))
    fig = go.Figure(data=fig, layout=layout)

    return fig

#weights feature coefficients/sensitivities by the number of models selected, returns as weighted feature coefficient dataframe feature_weight_df
def get_ensemble_results(to_bucket_results):
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
        if method not in NON_CAUSAL:
            print('method: ', method)
            method_names += [method]
            # get feats sorted by highest absolute value; note for nonlinear models these are sensitivities (which should be used on normalized data)
            coefs = pd.DataFrame()
            coefs['feature'] = method_dict['features']
            #coefs['coefficient'] = method_dict['coefficients']
            ########################################
            # JC: use maxmin scaler for coefficients             
            scaler = MinMaxScaler(feature_range=(-1,1))
            #scaler = StandardScaler()
            coefs_2d = []
            if method_dict['coefficients'] is None:
                coefs['coefficient'] = method_dict['coefficients']
            elif isinstance(method_dict['coefficients'], float):
                coefs['coefficient'] = [method_dict['coefficients']]
            else:
                for c in method_dict['coefficients']:
                    coefs_2d.append([c])
                my_coefs = scaler.fit_transform(coefs_2d)
                coefs['coefficient'] = my_coefs
            ########################################


            ########################################
            # JC: use coef/stdev


            '''if method_dict['coefficients'] is None:
                coefs['coefficient'] = method_dict['coefficients']
            else:
                if type(method_dict['coefficients']) == list and len(method_dict['coefficients']) == 1:
                    coef_stdev = statistics.pstdev(method_dict['coefficients'][0])
                    coef_mean = statistics.mean(method_dict['coefficients'][0])
                    if coef_stdev != 0:
                        my_coefs = [(n - coef_mean) / coef_stdev if n else 1 for n in method_dict['coefficients'][0]]
                        coefs['coefficient'] = my_coefs
                    else:
                        coefs['coefficient'] = method_dict['coefficients'][0]
                else:
                    coef_stdev = statistics.pstdev(method_dict['coefficients'])
                    coef_mean = statistics.mean(method_dict['coefficients'])
                    if coef_stdev != 0:
                        my_coefs = [(n - coef_mean) / coef_stdev if n else 1 for n in method_dict['coefficients']]
                        coefs['coefficient'] = my_coefs
                    else:
                        coefs['coefficient'] = method_dict['coefficients']
            print('coefs[coefficient]: ', coefs['coefficient'])'''

            ########################################

            ########################################
            # JC: use log
            ''''my_coefs = list()
            for i in range(len(method_dict['coefficients'])):
                x = method_dict['coefficients'][i]
                if x == 0: 
                    my_coefs.append(0)
                else:
                   my_coefs.append(np.sign(x) * math.log(np.abs(x)))
            coefs['coefficient'] = my_coefs'''
            ########################################

            ########################################
            # JC: use coef/max(coefs) 
            '''my_coefs = list()
            theMax = max(np.abs(method_dict['coefficients']))
            for i in range(len(method_dict['coefficients'])):
                x = method_dict['coefficients'][i]
                if x == 0:
                    my_coefs.append(0)
                else:
                    my_coefs.append( x / theMax)
            coefs['coefficient'] = my_coefs'''

            ''' ORIG
            method_names += [method]
            # get feats sorted by highest absolute value; note for nonlinear models these are sensitivities (which should be used on normalized data)
            coefs = pd.DataFrame()
            coefs['feature'] = method_dict['features']
            coefs['coefficient'] = method_dict['coefficients']
            coefs['pvals'] = method_dict['pvals']
            coefs['sort'] = method_dict['test_acc'] * coefs['coefficient'].abs()
            coefs = coefs.sort_values('sort', ascending=False)
            coefs['pvals'] = [p if p else 1 for p in coefs['pvals']]
            feat_dicts.append(coefs)
            # update all_features with all features seen across all models
            all_features.update(list(coefs['feature'].values))
            /ORIG '''

            coefs['pvals'] = method_dict['pvals']
            # JC sorting based on product of model accuracy * feature coefficient
            coefs['sort'] = method_dict['test_acc'] * coefs['coefficient'].abs()
            coefs = coefs.sort_values('sort', ascending=False)
            coefs['pvals'] = [p if p else 1 for p in coefs['pvals']]
            feat_dicts.append(coefs)
            # update all_features with all features seen across all models
            all_features.update(list(coefs['feature'].values))

    # create df with coefficients each method, and the 'selected' parameter which is proportional to the number of
    # models that feature intersected with
    feature_weight_df = pd.DataFrame(columns=method_names, index=all_features)
    for i, feat_df in enumerate(feat_dicts):
        method = method_names[i]
        feature_weight_df[method][feat_df['feature']] = feat_df['coefficient'].values

    feature_weight_df = feature_weight_df.fillna(0)

    # count number of models feature is chosen by
    # JC
    #feature_weight_df['count_models'] = (feature_weight_df != 0).sum(axis=1)
    feature_weight_df['count_models'] = [0 for i in range(len(feature_weight_df))]
    top20_per_method = dict()
    for method_index in range(len(method_names)):
        m = method_names[method_index]
        top20_per_method[m] = list(feat_dicts[method_index][0:20]['feature'])
    for feature_index in range(len(feature_weight_df)):
        feature_weight_df.iloc[feature_index]['count_models'] = 0
        for m in top20_per_method:
            if feature_weight_df.index[feature_index] in top20_per_method[m]:
                feature_weight_df.iloc[feature_index, feature_weight_df.columns.get_loc('count_models')] += 1

    # calc proportion of models feature is chosen by
    method_count = len(method_names)
    feature_weight_df['selected'] = feature_weight_df['count_models'] / method_count
    feature_weight_df['weighted_coefficient'] = feature_weight_df[method_names].mean(axis=1) * feature_weight_df['selected']
    feature_weight_df['feature'] = feature_weight_df.index
    feature_weight_df.replace([-np.inf], 0, inplace=True)

    return feature_weight_df


# returns figure of method accuracies to streamlit
# JC: accuracy plot
def get_test_accuracy_figure(results_dict):
    model_names = []
    model_errors = []

    for res_dict in results_dict['results']:
        if res_dict.get('test_acc', False):
            model_names.append(res_dict['method'])
            model_errors.append(res_dict['test_acc'])

    df = pd.DataFrame({'method': model_names, 'accuracy': model_errors})

    fig = px.scatter(df, x="method", y="accuracy")
    fig.update_yaxes(range=[0, 1])

    return fig


# finds intersections in methods top weighted features and returns these for streamlit/ causal potential calculation
def get_method_intersections(results_dict, top_k=50):
    feat_dicts = []
    method_names = []
    method_accs = []

    all_features = set()

    # Collect ranked (sorted) features and weights into dataframes for each method
    for i, method_dict in enumerate(results_dict['results']):
        method = method_dict.get('method', "Non-Causal ERM")
        if method not in NON_CAUSAL:
            method_names += [method]
            method_accs += [method_dict['test_acc']]

            coefs = pd.DataFrame()
            coefs['feature'] = method_dict['features']

            # JC
            #coefs['coefficient'] = method_dict['coefficients']
            if method_dict['coefficients'] is None:
                coefs['coefficient'] = method_dict['coefficients']
            else:
                if type(method_dict['coefficients']) == list and len(method_dict['coefficients']) == 1:
                    _coefs = method_dict['coefficients'][0]
                elif isinstance(method_dict['coefficients'], float):
                    _coefs = [method_dict['coefficients']]
                else:
                    _coefs = method_dict['coefficients']
                coef_stdev = statistics.pstdev(_coefs)
                coef_mean = statistics.mean(_coefs)

                if coef_stdev != 0:
                    coefs['coefficient'] = [(n - coef_mean) / coef_stdev if n else 1 for n in _coefs]
                else:
                    coefs['coefficient'] = method_dict['coefficients']

            coefs['sort'] = coefs['coefficient'].abs()
            coefs = coefs.sort_values('sort', ascending=False)
            feat_dicts.append(coefs)
            all_features.update(list(coefs['feature'].values))

    # Find intersection between each methods top k features for every pair of methods
    method_intersections = pd.DataFrame(columns=method_names, index=method_names)
    for i, feat_dict in enumerate(feat_dicts):
        this_method_top_k = set(feat_dict['feature'][:top_k])
        for j in range(len(feat_dicts)):
            sets = [set(this_method_top_k)]
            if i == j:
                method_intersections[method_names[i]][method_names[j]] = ['-']
            else:
                sets.append(set(feat_dicts[j]['feature'][:top_k]))
                method_intersections[method_names[i]][method_names[j]] = list(set.intersection(*sets))

    return method_intersections, all_features, method_names, method_accs


# returns sankey diagram for streamlit. calculated causal potential for each feature given top_k
def plot_sankey_for_results(config, top_k, overall_importance):
    target_feature = config['data_options']['targets'][0]

    overall_importance = overall_importance.sort_values('sort', ascending=False)

    signs = []
    abs_weight = []
    for i, row in overall_importance.iterrows():
        signs.append(np.sign(row['weighted_coefficient']))
        abs_weight.append(np.abs(row['weighted_coefficient']))

    overall_importance['sign'] = signs
    overall_importance['weight'] = abs_weight

    weights = overall_importance

    labels_list = np.array(weights['feature'])[:top_k].tolist()
    labels_list += ['0:' + target_feature, f'1:{target_feature}']
    source = [i for i, l in enumerate(labels_list) if l not in [f'0:{target_feature}', f'1:{target_feature}']]
    value = [weights.loc[l, 'weight'] for i, l in enumerate(labels_list) if
             l not in [f'0:{target_feature}', f'1:{target_feature}']]

    # if sign is positive target -> 1:target_feature else 0:target_feature
    target = [
        labels_list.index(f'0:{target_feature}') if weights.loc[l, 'sign'] < 0 else labels_list.index(
            f'1:{target_feature}') for i, l in enumerate(labels_list) if
        l not in [f'0:{target_feature}', f'1:{target_feature}']]
    #     target = np.ones(len(labels_list) - 1) * labels_list.index(target_feature)
    colors = ['moccasin' if weights.loc[l, 'sign'] < 0 else 'powderblue' for i, l in
              enumerate(labels_list) if l not in [f'0:{target_feature}', f'1:{target_feature}']] + ['moccasin',
                                                                                                    'powderblue']

    # Plot Sankey diagram
    just_methods_df = weights.drop(columns=['weight', 'sort', 'sign', 'feature', 'weighted_coefficient', 'selected', 'count_models'])
    method_data = [[just_methods_df.loc[l, col] for col in just_methods_df.columns] for l in labels_list if
                   l not in [f'0:{target_feature}', f'1:{target_feature}']]

    hovertemplate = 'Feature %{source.label}:<br />'
    template_parts = [str(col) + ' weight %{label[' + str(i) + ']:.3f} <br />' for i, col in
                      enumerate(just_methods_df.columns)]
    for part in template_parts:
        hovertemplate += part
    hovertemplate += '<extra></extra>'

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels_list,
            color=colors,
            hovertemplate='%{label} has weight %{value}<extra></extra>'
        ),
        link=dict(
            source=source,  # indices correspond to labels, eg A1, A2, A2, B1, ...
            target=target,
            value=value,
            color=colors,
            label=method_data,
            hovertemplate=hovertemplate
        ))])

    fig.update_layout(title_text=f"Top {top_k} features when predicting {target_feature}", font_size=10)
    #     fig.write_image("streamlit_frontend/streamlit sankey.pdf")
    return fig


def get_corr_matrix(results_dict):
    from functools import reduce
    all_features = set()
    corrs = []
    for i, method_dict in enumerate(results_dict['results']):
        method = method_dict.get('method', "Non-Causal ERM")
        """
        Structure of method_dict
        "to_bucket": {
            'method': 'Method Name',
            'features': [X1, X2, X4, X32, .. Xmax],
            'coefficients': [w1, w2, w4, w32, .. wmax],
            'pvals': [p1, p2, p4, p32, .. pmax] || p_total || None
            'test_acc': 0.97 || None,
            'coefficient_correlation_matrix'
        }
        """
        if method not in NON_CAUSAL:
            # get feats sorted by highest absolute value;
            all_features.update(method_dict['features'])
            if 'coefficient_correlation_matrix' in method_dict.keys():
                coef_mat = method_dict['coefficient_correlation_matrix']
                df = pd.DataFrame(coef_mat, columns=method_dict['features'])
                df.index = method_dict['features']
                df.fillna(0, inplace=True)  # there are NAs here if std of either coef is 0, in which case uncorrelated
                corrs.append(df)
            else:
                print('no coefficient correlation matrix in method', method)

    # combine the dfs in corrs
    take_sum = lambda s1, s2: np.nansum(np.dstack((s1, s2)), 2)[
        0]  # if nan on one side treat as zero; if both are nan will return zero (this happens if there are features in Non-Causal ERM that aren't in any other models)
    combined = reduce(lambda left, right: left.combine(right, take_sum, overwrite=False),
                      corrs)  # this gets the sum of the correlation matrix across all the dataframes
    combined = combined / len(corrs)  # return mean coefficient correlation across all the models

    return combined


# helper function for sankey plot
def assign_edge_color(correlation):
    if correlation <= 0:
        return "blue"
    else:
        return "red"


def create_corr_network(G, corr_cutoff=0.7, fout='test.png', save_fig=False):
    H = G.copy()

    # Checks all the edges and removes some based on corr_direction
    to_remove_a = []
    to_remove_b = []

    for feat1, feat2, weight in H.edges(data=True):
        # if absolute value of weight is below our threshold, remove this edge
        if abs(weight["weight"]) < corr_cutoff:
            to_remove_a.append(feat1)
            to_remove_b.append(feat2)

    for i, el in enumerate(to_remove_a):
        H.remove_edge(el, to_remove_b[i])

    # remove any nodes without any connections
    remove = [node for node, degree in nx.degree(H) if degree <= 2]
    H.remove_nodes_from(remove)

    # positions
    positions = nx.spring_layout(H)

    edge_x = []
    edge_y = []
    edge_x2 = []
    edge_y2 = []
    for feat1, feat2, weight in H.edges(data=True):
        x0, y0 = positions[feat1][0], positions[feat1][1]
        x1, y1 = positions[feat2][0], positions[feat2][1]
        if weight['weight'] < 0:
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)
        else:
            edge_x2.append(x0)
            edge_x2.append(x1)
            edge_x2.append(None)
            edge_y2.append(y0)
            edge_y2.append(y1)
            edge_y2.append(None)

    edge_trace1 = go.Scatter(
        name="Negative",
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#0000ff'),
        hoverinfo='text',
        mode='lines')

    edge_trace2 = go.Scatter(
        name="Positive",
        x=edge_x2, y=edge_y2,
        line=dict(width=0.5, color='#ff0000'),
        hoverinfo='text',
        mode='lines')

    node_x = []
    node_y = []
    node_names = []
    for node in H.nodes():
        x, y = positions[node][0], positions[node][1]
        node_x.append(x)
        node_y.append(y)
        node_names.append(node)

    node_trace = go.Scatter(
        name='Features',
        x=node_x, y=node_y,
        text=node_names,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            # colorscale options
            # 'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            # 'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            # 'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))
    node_adjacencies = []

    for node, adjacencies in enumerate(H.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_names

    fig = go.Figure(data=[edge_trace1, edge_trace2, node_trace],
                    layout=go.Layout(
                        title='<br>Feature Correlations',
                        titlefont_size=16,
                        showlegend=True,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[dict(
                            text="AH Causal Ensemble",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002)],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="top",
        y=1.02,
        xanchor="right",
        x=1
    ))
    fig.update_layout(
        legend_title="")

    return fig


# returns sankey figure of correlations to target variable for streamlit
def plot_sankey_for_data(data_used, predictor_cols, target_col="PMMR_DMMR", top_k=20):
    correlations = [data_used[x].astype(float).corr(data_used[target_col].astype(float)) for x in predictor_cols]
    # sort correlations by absolute value
    r2abs = [abs(x) if not np.isnan(x) else 0 for x in correlations]
    sort_ix = np.argsort(r2abs)[::-1]  # descending order of absolute magnitude

    values = np.array(correlations)[sort_ix][0:top_k].tolist()
    labels_list = np.array(predictor_cols)[sort_ix][0:top_k].tolist()
    labels_list.append(target_col)
    source = [i for i, l in enumerate(labels_list) if l != target_col]
    target = np.ones(len(labels_list) - 1) * labels_list.index(target_col)
    abs_values = [abs(x) for x in values]
    hover_labels = [str(labels_list[i]) + ' : ' + str(round(value, 2)) for i, value in enumerate(values)]
    colors = ['powderblue' if v < 0 else 'salmon' for v in values]

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels_list,
            color=colors,
            hovertemplate='%{label}'
        ),
        link=dict(
            source=source,  # indices correspond to labels, eg A1, A2, A2, B1, ...
            target=target,
            value=abs_values,
            color=colors,
            label=hover_labels,
            hovertemplate='Feature correlation %{label}<br />'
        ))])

    fig.update_layout(title_text=f"Top {top_k} Feature Correlations With Target Variable", font_size=10)
    # fig.show()
    return fig
