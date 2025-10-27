import json
import networkx as nx
import os
import pandas as pd
import streamlit as st
import traceback
import statistics
import sys

from utils.gcp_helpers import get_json_from_bucket
from utils.plotting import get_test_accuracy_figure, get_method_intersections, plot_sankey_for_results, get_corr_matrix, \
    create_corr_network, get_ensemble_results, get_fig_overall_most_predictive, get_fig_most_predictive


def streamlit_results(config):
    st.write('Results tab')

    # Show experiment config

    st.json(config)

    # Get results from local filesystem / cloud

    got_results = False
    try:
        if config.get('use_cloud', False):
            res_fp = config['bucket_path'] + config['bucket_exp_path']
            res_json = get_json_from_bucket(res_fp + 'results.json', config['bucket_project'], config['bucket_name'])
            got_results = True
        else:
            cwd = os.getcwd()
            results_directory = os.path.join(cwd, 'results', config['short_name'])
            res_fp = results_directory + '/'
            with open(os.path.join(res_fp, 'results_for_bucket.json')) as json_file:
                res_json = json.load(json_file)
            got_results = True
    except Exception as e:
        tb = traceback.format_exc()
        print('Caught exception ', e, tb)
        st.write('This experiment is still running, please check back later!')
    #     st.json(res_json)

    if got_results:

        # Plot method predictive accuracies

        test_acc_fig = get_test_accuracy_figure(res_json)
        st.plotly_chart(test_acc_fig)

        # Causal potential bar chart:
        max_value = 50  # max([len(res['features']) for res in res_json['results']])
        try:
            # Combined method plot
            overall_importance_slider_val = st.slider(label='How many features for top method?', min_value=5,
                                                      max_value=100, step=1, value=20)

            overall_importance = get_ensemble_results(res_json['results'])
            overall_fig = get_fig_overall_most_predictive(overall_importance, top_N=overall_importance_slider_val)
            st.write(overall_fig)

        except Exception as e:
            tb = traceback.format_exc()
            print('Caught exception ', e, tb)
            sys.exit(1)

        # Individual method plots

        for i, method_dict in enumerate(res_json['results']):
            method = method_dict.get('method', False)

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
            if method:
                print('Processing', method)
                coefs = pd.DataFrame()
                coefs['feature'] = method_dict['features']
                #coefs['coefficient'] = method_dict['coefficients']

                # JC fix for methods which return singleton list (ie a list with one element -- a list)
                if method_dict['coefficients'] is None:
                    coefs['coefficient'] = method_dict['coefficients']
                else:
                    if type(method_dict['coefficients']) == list and len(method_dict['coefficients']) == 1:
                        coef_stdev = statistics.pstdev(method_dict['coefficients'][0])
                        coef_mean = statistics.mean(method_dict['coefficients'][0])
                        my_coefs = [(n - coef_mean) / coef_stdev if n else 1 for n in method_dict['coefficients'][0]]

                    elif isinstance(method_dict['coefficients'], float):
                        coef_stdev = 0
                        my_coefs = [method_dict['coefficients']]

                    else:
                        coef_stdev = statistics.pstdev(method_dict['coefficients'])
                        coef_mean = statistics.mean(method_dict['coefficients'])
                        if coef_stdev != 0:
                            my_coefs = [(n - coef_mean) / coef_stdev if n else 1 for n in method_dict['coefficients']]
                        else:
                            my_coefs = [0 for n in method_dict['coefficients']]

                    if coef_stdev != 0:
                        coefs['coefficient'] = my_coefs
                    else:
                        coefs['coefficient'] = method_dict['coefficients']

                coefs['pvals'] = method_dict['pvals']

                fig = get_fig_most_predictive(coefs, method, overall_importance_slider_val)
                st.plotly_chart(fig)

        # Method intersections

        intersection_slider_val = st.slider(label='How many features from each method?', min_value=5,
                                            max_value=max_value, step=1, value=20)

        method_intersections, all_features, method_names, method_accs = get_method_intersections(res_json,
                                                                                                 intersection_slider_val)

        st.table(method_intersections)

        # Causal Potential Sankey plot

        top_k_slider_val = st.slider(label='Show top features in Sankey:', min_value=1, max_value=max_value, step=1,
                                     value=20)

        
        sankey = plot_sankey_for_results(config, top_k_slider_val, overall_importance)

        st.plotly_chart(sankey, use_container_width=True)

'''        try:

            # correlation/sensitivity network plot

            corr_cutoff = st.slider(label='Correlation cutoff:', min_value=0.0, max_value=1.0, step=0.1, value=0.5)

            corr = get_corr_matrix(res_json)
            corr = corr.fillna(0)
            features = corr.index.values
            G = nx.from_numpy_matrix(corr.values)
            G = nx.relabel_nodes(G, lambda x: features[x])

            fig = create_corr_network(G, corr_cutoff=corr_cutoff, save_fig=False)
            st.write(fig)

        except Exception as e:
            tb = traceback.format_exc()
            print('Caught exception ', e, tb)'''
