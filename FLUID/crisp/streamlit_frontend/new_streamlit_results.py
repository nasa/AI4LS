import json
import networkx as nx
import os
import pandas as pd
import streamlit as st
import traceback
import plotly.express as px

from utils.gcp_helpers import get_json_from_bucket
from utils.plotting import get_test_accuracy_figure, new_get_method_intersections, plot_sankey_for_results, new_get_corr_matrix, \
    create_corr_network, get_ensemble_results, get_fig_overall_most_predictive, get_fig_most_predictive

from utils.streamlit_compute_average_voting import get_average_voting_ensemble
from utils.gseapy_import import gsea
import base64

def download_button(dataframe, button_name = 'download'):
    button_object=st.button(button_name)
    if button_object:
        csv = dataframe.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # some strings
        linko = f'<a href="data:file/csv;base64,{b64}" download="myfilename.csv">Download csv file</a>'
        st.markdown(linko, unsafe_allow_html=True)

def streamlit_results(results, final_frame):
    st.header('Results tab')
     
    st.header('method accuracy accross seeds')
    test_acc_fig = get_test_accuracy_figure(results)
    st.write(test_acc_fig)
    
    st.header('scores accross models and ensemble score')
    average_voting_ensemeble = get_average_voting_ensemble(results)
    st.write(average_voting_ensemeble)
    download_button(average_voting_ensemeble, 'download model-assigned values')
    
    st.header('plot of scores of models with ensemble scores')
    n_display = st.slider(label='How many features to display?', min_value=10, max_value=1000, step=10, value=100)
    values_to_plot = average_voting_ensemeble.sort_values(average_voting_ensemeble.columns[-1], ascending=False).iloc[:n_display]
#     st.bar_chart(values_to_plot)
    plotly_object = px.bar(values_to_plot, labels = {'index':'Gene','value':'CRISP association score'})
    st.write(plotly_object)
    
    
    st.header('plot of scores of each individual method')
    overall_importance_slider_val = st.slider(label='How many features per model to display?', min_value=10, max_value=1000, step=10, value=100) 
    for column in average_voting_ensemeble.columns:
        to_plot = average_voting_ensemeble[column].sort_values(ascending=False).iloc[:overall_importance_slider_val]
#         st.bar_chart(to_plot)
        plotly_object = px.bar(to_plot, labels = {'index':'Gene','value':'CRISP association score'})
        st.write(plotly_object)
        
        
    st.header('plot of pathways with greatest overlap with identified genes')
    st.write("Please note: the genes quered are based on the option picked in 'How many features per model to display?' slider")
    top_pathways_to_display = st.slider(label='How many pathways to display?', min_value=5, max_value=500, step=5, value=10) 
    top_n_genes= average_voting_ensemeble[average_voting_ensemeble.columns[-1]].sort_values(ascending=False).iloc[:overall_importance_slider_val].index.to_list()
    plot = gsea([x.upper() for x in top_n_genes], 'c6.all.v7.4.symbols.gmt',pcutoff=0.05, nPathways=top_pathways_to_display, figSize=(3,5), labelSize=20, title='identified oncogenic signiature genesets', filename='', streamlit_mode=True)
    st.write(plot)
    download_button(plot, 'download selected genesets')

    st.header('plot of gene targets that methods are concordant between')
    intersection_slider_val = st.slider(label='How many of the top-n results from each method would you like to consider?', min_value=5, max_value=500, step=1, value=50)
    method_intersections = new_get_method_intersections(average_voting_ensemeble,intersection_slider_val)
    st.table(method_intersections)
    download_button(method_intersections, 'download method intersections ')

    st.header('correlation matrix of the top-n genes identified by the ensemble')
    st.write("Please note: the genes quered are based on the option picked in 'How many features per model to display?' slider")
    predictor_frame = final_frame.drop(['Target','environment'], axis=1)
    print(predictor_frame.columns)
    correlation_matrix = predictor_frame[top_n_genes].corr()
    st.write(px.imshow(correlation_matrix))
    download_button(correlation_matrix, 'download correlation_matrix')
    
    st.header('web visualisation of the correlation matrix')
    corr_cutoff = st.slider(label='Correlation cutoff:', min_value=0.0, max_value=1.0, step=0.1, value=0.5)
    corr = correlation_matrix
    corr = corr.fillna(0)
    features = corr.index.values
    G = nx.from_numpy_matrix(corr.values)
    G = nx.relabel_nodes(G, lambda x: features[x])
    corr_fig = create_corr_network(G, corr_cutoff=corr_cutoff, save_fig=False)
    st.write(corr_fig)
    
    st.header('the raw JSON data for each of the tested methods can be found below')
   # Show experiment config
    for result in results:
        st.json(result)
    
