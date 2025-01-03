import streamlit as st
import pandas as pd
import networkx as nx
from utils.plotting import create_corr_network, plot_sankey_for_data
from utils.streamlit_compute_average_voting import get_average_voting_ensemble
import plotly.express as px
from utils.gseapy_import import gsea

import traceback

'''
data options looks like this:
"data_options": {
        "dataset_fp": "data/metadata_variant.pickle",
        "subject_keys": "Subj_ID",
        "targets": ["PMMR_DMMR"],
        "predictors": "All",
        "environments": ["BMIgroup"],
        "exclude": ["SampleLocation", "BMIgroup","PMMR_DMMR", "smoke_ever", "Loc_APPENDIX", "Loc_ASCENDING", "Loc_CECUM", "Loc_DESCENDING", "Loc_SIGMOID", "Loc_TRANSVERSE", "TATTOO", "Stage", "Sex", "Loc_RECTUM", "Agegroup", "Type_TUMOR", "Type_NORMAL", "Type_ADJACENT", "num_variants_raw"],
        "remove_keys": ["Subj_ID", "SampleID"],
        "merge_keys": ["SampleID", "Subj_ID", "PMMR_DMMR"]
    },
'''

def single_seed_streamlit_data(results):
    
    n_display = st.slider(label='How many features per model to display?', min_value=20, max_value=1000, step=20, value=20)
    top_pathways_to_display = st.slider(label='How many pathways to display?', min_value=5, max_value=500, step=5, value=10) 
    
    for result in results:
        st.header(result['method'])
        data = pd.Series(dict(zip(result['features'],result['coefficients']))).abs().sort_values(ascending=False)
        sub_list = data.abs().sort_values(ascending=False).iloc[0:n_display] 
        st.write('all  features')
        st.write(px.bar(data, labels = {'index': result['method']+' all genes identified', 'value': result['method']+' assigned score'}))
        download_button(data, 'download all coefficient data from '+result['method'])
    
        st.write('top '+str(n_display)+' features')
        st.write(px.bar(sub_list, labels = {'index': result['method']+' top '+str(n_display)+' genes identified', 'value': result['method']+' assigned score'}))

        st.header('plot of pathways with greatest overlap with identified genes in '+result['method'])
        st.write("Please note: the genes quered are based on the option picked in 'How many features per model to display?' slider")

        top_n_genes= data.sort_values(ascending=False).iloc[:n_display].index.to_list()
        plot = gsea([x.upper() for x in top_n_genes], 'c6.all.v7.4.symbols.gmt',pcutoff=0.05, nPathways=top_pathways_to_display, figSize=(3,5), labelSize=20, title='identified oncogenic signiature genesets', filename='', streamlit_mode=True)
        st.write(plot)
        download_button(plot, 'download selected genesets identified by '+result['method'])
        
        
    st.header('the raw JSON data for each of the tested methods can be found below')
    for result in results:
        st.json(result)

        
def download_button(dataframe, button_name = 'download'):
    button_object=st.button(button_name)
    if button_object:
        csv = dataframe.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # some strings
        linko = f'<a href="data:file/csv;base64,{b64}" download="myfilename.csv">Download csv file</a>'
        st.markdown(linko, unsafe_allow_html=True)

    