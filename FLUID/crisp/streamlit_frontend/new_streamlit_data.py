import streamlit as st
import pandas as pd
import networkx as nx
from utils.plotting import create_corr_network, plot_sankey_for_data
from utils.streamlit_compute_average_voting import get_average_voting_ensemble
from utils.gseapy_import import gsea
import base64

import traceback
from io import StringIO

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

def streamlit_data(dataframe):
    st.header('Data tab') 
    predictor_frame = dataframe.drop(['Target','environment'], axis=1)
    st.header('Original Data')
    st.write(dataframe)
    
    try:
        st.header('Sankey plot visualisation')
        max_value = min(100, len(predictor_frame.columns))
        top_k_sankey = st.slider(label='How many top most correlated features to target variable to show?', min_value=5,
                                 max_value=max_value, step=5, value=20)
        sankey_fig = plot_sankey_for_data(dataframe, predictor_frame.columns, target_col='Target',
                                          top_k=top_k_sankey)
        st.plotly_chart(sankey_fig, use_container_width=True)
    except Exception as e:
        st.write('Unable to show ')
        tb = traceback.format_exc()
        print('Caught exception ', e, tb)

    
    print(len(predictor_frame.columns))
    build_corr = st.sidebar.selectbox("Would you like to build a correlation matrix?", ("No","Yes"))
  
    if len(predictor_frame.columns) > 1000 and build_corr == 'Yes':
        st.write('this dataframe has more than 1000 columns. Please reduce dimensionality to load a correlation matrix')
   
    elif len(predictor_frame.columns) <= 1000 and build_corr == 'Yes':
        st.header('Data correlation visualisation')
        corr_cutoff = st.slider(label='Correlation cutoff:', min_value=0.0, max_value=1.0, step=0.1, value=0.9)

        corr = predictor_frame.corr()
        corr = corr.fillna(0)
        features = corr.index.values
        G = nx.from_numpy_matrix(corr.values)
        G = nx.relabel_nodes(G, lambda x: features[x])

        corr_fig = create_corr_network(G, corr_cutoff=corr_cutoff, save_fig=False)
        st.write(corr_fig)