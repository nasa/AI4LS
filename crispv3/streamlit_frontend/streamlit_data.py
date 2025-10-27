import streamlit as st
import pandas as pd
import networkx as nx
from utils.plotting import create_corr_network, plot_sankey_for_data

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


def streamlit_data(config):
    st.write('Data tab')

    # Show experiment config

    st.json(config)

    # Show dataframe used

    data_used = pd.read_pickle(config['data_options']['dataset_fp'])
    st.write(data_used)

    # Show correlations in dataframe

    '''try:
        # from data_used get predictor cols, remove any features that you're excluding in the config
        if config['data_options']['predictors'] != 'All':
            predictor_cols = config['data_options']['predictors']
        else:
            predictor_cols = list(data_used.columns)
            to_remove = config['data_options'].get('remove_keys', [])
            to_remove.extend(config['data_options']['targets'])
            to_remove.extend(config['data_options']['exclude'])
            [predictor_cols.remove(x) for x in to_remove if x in predictor_cols]

        ###feature correlations network plot###
        corr_cutoff = st.slider(label='Correlation cutoff:', min_value=0.0, max_value=1.0, step=0.1, value=0.9)

        corr = data_used[predictor_cols].corr()
        corr = corr.fillna(0)
        features = corr.index.values
        G = nx.from_numpy_matrix(corr.values)
        G = nx.relabel_nodes(G, lambda x: features[x])

        corr_fig = create_corr_network(G, corr_cutoff=corr_cutoff, save_fig=False)
        st.write(corr_fig)
    except Exception as e:
        tb = traceback.format_exc()
        print('Caught exception ', e, tb)

    # Show Feature correlations to target as sankey

    try:
        max_value = min(100, len(predictor_cols))
        top_k_sankey = st.slider(label='How many top most correlated features to target variable to show?', min_value=5,
                                 max_value=max_value, step=5, value=20)

        sankey_fig = plot_sankey_for_data(data_used, predictor_cols, target_col=config['data_options']['targets'][0],
                                          top_k=top_k_sankey)
        st.plotly_chart(sankey_fig, use_container_width=True)
    except Exception as e:
        st.write('Unable to show ')
        tb = traceback.format_exc()
        print('Caught exception ', e, tb)'''
