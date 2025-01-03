from PIL import Image

import glob
import json
import os
import streamlit as st

from streamlit_frontend.new_streamlit_data import streamlit_data
from streamlit_frontend.new_streamlit_results import streamlit_results
from streamlit_frontend.single_seed_streamlit_data import single_seed_streamlit_data

from io import StringIO
import pandas as pd
from tqdm import tqdm

image = Image.open('streamlit_frontend/ah_streamlit_banner.png')
st.image(image, use_column_width=True)

# plotly.io.orca.config.executable = '/opt/conda/bin/orca'
st.header('Welcome to the 2021 Astronaut Health FDL Streamlit!')
st.write('please use the sidebar on the left to select which experiment you would like to view. You can view results or data. Initialising a new experiment can take a few seconds, especially when building correlation matrices!')

experiment_list = [folder for folder in os.listdir('streamlit_data') if '.ipynb' not in folder] 
experiment_list = [folder for folder in experiment_list  if '.py' not in folder] 

experiment_select = st.sidebar.selectbox("Which experiment would you like to explore?",experiment_list)
mode_select = st.sidebar.selectbox("What would you like to view?", ("Results","Original Data"))

# config = {}
if experiment_select:
    print(experiment_select)
    json_files = [x for x in os.listdir('streamlit_data/'+experiment_select) if '.json' in x]
    json_files = [x for x in json_files if 'CSNX' not in x]

    if len(json_files) == 0:
        print('no json files here!')
    model_files = []
    for model in json_files:
        print(model)
        
        with open('streamlit_data/'+experiment_select+'/'+model) as json_file:
            data = json.load(json_file)  
        model_files.append(data)

    data = pd.read_csv('streamlit_data/'+experiment_select+'/data.csv')
    unnamed_columns = [col for col in data.columns if 'Unnamed' in col]
    final_frame = data.drop(unnamed_columns, axis = 1)

    if mode_select == 'Results':
        if len(json_files) == 0:
            st.write('Error, cannot print results: No JSON files found!')
        
        else:
            seed_mode = st.sidebar.selectbox("would you like to view one seed?", ['no','yes'])

            if seed_mode == 'no':
                print(len(model_files))
                streamlit_results(model_files, final_frame)

            if seed_mode == 'yes':
                selected_seed = st.sidebar.selectbox("which seed?", list(results.keys()))
                single_seed_streamlit_data([files[selected_seed] for files in model_files])

    if mode_select == 'Original Data':
        print('loading results')
        streamlit_data(final_frame)
    
    
