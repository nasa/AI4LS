from PIL import Image

import glob
import json
import os
import streamlit as st

from streamlit_frontend.new_streamlit_data import streamlit_data
from streamlit_frontend.new_streamlit_results import streamlit_results
from streamlit_frontend.single_seed_streamlit_data import single_seed_streamlit_data

from google.cloud import storage
from io import StringIO
import pandas as pd
from tqdm import tqdm

image = Image.open('streamlit_frontend/ah_streamlit_banner.png')
st.image(image, use_column_width=True)


# plotly.io.orca.config.executable = '/opt/conda/bin/orca'
st.header('Welcome to the 2021 Astronaut Health FDL Streamlit!')
st.write('please use the sidebar on the left to select which experiment you would like to view. You can view results or data. Initialising a new experiment can take a few seconds!')


bucket_name = 'ah21_data'
client = storage.Client()
bucket = client.bucket(bucket_name)
iterator = bucket.list_blobs(delimiter='/')
response = iterator._get_next_page_response()
# experiment_list = [prefix for prefix in response['prefixes'] if 'inal' in prefix]
experiment_list = [prefix for prefix in response['prefixes'] if 'reduced' in prefix]

# experiment_select = st.sidebar.selectbox("Which experiment would you like to explore?", [experiment_list[-3]])

experiment_select = st.sidebar.selectbox("Which experiment would you like to explore?",experiment_list)
mode_select = st.sidebar.selectbox("What would you like to view?", ("Results","Original Data"))

# config = {}
if experiment_select:
    print(experiment_select)
    iterator = bucket.list_blobs(prefix=experiment_select)
    json_files = [x.name for x in iterator if '.json' in x.name]
    temp_fix = [file for file in json_files if '/IRM' not in file]
#     temp_fix = [json_files[0]] + [json_files[2]] 
#     '''odhran - fix this when IRM.json is not 700mb boi'''
    json_files = temp_fix
    
    if len(json_files) == 0:
        print('no json files here!')
    model_files = []
    for model in json_files:
        print(model)
        blob = bucket.blob(model)
        json_data = blob.download_as_string()
        load = json.loads(json_data)
        intermediate = json.dumps({i : load[r]['to_bucket'] for i,r in enumerate(load)}) 
    #         '''odhran -this may no longer be needed'''
        results = json.loads(intermediate)    
        model_files.append(results)

    # with open('data.json', 'w') as f:
    #     json.dump(results, f)

    iterator = bucket.list_blobs(prefix=experiment_select)
    csv_files = [x.name for x in iterator if '.csv' in x.name]
    seed_0_files = [file for file in csv_files if experiment_select+'seed_0/' in file]

    
    file_array = []
    for file in tqdm(seed_0_files):
     #     print(model)
        blob = bucket.blob(file)
        pandas_data = blob.download_as_string()
        s=str(pandas_data,'utf-8')
        pandas_data = StringIO(s) 
        env_frame = pd.read_csv(pandas_data)
        env_frame['environment'] = file[len(experiment_select+'seed_0'):]
        file_array.append(env_frame)

    final_frame = pd.concat(file_array)
    print('final frame loaded successfully')
    
    if mode_select == 'Results':

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
    
    
