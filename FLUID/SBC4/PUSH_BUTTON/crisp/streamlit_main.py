from PIL import Image

import glob
import json
import os
import streamlit as st

from streamlit_frontend.streamlit_data import streamlit_data
from streamlit_frontend.streamlit_results import streamlit_results

image = Image.open('streamlit_frontend/ah_streamlit_banner.png')
st.image(image, use_column_width=True)


# plotly.io.orca.config.executable = '/opt/conda/bin/orca'
st.header('Welcome to the Astronaut Health FDL Streamlit Demo')

experiment_list = glob.glob('experiment_configs/*.json')

experiment_select = st.sidebar.selectbox("Which experiment would you like to explore?", experiment_list)

mode_select = st.sidebar.selectbox("What would you like to view?", ("Data", "Results"))

config = {}
if experiment_select:
    with open(os.path.join(os.getcwd(),experiment_select)) as json_file:
        config = json.load(json_file)
        bucket_fp = 'exp_y_' + config['data_options']['targets'][0] + '_env_' + config['data_options']['environments'][0] + '_' + config['short_name'] + '/' 
        config['bucket_exp_path'] = bucket_fp


if mode_select == 'Data':
    streamlit_data(config)
elif mode_select == 'Results':
    streamlit_results(config)
