import utils
import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import scanpy.external as sce
import scanorama
import os

# read in h5 files
adata_612 = sc.read_10x_h5('data/612_filtered_feature_bc_matrix.h5')
adata_613 = sc.read_10x_h5('data/613_filtered_feature_bc_matrix.h5')
adata_352 = sc.read_10x_h5('data/352_filtered_feature_bc_matrix.h5')

adata_list = [adata_612, adata_613, adata_352]

# extract pseudo batch number
for adata in adata_list:
    adata.obs['batch'] = adata.obs_names.map(utils.extract_batch)
    adata.obs['batch'] = adata.obs['batch'].astype(int)
    adata.obs['batch'] = adata.obs_names.map(utils.extract_batch)
    adata.obs['batch'] = adata.obs['batch'].astype(int)
    adata = utils.preprocess_adata(adata, n_top_genes=1000)
    sc.pp.pca(adata, n_comps=50)

# Read metadata directly from txt files
def read_metadata_txt(file_path):
    """Read tab-delimited metadata from txt file"""
    try:
        df = pd.read_csv(file_path, sep='\t')
        return df
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return pd.DataFrame()

# Read sample metadata files
s_352_df = read_metadata_txt('data/s_OSD-352.txt')
s_612_df = read_metadata_txt('data/s_OSD-612.txt')
s_613_df = read_metadata_txt('data/s_OSD-613.txt')

# Get samples that have single-cell RNA sequencing data for 352 dataset
def get_sc_samples_352():
    """Get samples with single-cell RNA sequencing data from a_OSD file"""
    sc_samples = []
    a_352_path = 'data/a_OSD-352_transcription-profiling_single-cell-rna-sequencing_illumina.txt'
    
    if os.path.exists(a_352_path):
        a_352_df = read_metadata_txt(a_352_path)
        if not a_352_df.empty:
            sc_samples = a_352_df['Sample Name'].tolist()
    
    return sc_samples

sc_samples_352 = get_sc_samples_352()

# Process 612 sample metadata
rows_612 = []
for _, row in s_612_df.iterrows():
    strain = row.get('Characteristics[Strain]', '')
    sex = row.get('Characteristics[Sex]', '')
    
    # Extract age at launch from the data - handle different formats
    age_value = row.get('Characteristics[Age at Launch]', '0')
    age_unit = row.get('Unit', 'week')
    if pd.isna(age_value) or pd.isna(age_unit):
        age_at_launch = 0
    else:
        age_text = f"{age_value} {age_unit}"
        age_at_launch = utils.parse_age_at_launch(age_text)
    
    # Extract duration
    duration_value = row.get('Parameter Value[duration]', '0')
    duration_unit = row.get('Unit.1', 'day')  # Using Unit.1 for the duration unit
    if pd.isna(duration_value) or pd.isna(duration_unit):
        duration = 0
    else:
        duration_text = f"{duration_value} {duration_unit}"
        duration = utils.parse_duration(duration_text)
    
    # Extract flight status
    flight = utils.parse_flight(row.get('Factor Value[Spaceflight]', ''))
    
    rows_612.append([strain, sex, age_at_launch, duration, flight])

# Process 613 sample metadata
rows_613 = []
for _, row in s_613_df.iterrows():
    strain = row.get('Characteristics[Strain]', '')
    sex = row.get('Characteristics[Sex]', '')
    
    # Extract age at launch from the data
    age_value = row.get('Characteristics[Age at Launch]', '0')
    age_unit = row.get('Unit', 'week')
    if pd.isna(age_value) or pd.isna(age_unit):
        age_at_launch = 0
    else:
        age_text = f"{age_value} {age_unit}"
        age_at_launch = utils.parse_age_at_launch_2(age_text)
    
    # Extract duration
    duration_value = row.get('Parameter Value[duration]', '0')
    duration_unit = row.get('Unit.1', 'day')  # Using Unit.1 for the duration unit
    if pd.isna(duration_value) or pd.isna(duration_unit):
        duration = 0
    else:
        duration_text = f"{duration_value} {duration_unit}"
        duration = utils.parse_duration(duration_text)
    
    # Extract flight status
    flight = utils.parse_flight(row.get('Factor Value[Spaceflight]', ''))
    
    rows_613.append([strain, sex, age_at_launch, duration, flight])

# Process 352 sample metadata - only include samples with single-cell RNA sequencing
rows_352 = []
for _, row in s_352_df.iterrows():
    sample_name = row.get('Sample Name', '')
    # Only include samples with single-cell RNA sequencing data
    if sample_name in sc_samples_352 or not sc_samples_352:  # Include all if sc_samples list is empty
        strain = row.get('Characteristics[Strain]', '')
        sex = row.get('Characteristics[Sex]', '')
        
        # Extract age at launch
        age_value = row.get('Characteristics[Age at Launch]', '0')
        age_unit = row.get('Unit', 'week')
        if pd.isna(age_value) or pd.isna(age_unit):
            age_at_launch = 0
        else:
            age_text = f"{age_value} {age_unit}"
            age_at_launch = utils.parse_age_at_launch_2(age_text)
        
        # Extract duration
        duration_value = row.get('Parameter Value[duration]', '0')
        duration_unit = row.get('Unit.2', 'day')  # Using Unit.2 for the duration unit in 352 dataset
        if pd.isna(duration_value) or pd.isna(duration_unit):
            duration = 0
        else:
            duration_text = f"{duration_value} {duration_unit}"
            duration = utils.parse_duration(duration_text)
        
        # Extract flight status
        flight = utils.parse_flight(row.get('Factor Value[Spaceflight]', ''))
        
        rows_352.append([strain, sex, age_at_launch, duration, flight])

# create dataframes
df_612 = pd.DataFrame(rows_612, columns=['Strain', 'Sex', 'Age at Launch', 'Duration', 'Flight'])
df_613 = pd.DataFrame(rows_613, columns=['Strain', 'Sex', 'Age at Launch', 'Duration', 'Flight'])
df_352 = pd.DataFrame(rows_352, columns=['Strain', 'Sex', 'Age at Launch', 'Duration', 'Flight'])

# display dataframe to check
# print(df_612)
# print(df_613)
# print(df_352)

for column in ['Strain', 'Sex', 'Age at Launch', 'Duration', 'Flight']:
    if column not in adata_612.obs.columns:
        adata.obs[column] = 1
    if column not in adata_613.obs.columns:
        adata.obs[column] = 1
    if column not in adata_352.obs.columns:
        adata.obs[column] = 1

adata_612.obs['dataset'] = '612'
adata_613.obs['dataset'] = '613'
adata_352.obs['dataset'] = '352'

df_list = [df_612, df_613, df_352]

# iterate over pairs of df and adata objects
for df, adata in zip(df_list, adata_list):
    for idx, row in df.iterrows():
        batch = idx + 1
        if batch in adata.obs['batch'].values:
            mask = adata.obs['batch'] == batch
            adata.obs.loc[mask, 'Strain'] = row['Strain']
            adata.obs.loc[mask, 'Sex'] = row['Sex']
            adata.obs.loc[mask, 'Age at Launch'] = row['Age at Launch']
            adata.obs.loc[mask, 'Duration'] = row['Duration']
            adata.obs.loc[mask, 'Flight'] = row['Flight']

# print(adata_612)
# print(adata_613)
# print(adata_352)

adata_612.write('data/processed_612_data.h5ad')
adata_613.write('data/processed_613_data.h5ad')
adata_352.write('data/processed_352_data.h5ad')

print("processed adatas")

# Can comment out all above and read in saved preprocessed data instead!
# adata_612 = sc.read_h5ad('data/processed_612_data.h5ad')
# adata_613 = sc.read_h5ad('data/processed_613_data.h5ad')
# adata_352 = sc.read_h5ad('data/processed_352_data.h5ad')

adata_list = [adata_612, adata_613, adata_352]

# scanorama.integrate_scanpy(adata_list)

adata_corrected = scanorama.correct_scanpy(adata_list, return_dimred=True, hvg=2000, seed=42)

# correct_adata = sc.concat(corrected_adatas, join='outer', index_unique=None)

# adata_corrected = sce.pp.mnn_correct(adata_list, batch_key='dataset')

print("Finished integration")

adata_combined = anndata.concat(adata_list, axis=0)

print(adata_combined)

adata_combined.write('data/corrected_data.h5ad')

# correct_adata.write('corrected_data.h5ad')