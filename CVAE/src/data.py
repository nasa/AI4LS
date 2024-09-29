import utils
import anndata
import numpy as np
import pandas as pd
import requests
import scanpy as sc
import scanpy.external as sce
import scanorama

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

# get metadata jsons from API
response_612 = requests.get("https://osdr.nasa.gov/osdr/data/osd/meta/612").json()
list_612 = response_612['study']['OSD-612']['additionalInformation']['samples']['s_OSD-612-txt']['raw']

response_613 = requests.get("https://osdr.nasa.gov/osdr/data/osd/meta/613").json()
list_613 = response_613['study']['OSD-613']['additionalInformation']['samples']['s_OSD-613-txt']['raw']

response_352 = requests.get("https://osdr.nasa.gov/osdr/data/osd/meta/352").json()
list_352 = response_352['study']['OSD-352']['additionalInformation']['samples']['s_OSD-352-txt']['raw']

# extra processing for 352 because not all samples are sequenced
sc_list_352 = response_352['study']['OSD-352']['additionalInformation']['assays']['a_OSD-352_transcription-profiling_single-cell-rna-sequencing_illumina-txt']['raw']

sc_samples_352 = []
for item in sc_list_352:
    sample_name = item.get('a100000samplename', '')
    sc_samples_352.append(sample_name)


rows_612 = []
for item in list_612:
    strain = item.get('a100005characteristicsstrain', '')
    sex = item.get('a100012characteristicssex', '')
    age_at_launch = utils.parse_age_at_launch(item.get('a100023characteristicsageatlaunch', '0 - 0'))
    duration = utils.parse_duration(item.get('a100033parametervalueduration', '0 day'))
    flight = utils.parse_flight(item.get('a100020factorvaluespaceflight'))

    rows_612.append([strain, sex, age_at_launch, duration, flight])


rows_613 = []
for item in list_613:
    strain = item.get('a100005characteristicsstrain', '')
    sex = item.get('a100012characteristicssex', '')
    age_at_launch = utils.parse_age_at_launch_2(item.get('a100018factorvalueage', '0 week'))
    duration = utils.parse_duration(item.get('a100034parametervalueduration', '0 days'))
    flight = utils.parse_flight(item.get('a100015factorvaluespaceflight'))

    rows_613.append([strain, sex, age_at_launch, duration, flight])

rows_352 = []
for item in list_352:
    if item['a100001samplename'] in sc_samples_352:
        strain = item.get('a100005characteristicsstrain', '')
        sex = item.get('a100013characteristicssex', '')
        age_at_launch = utils.parse_age_at_launch_2(item.get('a100009characteristicsageatlaunch', '0 week'))
        duration = utils.parse_duration(item.get('a100031parametervalueduration', '0 day'))
        flight = utils.parse_flight(item.get('a100021factorvaluespaceflight'))

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