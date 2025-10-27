import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def normalize_except_exclude(tabular_data_df, exclude_vars):
    """
    Normalize all columns in the DataFrame except the columns specified in exclude_vars.
    
    Parameters:
    - tabular_data_df: pandas DataFrame containing the data
    - exclude_vars: List of column names to exclude from normalization
    
    Returns:
    - A pandas DataFrame with normalized data except for the excluded columns
    """
    # Separate the columns to exclude from the rest
    columns_to_exclude = tabular_data_df[exclude_vars]
    columns_to_normalize = tabular_data_df.drop(columns=exclude_vars)

    # Normalize the remaining columns using MinMaxScaler
    scaler = MinMaxScaler()  
    normalized_columns = scaler.fit_transform(columns_to_normalize)

    # Create a DataFrame from the normalized data
    normalized_columns_df = pd.DataFrame(normalized_columns, columns=columns_to_normalize.columns)

    # Concatenate the excluded columns with the normalized columns
    result_df = pd.concat([columns_to_exclude.reset_index(drop=True), normalized_columns_df.reset_index(drop=True)], axis=1)

    # Re-order columns to match the original DataFrame
    result_df = result_df[tabular_data_df.columns]

    return result_df

def merge_environments(gradcam_features_df, tabular_data_df, subject_key, env_column, merge_config):
    """
    Merge image and tabular data based on environment mappings defined in merge_config.
    
    Parameters:
    - gradcam_features_df: DataFrame containing GradCAM features.
    - tabular_data_df: DataFrame containing tabular features.
    - subject_key: Column name used to merge (e.g., 'sample').
    - env_column: Column name for environment labels (e.g., 'env_split').
    - merge_config: Dictionary defining environment mappings (from config JSON).
    
    Returns:
    - Merged DataFrame with unified environments.
    """
    merged_dfs = []
    
    for unified_env, env_mapping in merge_config.items():
        # Filter image data
        img_envs = env_mapping["img_env"]
        img_filtered = gradcam_features_df[gradcam_features_df[env_column].isin(img_envs)].copy()
        
        # Filter tabular data
        tabular_envs = env_mapping["tabular_env"]
        tabular_filtered = tabular_data_df[tabular_data_df[env_column].isin(tabular_envs)].copy()
        
        # Merge on subject_key
        merged = pd.merge(img_filtered, tabular_filtered, on=subject_key, how="inner", suffixes=('_img', '_tabular'))
        
        # Assign unified environment label
        merged[env_column] = unified_env
        
        merged_dfs.append(merged)
    
    # Combine all merged DataFrames
    return pd.concat(merged_dfs, ignore_index=True)

def log_alignment_summary(tabular_df, gradcam_df, subject_key):
    """
        Logs summary of alignment between tabular and GradCAM datasets using the subject key.
    """
    tabular_ids = set(tabular_df[subject_key].astype(str))
    gradcam_ids = set(gradcam_df[subject_key].astype(str))
    common_ids = tabular_ids & gradcam_ids
    only_in_tabular = sorted(tabular_ids - gradcam_ids)
    only_in_gradcam = sorted(gradcam_ids - tabular_ids)

    print(f"[INFO] Merging tabular and image features on key: '{subject_key}'")
    print(f"[INFO] Total tabular samples: {len(tabular_ids)}")
    print(f"[INFO] Total image samples: {len(gradcam_ids)}")
    print(f"[INFO] Retained common samples: {len(common_ids)}")
    print(f"[INFO] Dropped tabular-only samples: {len(only_in_tabular)}")
    print(f"[INFO] Dropped image-only samples: {len(only_in_gradcam)}")

    # Show mismatch subject keys
    if only_in_tabular:
        print(" - Tabular-only sample IDs:", ", ".join(only_in_tabular[:10]) + (" ..." if len(only_in_tabular) > 10 else ""))
    if only_in_gradcam:
        print(" - Image-only sample IDs:", ", ".join(only_in_gradcam[:10]) + (" ..." if len(only_in_gradcam) > 10 else ""))


def save_merged_features(config):
    """
        Combine tabular and image(gradcam) features based on subject key/sample
    """
    tabular_data_df = pd.read_pickle(config["image_data"]["tabular_features_path"])
    gradcam_features_df = pd.read_pickle(config["image_data"]["gradcam_features_save_path"])
    
    subject_key = config["data_options"]["subject_keys"]
    target_var = config["data_options"]["targets"]
    environments = config["data_options"]["environments"][0]
    exclude = config["data_options"]["exclude"]
    
    # Standardize keys
    tabular_data_df[subject_key] = tabular_data_df[subject_key].astype(str)
    gradcam_features_df[subject_key] = gradcam_features_df[subject_key].astype(str)

    # Log sample alignment
    log_alignment_summary(tabular_data_df, gradcam_features_df, subject_key)
    
    # Drop overlapping feature columns
    overlap_columns = tabular_data_df.columns.intersection(gradcam_features_df.columns).tolist()
    for col in [subject_key, environments]:
        if col in overlap_columns:
            overlap_columns.remove(col)
    tabular_data_df = tabular_data_df.drop(columns=overlap_columns)

    print("Tabular features", tabular_data_df.shape, "GradCAM features", gradcam_features_df.shape)

    # Merge data based on environment mappings if specified
    if "multimodal_merge_options" in config:
        merge_config = config["multimodal_merge_options"]["environment_split_unified"]
        merged_features_df = merge_environments(
            gradcam_features_df, tabular_data_df, subject_key, environments, merge_config
        )
    else:
        # Default merge (no environment mapping)
        merged_features_df = pd.merge(gradcam_features_df, tabular_data_df, on=subject_key, how="inner")

    # Normalize features (excluding specified columns)
    exclude_vars = list(set(exclude + target_var + [environments, environments+'_img', environments+'_tabular'] + ['num_activation_clusters']))
    merged_features_df = normalize_except_exclude(merged_features_df, exclude_vars)

    # Remove environments variable columns (_img, _tabular)
    merged_features_df = merged_features_df.drop(columns=[environments+'_img', environments+'_tabular'])
    print("Merged data", merged_features_df.shape)
    save_path = config["data_options"]["dataset_fp"]
    print('Saving', save_path.split('.')[0] + '.csv')
    merged_features_df.to_csv(save_path.split('.')[0] + '.csv', index=False)
    merged_features_df.reset_index(drop=True).to_pickle(save_path)