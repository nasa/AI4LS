# src/transformations.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import List, Dict
from rnanorm import TPM

# define path to GTF file
gtf_filename = '/Users/jcasalet/Desktop/NASA/FOUNDATION_MODEL/ML_PIPELINE//DATA/Mus_musculus.GRCm39.115.gtf.gz'

class DataTransformer:
    """Handle various data transformations"""
    
    @staticmethod
    def tpm(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Apply transcripts per million transformation to specified columns"""
        tpm_calculator = TPM(gtf=gtf_filename)
        # assumes genes x samples
        df_t = df.T.copy() 

        # Set output format (e.g., pandas DataFrame)
        tpm_calculator.set_output(transform="pandas")

        # Transform raw counts to TPM
        temp_tpm = tpm_calculator.fit_transform(df_t)
        return df_t.T 

    @staticmethod
    def log_transform(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Apply log transformation to specified columns"""
        df_copy = df.copy()
        for col in columns:
            if df_copy[col].min() <= 0:
                # Add constant to handle non-positive values
                df_copy[col] = np.log1p(df_copy[col] - df_copy[col].min() + 1)
            else:
                df_copy[col] = np.log(df_copy[col])
        return df_copy
    
    @staticmethod
    def standardize(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Standardize columns to zero mean and unit variance"""
        df_copy = df.copy()
        scaler = StandardScaler()
        df_copy[columns] = scaler.fit_transform(df_copy[columns])
        return df_copy
    
    @staticmethod
    def normalize(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Normalize columns to [0, 1] range"""
        df_copy = df.copy()
        scaler = MinMaxScaler()
        df_copy[columns] = scaler.fit_transform(df_copy[columns])
        return df_copy
    
    @staticmethod
    def one_hot_encode(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """One-hot encode categorical columns"""
        return pd.get_dummies(df, columns=columns, drop_first=True)
