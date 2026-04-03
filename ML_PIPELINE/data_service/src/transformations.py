# src/transformations.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import List, Dict
from rnanorm import TPM
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# define path to GTF file
#gtf_filename = '/Users/jcasalet/Desktop/NASA/FOUNDATION_MODEL/ML_PIPELINE//DATA/Mus_musculus.GRCm39.115.gtf.gz'
gtf_filename = '/app/Mus_musculus.GRCm39.115.gtf.gz'

class DataTransformer:
    """Handle various data transformations"""
    
    @staticmethod
    def tpm_transform(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Apply transcripts per million transformation to specified columns"""
        logger.info("doing tpm transform")
        logger.info(f"factor value in cols: {'Factor Value[Spaceflight]' in list(df.columns)}")
        logger.info(f"using GTF file {gtf_filename}")
        tpm_calculator = TPM(gtf=gtf_filename)
        # assumes genes x samples
        df_copy = df.copy() 
    
        factor_col = None 
        if 'Factor Value[Spaceflight]' in list(df_copy.columns):
            factor_col = list(df_copy['Factor Value[Spaceflight]'])
            df_copy.drop(columns=['Factor Value[Spaceflight]'], inplace=True) 

        # Set output format (e.g., pandas DataFrame)
        tpm_calculator.set_output(transform="pandas")

        # Transform raw counts to TPM
        temp_tpm = tpm_calculator.fit_transform(df_copy)
        temp_tpm = temp_tpm.dropna(axis=1)

        if factor_col != None:
            temp_tpm['Factor Value[Spaceflight]'] = factor_col

        return temp_tpm

    @staticmethod
    def log_transform(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Apply log transformation to specified columns"""
        logger.info("doing log transform")
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
        logger.info("doing standardize transform")
        df_copy = df.copy()
        scaler = StandardScaler()
        df_copy[columns] = scaler.fit_transform(df_copy[columns])
        return df_copy
    
    @staticmethod
    def normalize(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Normalize columns to [0, 1] range"""
        logger.info("doing normalize transform")
        df_copy = df.copy()
        scaler = MinMaxScaler()
        df_copy[columns] = scaler.fit_transform(df_copy[columns])
        return df_copy
    
    @staticmethod
    def one_hot_encode(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """One-hot encode categorical columns"""
        return pd.get_dummies(df, columns=columns, drop_first=True)
