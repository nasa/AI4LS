# data_service/src/service.py

import grpc
import logging
import os
import json
import uuid
import hashlib
import zipfile
import pandas as pd
import numpy as np
from pathlib import Path
from io import StringIO
import requests

from generated import data_service_pb2, data_service_pb2_grpc

from src.transformations import DataTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# NASA OSDR API URLs
NASA_FILES_URL = "https://osdr.nasa.gov/osdr/data/osd/files/{osd_id}"
NASA_DOWNLOAD_URL = "https://osdr.nasa.gov{remote_url}"  # remote_url already has full path

class DataServiceImpl(data_service_pb2_grpc.DataServiceServicer):
    """Implementation of DataService gRPC methods"""
    
    def __init__(self, dataset_path: str = "/app/datasets"):
        self.dataset_path = Path(dataset_path)
        self.dataset_path.mkdir(parents=True, exist_ok=True)

        self.transformer = DataTransformer()
        
        # In-memory cache
        self.datasets = {}
        
        # Download cache (maps download parameters to dataset_id)
        self.download_cache_file = self.dataset_path / "download_cache.json"
        self.download_cache = self._load_download_cache()
        
        logger.info(f"DataService initialized with dataset path: {self.dataset_path}")
        logger.info(f"Loaded {len(self.download_cache)} cached downloads")
    
    def _load_download_cache(self):
        """Load download cache from disk"""
        if self.download_cache_file.exists():
            try:
                with open(self.download_cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading download cache: {e}")
                return {}
        return {}
    
    def _save_download_cache(self):
        """Save download cache to disk"""
        try:
            with open(self.download_cache_file, 'w') as f:
                json.dump(self.download_cache, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving download cache: {e}")
    
    def _save_dataset_to_disk(self, dataset_id: str, df: pd.DataFrame):
        """Save dataset to disk as parquet"""
        try:
            filepath = self.dataset_path / f"{dataset_id}.parquet"
            df.to_parquet(filepath, index=True)
            logger.info(f"Saved dataset {dataset_id} to {filepath}")
        except Exception as e:
            logger.error(f"Error saving dataset to disk: {e}")
    
    def _fetch_json(self, url: str):
        """Fetch JSON from URL"""
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.json()
    
    def _fetch_text(self, url: str):
        """Fetch text content from URL"""
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.text
    
    def _fetch_bytes(self, url: str):
        """Fetch binary content from URL"""
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.content
    
    def _find_file(self, files, osd_key, patterns):
        """Find files matching patterns"""
        matches = []
        
        # Handle different response structures
        if isinstance(files, dict):
            # New API structure: {"studies": {osd_key: {"study_files": [...]}}}
            if "studies" in files and osd_key in files["studies"]:
                study_data = files["studies"][osd_key]
                if "study_files" in study_data:
                    file_list = study_data["study_files"]
                else:
                    logger.error(f"No 'study_files' in study data")
                    return []
            # Old structure: {osd_key: {"files": [...]}}
            elif osd_key in files and "files" in files[osd_key]:
                file_list = files[osd_key]["files"]
            # Alternative: {"files": [...]}
            elif "files" in files:
                file_list = files["files"]
            else:
                logger.error(f"Unexpected files structure. Keys: {files.keys()}")
                return []
        elif isinstance(files, list):
            file_list = files
        else:
            logger.error(f"Unexpected files type: {type(files)}")
            return []
        
        # Extract filenames from file_list
        all_filenames = []
        for item in file_list:
            # Handle if item is a string (filename)
            if isinstance(item, str):
                filename = item
            # Handle if item is a dict with various possible keys
            elif isinstance(item, dict):
                filename = item.get("file_name", item.get("file_url", item.get("filename", "")))
            else:
                continue
            
            if filename:
                all_filenames.append(filename)
        
        logger.info(f"Total files available: {len(all_filenames)}")
        if all_filenames:
            logger.info(f"First 5 files: {all_filenames[:5]}")
        
        # Now search for matches
        for filename in all_filenames:
            # Check if all patterns match (case-insensitive)
            if all(p.lower() in filename.lower() for p in patterns):
                matches.append(filename)
        
        logger.info(f"Found {len(matches)} files matching patterns {patterns}")
        if not matches and all_filenames:
            logger.warning(f"No matches found. Looking for patterns: {patterns}")
            logger.warning(f"Sample filenames: {all_filenames[:10]}")
        
        return matches
    
    def _pick_rna_file(self, matches: list):
        """Pick the best RNA-seq file from matches"""
        # Prefer gene counts over transcript counts
        gene_files = [f for f in matches if 'gene' in f.lower()]
        if gene_files:
            return gene_files[0]
        return matches[0]
    
    def check_for_nans(self, df: pd.DataFrame):
        """Remove rows/columns with excessive NaNs"""
        # Remove rows with >50% NaN
        row_threshold = len(df.columns) * 0.5
        df = df.dropna(thresh=row_threshold)
        
        # Remove columns with >50% NaN
        col_threshold = len(df) * 0.5
        df = df.dropna(axis=1, thresh=col_threshold)
        
        return df
    
    '''def transpose_df(self, df: pd.DataFrame):
        """Transpose if genes are columns instead of rows"""
        # Heuristic: if there are more columns than rows, likely needs transpose
        if len(df.columns) > len(df):
            logger.info("Transposing DataFrame (genes as columns -> genes as rows)")
            df = df.T
        return df'''

    def _filter_cvs(self, df, start, step=0.25, min_features=1000):
        # calculate coefficient of variation
        # assumes samples x genes
        if df.shape[1] <= min_features:
            logger.info(f"len(df) is less than min_features {min_features}") 
            return df 
        keep_columns_use = list(df.columns)
        while True: 
            keep_columns = list() 
            for col in list(df.columns):
                m = np.mean(df[col])
                sd = np.std(df[col])
                if m != 0 and sd/m > start:
                    keep_columns.append(col)
            if len(keep_columns) < min_features:
                logger.info(f"keep cols less than min cols: {len(keep_columns)} ")
                logger.info(f"length of keep_cols_use: {len(keep_columns_use)}")
                break
            else:
                keep_columns_use = keep_columns 
                start += step
                logger.info(f"stepping up start: {start}")

        return df[keep_columns_use]
    
    '''def _filter_cvs(self, df: pd.DataFrame, start: float = 1, step: float = 0.25, 
                    min_features: int = 1000):
        """
        Filter features by coefficient of variation (CV).
        Keeps features with CV above a threshold.
        """
        logger.info(f"Filtering features: start={start}, step={step}, min_features={min_features}")
        
        # Calculate CV for each feature (column)
        means = df.mean(axis=0)
        stds = df.std(axis=0)
        
        # Avoid division by zero
        cvs = stds / (means + 1e-10)
        
        # Start with high threshold and reduce until we have enough features
        threshold = start
        while True:
            selected = cvs >= threshold
            num_selected = selected.sum()
            
            logger.info(f"  CV threshold {threshold:.2f}: {num_selected} features")
            
            if num_selected >= min_features or threshold <= 0:
                break
            
            threshold -= step
        
        # Keep features above threshold
        filtered_df = df.loc[:, selected]
        
        logger.info(f"CV filtering: {df.shape[1]} → {filtered_df.shape[1]} features")
        
        return filtered_df'''

    def transpose_df(self, df):
        # if num cols > num rows, assume cols = genes and rows = samples 
        logger.info(f"before transpose: head is {df.head()}")
        dft = df.T
        logger.info(f"after transpose: head is {dft.head()}")
        dft.columns = dft.iloc[0]
        dft.drop(dft.index[0], inplace=True)
        logger.info(f"after drop index: head is {dft.head()}")
        dft.rename_axis('sample', inplace=True)
        logger.info(f"after rename axis: head is {dft.head()}")
        return dft 

    
    def _generate_download_cache_key(self, osd_id, patterns, factor_name, 
                                     factor_values, exclude_columns):
        """Generate cache key for RAW downloads (no filtering/transformation)"""
        key_parts = [
            str(osd_id),
            ",".join(sorted(patterns)),
            factor_name,
            ",".join(sorted(factor_values)),
            ",".join(sorted(exclude_columns))
        ]
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _build_dataset_info(self, dataset_id: str, df: pd.DataFrame):
        """Build DatasetInfo message from DataFrame"""
        
        # Build ColumnInfo for each column
        columns_info = []
        for col in df.columns:
            # Determine dtype
            if pd.api.types.is_numeric_dtype(df[col]):
                dtype = "numeric"
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                dtype = "datetime"
            elif pd.api.types.is_categorical_dtype(df[col]) or df[col].dtype == "object":
                dtype = "categorical"
            else:
                dtype = "text"
            
            # Get sample values
            sample_values = df[col].dropna().astype(str).head(3).tolist()
            
            columns_info.append(data_service_pb2.ColumnInfo(
                name=str(col),
                dtype=dtype,
                null_count=int(df[col].isnull().sum()),
                sample_values=sample_values
            ))
        
        return data_service_pb2.DatasetInfo(
            dataset_id=dataset_id,
            num_rows=int(len(df)),
            num_columns=int(len(df.columns)),
            column_names=[str(c) for c in df.columns],
            sample_names=[str(s) for s in df.index],
            size_bytes=int(df.memory_usage(deep=True).sum()),
            columns=columns_info
        )
    
    def HealthCheck(self, request, context):
        """Health check endpoint"""
        return data_service_pb2.HealthCheckResponse(
            healthy=True,
            version="1.0.0"
        )
    
    def ValidateDataset(self, request, context):
        """Validate dataset parameters without downloading"""
        logger.info(f"ValidateDataset request: {request}")
        
        errors = []
        warnings = []
        
        # Basic validation
        if not request.osd_id:
            errors.append("osd_id is required")
        
        if not request.factor_name:
            warnings.append("factor_name not specified")
        
        if len(request.factor_values) == 0:
            warnings.append("factor_values not specified")
        
        return data_service_pb2.ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
   
    def DownloadDataset(self, request, context):
        """
        Download an RNA-seq counts file from NASA OSDR, apply basic cleanup,
        add metadata/condition, and return RAW dataset (NO CV filtering).
        """
        logger.info(f"request in DownloadDataset: {request}")
        try:
            osd_id = request.osd_id
            patterns = list(request.patterns) or ["Unnormalized", "RSEM"]
            osd_key = f"OSD-{osd_id}"
            factor_name = request.factor_name
            factor_values = list(request.factor_values)
            exclude_columns = list(request.exclude_columns)

            # Generate cache key - ONLY based on download params (no cv_step/min_features)
            cache_dict = {
                "osd_id": osd_id,
                "patterns": sorted(patterns),
                "factor_name": factor_name or "",
                "factor_values": sorted(factor_values) if factor_values else [],
                "exclude_columns": sorted(exclude_columns) if exclude_columns else []
            }
            cache_str = json.dumps(cache_dict, sort_keys=True)
            cache_key = hashlib.md5(cache_str.encode()).hexdigest()
        
            # Check cache
            if cache_key in self.download_cache:
                cached_dataset_id = self.download_cache[cache_key]
                cached_file = self.dataset_path / f"{cached_dataset_id}.parquet"
                
                if cached_file.exists():
                    logger.info(f"✓ Found cached raw dataset for OSD-{osd_id}: {cached_dataset_id}")
                    
                    if cached_dataset_id not in self.datasets:
                        df = pd.read_parquet(cached_file)
                        self.datasets[cached_dataset_id] = df
                    
                    df = self.datasets[cached_dataset_id]
                    dataset_info = self._build_dataset_info(cached_dataset_id, df)
                    
                    return data_service_pb2.ValidationResult(
                        is_valid=True,
                        dataset_id=cached_dataset_id,
                        errors=[],
                        warnings=["Using cached raw dataset from previous download"],
                        dataset_info=dataset_info
                    )
                else:
                    logger.warning(f"Cached dataset file not found, will re-download")
                    del self.download_cache[cache_key]
                    self._save_download_cache()
        
            logger.info(f"Downloading OSD-{osd_id} (cache key: {cache_key})")

            # Step 1: fetch file listing
            files = self._fetch_json(NASA_FILES_URL.format(osd_id=osd_id))
            logger.info(f"API response type: {type(files)}")

            # Step 2: find matching files
            matches = self._find_file(files, osd_key, patterns)
            if not matches:
                return data_service_pb2.ValidationResult(
                    is_valid=False,
                    errors=[f"No files matched patterns {patterns} in {osd_key}"]
                )
            logger.info(f"Matched files: {matches}")

            # Step 3: pick best file (prefer rRNArm)
            rna_file = self._pick_rna_file(matches)
            logger.info(f"Downloading: {rna_file}")

            # Find the file entry in study_files to get remote_url
            study_files = files["studies"][osd_key]["study_files"]
            remote_url = None
            for file_entry in study_files:
                if file_entry.get("file_name") == rna_file:
                    remote_url = file_entry.get("remote_url")
                    break

            if not remote_url:
                return data_service_pb2.ValidationResult(
                    is_valid=False,
                    errors=[f"Could not find download URL for {rna_file}"]
                )


            # Step 4: download file content
            #download_url = NASA_DOWNLOAD_URL.format(osd_id=osd_id, filename=rna_file)
            download_url = NASA_DOWNLOAD_URL.format(remote_url=remote_url)
            logger.info(f"Download URL: {download_url}")
            rna_seq_text = self._fetch_text(download_url)

            # Step 5: parse into DataFrame
            sep = "\t" if "\t" in rna_seq_text.split("\n")[0] else ","
            df = pd.read_csv(StringIO(rna_seq_text), sep=sep)
            logger.info(f"Downloaded shape: {df.shape[0]} by {df.shape[1]}")

            # Step 6: replace nans with 0's
            df = self.check_for_nans(df)
            logger.info(f"After NaN replacement: {df.shape}")

            # Step 7: transpose df into samples x genes
            df = self.transpose_df(df)
            logger.info(f"After transpose: {df.shape[0]} samples × {df.shape[1]} genes")

            # Step 8: remove column names axis
            df.columns.name = None

            # Step 9: set dtype for entire dataframe
            df = df.astype(float)

            # Step 10: Download and add metadata
            matches = self._find_file(files, osd_key, ['metadata', 'zip'])
            if not matches:
                return data_service_pb2.ValidationResult(
                    is_valid=False,
                    errors=[f"No metadata files matched patterns ['metadata', 'zip'] in {osd_key}"]
                )
            
            meta_file = matches[0]
            # Get remote_url for metadata file
            remote_url = None
            for file_entry in study_files:
                if file_entry.get("file_name") == meta_file:
                    remote_url = file_entry.get("remote_url")
                    break

            if not remote_url:
                return data_service_pb2.ValidationResult(
                    is_valid=False,
                    errors=[f"Could not find download URL for metadata {meta_file}"]
                )

            #download_url = NASA_DOWNLOAD_URL.format(osd_id=osd_id, filename=meta_file)
            download_url = NASA_DOWNLOAD_URL.format(remote_url=remote_url)
            meta_zip_data = self._fetch_bytes(download_url)
            
            with open(meta_file, 'wb') as f:
                f.write(meta_zip_data)

            # Step 11: unzip metadata
            #dest_dir = 'TMP'
            dest_dir = '/tmp/metadata_extract'
            os.makedirs(dest_dir, exist_ok=True)
            with zipfile.ZipFile(meta_file, 'r') as zip_ref:
                zip_ref.extractall(dest_dir)

            # Step 12: read metadata
            metadata_file = 's_OSD-' + osd_id + '.txt'
            meta_path = os.path.join(dest_dir, metadata_file)
            meta = pd.read_csv(meta_path, sep='\t', header=0)

            # Check if factor_name exists in metadata
            if factor_name not in meta.columns:
                available_cols = [c for c in meta.columns if 'Factor' in c or 'factor' in c]
                return data_service_pb2.ValidationResult(
                    is_valid=False,
                    errors=[f"Factor '{factor_name}' not found in metadata. Available: {available_cols}"]
                )

            metadata = meta[['Sample Name', factor_name]]
            logger.info(f"Metadata shape: {metadata.shape}")

            # Step 13: combine condition with expression data
            '''conditions = []
            for sample in list(df.index):
                matching = metadata[metadata['Sample Name'] == sample][factor_name]
                if len(matching) > 0:
                    condition = matching.values[0]
                else:
                    condition = 'Unknown'
                conditions.append(condition)
            
            logger.info(f"Conditions: {conditions}")
            df[factor_name] = conditions'''
            
            # Step 13: combine condition with expression data
            conditions = []
            logger.info(f"Factor values for encoding: {factor_values}")

            for sample in list(df.index):
                matching = metadata[metadata['Sample Name'] == sample][factor_name]
                if len(matching) > 0:
                    condition = matching.values[0]
                else:
                    condition = 'Unknown'
    
                # Map based on factor_values
                # First value in factor_values = 1, second value = 0, everything else = -1 (or exclude)
                if len(factor_values) >= 1 and factor_values[0].lower() in condition.lower():
                    conditions.append(1)
                elif len(factor_values) >= 2 and factor_values[1].lower() in condition.lower():
                    conditions.append(0)
                else:
                    # If not in factor_values, mark as -1 or exclude
                    logger.warning(f"Sample {sample} has condition '{condition}' not in factor_values {factor_values}")
                    conditions.append(-1)  # Or you could skip this sample entirely

            logger.info(f"Encoded conditions: {conditions}")
            df[factor_name] = conditions

            # Optional: Filter out samples with -1 (unknown/excluded conditions)
            if -1 in conditions:
                df = df[df[factor_name] != -1]
                logger.info(f"After filtering unknown conditions: {df.shape}")

            # Step 14: validate and store RAW dataset
            errors = []
            warnings = []

            if df.empty:
                errors.append("Downloaded file produced an empty DataFrame")
            if len(df.columns) == 0:
                errors.append("Downloaded file has no columns")

            null_pct = df.isnull().sum().sum() / max(df.shape[0] * df.shape[1], 1)
            if null_pct > 0.5:
                warnings.append(f"Dataset has {null_pct:.1%} missing values")

            dataset_id = request.dataset_id or str(uuid.uuid4())
            
            if not errors:
                self.datasets[dataset_id] = df
                self._save_dataset_to_disk(dataset_id, df)
                
                # Add to cache
                self.download_cache[cache_key] = dataset_id
                self._save_download_cache()
                
                logger.info(f"✓ Stored RAW dataset {dataset_id}: {df.shape}")
                logger.info(f"  Columns include condition: {factor_name}")

            dataset_info = self._build_dataset_info(dataset_id, df)

            return data_service_pb2.ValidationResult(
                is_valid=len(errors) == 0,
                dataset_id=dataset_id,
                errors=errors,
                warnings=warnings,
                dataset_info=dataset_info
            )

        except Exception as e:
            msg = f"DownloadDataset failed: {e}"
            logger.error(msg, exc_info=True)
            return data_service_pb2.ValidationResult(is_valid=False, errors=[msg])


    def TransformDataset(self, request, context):
        """
        Apply CV filtering and transformations to a RAW dataset.
        Input: raw dataset (samples × genes + condition)
        Output: transformed dataset (samples × filtered_genes + condition)
        """
        try:
            dataset_id = request.dataset_id
            transformations = list(request.transformations)  # e.g., ["log", "standardize"]
            cv_step = request.cv_step or 0.25
            min_features = request.min_features or 1000
            
            logger.info(f"Transforming dataset {dataset_id}")
            logger.info(f"  Transformations: {transformations}")
            logger.info(f"  CV step: {cv_step}")
            logger.info(f"  Min features: {min_features}")
            
            # Load dataset
            if dataset_id not in self.datasets:
                dataset_file = self.dataset_path / f"{dataset_id}.parquet"
                if not dataset_file.exists():
                    return data_service_pb2.TransformResponse(
                        success=False,
                        error_message=f"Dataset {dataset_id} not found"
                    )
                df = pd.read_parquet(dataset_file)
                self.datasets[dataset_id] = df
            
            df = self.datasets[dataset_id].copy()
            logger.info(f"Original shape: {df.shape}")
            
            # Step 1: Extract condition column
            condition_columns = [col for col in df.columns if 'Factor' in col or 'Condition' in col]
            if condition_columns:
                condition_col_name = condition_columns[0]
                condition_values = df[condition_col_name].copy()
                df = df.drop(columns=condition_columns)
                logger.info(f"Extracted condition column: '{condition_col_name}'")
                logger.info(f"  Condition distribution: {condition_values.value_counts().to_dict()}")
            else:
                condition_col_name = None
                condition_values = None
                logger.info("No condition column found")
            
            # Now df is: samples × genes (all numeric)
            logger.info(f"Expression matrix: {df.shape[0]} samples × {df.shape[1]} genes")
            
            # Step 2: Apply CV filtering (assumes samples × genes)
            if cv_step > 0 and min_features > 0:
                logger.info(f"Applying CV filtering (start=1, step={cv_step}, min={min_features})...")
                df = self._filter_cvs(df, start=1, step=cv_step, min_features=min_features)
                logger.info(f"After CV filtering: {df.shape[0]} samples × {df.shape[1]} genes")
            
            # Step 3: Apply transformations
            for transform_type in transformations:
                columns = list(df.columns)  # Apply to all gene columns
                
                if transform_type == "log" or transform_type == "l":
                    logger.info("Applying log transformation...")
                    df = self.transformer.log_transform(df, columns)
                
                elif transform_type == "standardize" or transform_type == "s":
                    logger.info("Applying standardization...")
                    df = self.transformer.standardize(df, columns)
                
                elif transform_type == "normalize" or transform_type == "n":
                    logger.info("Applying normalization...")
                    df = self.transformer.normalize(df, columns)
                
                elif transform_type == "tpm" or transform_type == "t":
                    logger.info("Applying TPM transformation...")
                    df = self.transformer.tpm_transform(df, columns)
                
                elif transform_type == "one_hot_encode":
                    logger.info("Applying one-hot encoding...")
                    df = self.transformer.one_hot_encode(df, columns)
                
                else:
                    logger.warning(f"Unknown transformation: {transform_type}")
            
            # Step 4: Add condition column back
            if condition_values is not None and condition_col_name is not None:
                df[condition_col_name] = condition_values.values
                logger.info(f"✓ Added condition column '{condition_col_name}' back")
            
            logger.info(f"Final transformed shape: {df.shape}")
            
            # Step 5: Save transformed dataset
            new_id = f"{dataset_id}_transformed_{uuid.uuid4().hex[:8]}"
            self.datasets[new_id] = df
            self._save_dataset_to_disk(new_id, df)
            
            logger.info(f"✓ Created transformed dataset: {new_id}")
            
            # Build response
            dataset_info = self._build_dataset_info(new_id, df)
            
            return data_service_pb2.TransformResponse(
                success=True,
                transformed_dataset_id=new_id,
                dataset_info=dataset_info
            )
            
        except Exception as e:
            logger.error(f"TransformDataset failed: {e}", exc_info=True)
            return data_service_pb2.TransformResponse(
                success=False,
                error_message=str(e)
            ) 
    def UploadDataset(self, request, context):
        """Upload dataset (legacy - for backward compatibility)"""
        try:
            dataset_id = request.dataset_id or str(uuid.uuid4())
            
            if request.format == "csv":
                from io import BytesIO
                df = pd.read_csv(BytesIO(request.file_content))
            elif request.format == "json":
                df = pd.read_json(BytesIO(request.file_content))
            else:
                return data_service_pb2.ValidationResult(
                    is_valid=False,
                    errors=[f"Unsupported format: {request.format}"]
                )
            
            # Exclude columns if specified
            exclude_columns = list(request.exclude_columns) if request.exclude_columns else []
            if exclude_columns:
                df = df.drop(columns=exclude_columns, errors='ignore')
            
            errors = []
            warnings = []
            
            if df.empty:
                errors.append("Dataset is empty")
            if len(df.columns) == 0:
                errors.append("Dataset has no columns")
            
            if not errors:
                self.datasets[dataset_id] = df
                self._save_dataset_to_disk(dataset_id, df)
                logger.info(f"Uploaded dataset {dataset_id}: {df.shape}")
            
            dataset_info = self._build_dataset_info(dataset_id, df)
            
            return data_service_pb2.ValidationResult(
                is_valid=len(errors) == 0,
                dataset_id=dataset_id,
                errors=errors,
                warnings=warnings,
                dataset_info=dataset_info
            )
            
        except Exception as e:
            logger.error(f"UploadDataset error: {e}", exc_info=True)
            return data_service_pb2.ValidationResult(
                is_valid=False,
                errors=[str(e)]
            )
    
    def ApplyTransformation(self, request, context):
        """Apply transformations (legacy - redirects to TransformDataset)"""
        try:
            dataset_id = request.dataset_id
            
            # Convert old transformation format to new
            transform_names = []
            for t in request.transformations:
                if t.type in ["log", "standardize", "normalize"]:
                    transform_names.append(t.type)
            
            # Call new TransformDataset
            new_request = data_service_pb2.TransformRequest(
                dataset_id=dataset_id,
                transformations=transform_names,
                cv_step=0.25,
                min_features=1000
            )
            
            response = self.TransformDataset(new_request, context)
            
            # Convert response to old format
            return data_service_pb2.TransformationResult(
                success=response.success,
                transformed_dataset_id=response.transformed_dataset_id,
                error_message=response.error_message,
                transformed_info=response.dataset_info
            )
            
        except Exception as e:
            logger.error(f"ApplyTransformation error: {e}", exc_info=True)
            return data_service_pb2.TransformationResult(
                success=False,
                error_message=str(e)
            )
    
    def GetDatasetInfo(self, request, context):
        """Get dataset info"""
        try:
            dataset_id = request.dataset_id
            
            # Load from disk if not in memory
            if dataset_id not in self.datasets:
                dataset_file = self.dataset_path / f"{dataset_id}.parquet"
                if not dataset_file.exists():
                    context.abort(grpc.StatusCode.NOT_FOUND, 
                                f"Dataset {dataset_id} not found")
                
                df = pd.read_parquet(dataset_file)
                self.datasets[dataset_id] = df
            
            df = self.datasets[dataset_id]
            return self._build_dataset_info(dataset_id, df)
            
        except Exception as e:
            logger.error(f"GetDatasetInfo error: {e}", exc_info=True)
            context.abort(grpc.StatusCode.INTERNAL, str(e))
    
    '''def GetDataset(self, request, context):
        """Stream dataset back to client"""
        try:
            dataset_id = request.dataset_id
            chunk_size = request.chunk_size or (1024 * 1024)  # Default 1MB
            
            # Load from disk if not in memory
            if dataset_id not in self.datasets:
                dataset_file = self.dataset_path / f"{dataset_id}.parquet"
                if not dataset_file.exists():
                    context.abort(grpc.StatusCode.NOT_FOUND, 
                                f"Dataset {dataset_id} not found")
                
                df = pd.read_parquet(dataset_file)
                self.datasets[dataset_id] = df
            
            df = self.datasets[dataset_id]
            
            # Convert to CSV and stream in chunks
            csv_data = df.to_csv(index=True)
            total_chunks = (len(csv_data) + chunk_size - 1) // chunk_size
            
            for i in range(0, len(csv_data), chunk_size):
                chunk = csv_data[i:i + chunk_size]
                chunk_number = i // chunk_size
                is_final = (i + chunk_size >= len(csv_data))
                
                yield data_service_pb2.DataChunk(
                    data=chunk.encode('utf-8'),
                    chunk_number=chunk_number,
                    is_final=is_final
                )
        
        except Exception as e:
            logger.error(f"GetDataset error: {e}")
            context.abort(grpc.StatusCode.INTERNAL, str(e))'''

    def GetDataset(self, request, context):
        """Stream dataset back to client"""
        try:
            dataset_id = request.dataset_id
            chunk_size = request.chunk_size or 1000  # Default 1000 rows (not bytes!)
        
            # Load from disk if not in memory
            if dataset_id not in self.datasets:
                dataset_file = self.dataset_path / f"{dataset_id}.parquet"
                if not dataset_file.exists():
                    context.abort(grpc.StatusCode.NOT_FOUND, 
                                f"Dataset {dataset_id} not found")
            
                df = pd.read_parquet(dataset_file)
                self.datasets[dataset_id] = df
        
            df = self.datasets[dataset_id]
        
            # Stream by ROWS, not bytes
            num_chunks = (len(df) + chunk_size - 1) // chunk_size
            logger.info(f"Streaming {len(df)} rows in {num_chunks} chunks of {chunk_size} rows")
        
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, len(df))
                chunk_df = df.iloc[start_idx:end_idx]
            
                # Convert chunk to CSV
                csv_data = chunk_df.to_csv(index=True)
            
                yield data_service_pb2.DataChunk(
                    data=csv_data.encode('utf-8'),
                    chunk_number=i,
                    is_final=(i == num_chunks - 1)
                )
        
            logger.info(f"Finished streaming dataset {dataset_id}")
    
        except Exception as e:
            logger.error(f"GetDataset error: {e}")
            context.abort(grpc.StatusCode.INTERNAL, str(e))
    
    def StreamDataset(self, request, context):
        """Stream dataset (alias for GetDataset)"""
        return self.GetDataset(request, context)
