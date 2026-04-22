# src/service.py
import grpc
import pandas as pd
import numpy as np
import requests
from io import StringIO, BytesIO
import uuid
from typing import Dict, List
from generated import data_service_pb2, data_service_pb2_grpc
from src.transformations import DataTransformer
import logging
import os
import zipfile
from pathlib import Path
import json
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NASA_FILES_URL  = "https://visualization.osdr.nasa.gov/biodata/api/v2/dataset/OSD-{osd_id}/files/"
NASA_DOWNLOAD_URL = "https://osdr.nasa.gov/geode-py/ws/studies/OSD-{osd_id}/download?source=datamanager&file={filename}"


class DataServiceImpl(data_service_pb2_grpc.DataServiceServicer):
    """gRPC service implementation for data operations"""

    def __init__(self):
        self.datasets: Dict[str, pd.DataFrame] = {}
        self.transformer = DataTransformer()
        
        # Dataset persistence
        self.dataset_path = Path("/app/datasets")
        self.dataset_path.mkdir(parents=True, exist_ok=True)

        # Cache mapping: stores download parameters -> dataset_id
        self.download_cache_file = self.dataset_path / "download_cache.json"
        self.download_cache = self._load_download_cache()
        
        # Load existing datasets from disk
        self._load_datasets_from_disk()
        
        logger.info("DataServiceImpl initialized")
        logger.info(f"Dataset storage path: {self.dataset_path}")

    # ------------------------------------------------------------------
    # NASA OSDR helpers
    # ------------------------------------------------------------------

    
    def _load_download_cache(self) -> Dict:
        """Load download cache from disk"""
        try:
            if self.download_cache_file.exists():
                with open(self.download_cache_file, 'r') as f:
                    cache = json.load(f)
                logger.info(f"Loaded download cache: {len(cache)} entries")
                return cache
            return {}
        except Exception as e:
            logger.error(f"Error loading download cache: {e}")
            return {}
    
    def _save_download_cache(self):
        """Save download cache to disk"""
        try:
            with open(self.download_cache_file, 'w') as f:
                json.dump(self.download_cache, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving download cache: {e}")

    
    def _generate_cache_key(self, osd_id: str, patterns: list, factor_name: str, 
                           factor_values: list, exclude_columns: list, min_features: int, cv_step: float) -> str:
        """
        Generate a unique cache key based on download parameters
        
        This ensures that the same dataset with the same filters will reuse the cached version
        """
        cache_dict = {
            "osd_id": osd_id,
            "patterns": sorted(patterns),  # Sort for consistency
            "factor_name": factor_name or "",
            "factor_values": sorted(factor_values) if factor_values else [],
            "exclude_columns": sorted(exclude_columns) if exclude_columns else [],
            "min_features": min_features,
            "cv_step": cv_step
        }
        
        # Create a hash of the parameters
        cache_str = json.dumps(cache_dict, sort_keys=True)
        cache_hash = hashlib.md5(cache_str.encode()).hexdigest()
        
        return cache_hash

    def _load_datasets_from_disk(self):
        """Load all datasets from disk on startup"""
        try:
            parquet_files = list(self.dataset_path.glob("*.parquet"))
            logger.info(f"Found {len(parquet_files)} datasets on disk")
            
            for file in parquet_files:
                try:
                    dataset_id = file.stem
                    df = pd.read_parquet(file)
                    self.datasets[dataset_id] = df
                    logger.info(f"✓ Loaded dataset {dataset_id}: {df.shape}")
                except Exception as e:
                    logger.error(f"✗ Failed to load {file.name}: {e}")
            
            logger.info("in _load_datasets_from_disk()")
            logger.info(f"Total datasets in memory: {len(self.datasets)}")
            
        except Exception as e:
            logger.error(f"Error loading datasets from disk: {e}")
    
    def _save_dataset_to_disk(self, dataset_id: str, df: pd.DataFrame):
        """Save dataset to disk as parquet file"""
        try:
            file_path = self.dataset_path / f"{dataset_id}.parquet"
            df.to_parquet(file_path, index=False)
            logger.info(f"✓ Saved dataset {dataset_id} to {file_path} ({df.shape})")
        except Exception as e:
            logger.error(f"✗ Error saving dataset {dataset_id}: {e}")
    
    def _fetch_json(self, url: str):
        """GET a URL and return parsed JSON, or raise on error."""
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        return response.json()

    def _fetch_text(self, url: str) -> str:
        """GET a URL and return response text, or raise on error."""
        response = requests.get(url, timeout=300)
        response.raise_for_status()
        return response.text

    def _fetch_bytes(self, url: str) -> str:
        """GET a URL and return response bytes, or raise on error."""
        response = requests.get(url, timeout=300)
        response.raise_for_status()
        return response.content



    def _find_file(self, files: dict, osd_key: str, patterns: List[str]) -> List[str]:
        """Return filenames whose paths match ALL patterns (case-insensitive)."""
        return [
            f for f in files[osd_key]["files"]
            if all(p.lower() in f.lower() for p in patterns)
        ]

    def _pick_rna_file(self, matches: List[str]) -> str:
        """Prefer the rRNArm file; fall back to the first match."""
        for f in matches:
            if "rRNArm" in f:
                return f
        return matches[0]

    # ------------------------------------------------------------------
    # DownloadDataset RPC
    # ------------------------------------------------------------------
    def _custom_mutual_info_regression(X, y):
        from sklearn.feature_selection import  mutual_info_regression
        return mutual_info_regression(X, y, random_state=seed)

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


    def _filterNotCorrelated(self, df, y, k, seed):
        # remove non-correlated
        from sklearn.feature_selection import SelectKBest, mutual_info_regression
        if k == 0:
            return df
        X = df.to_numpy()
        selector = SelectKBest(score_func=_custom_mutual_info_regression, k=k)
        X_selected = selector.fit_transform(X, y)
        # get indices of remaining cols
        indices = selector.get_support(indices=True)
        return df.iloc[indices]

    def check_for_nans(self, df):
        # check for nans
        nan_count = df.isna().sum().sum()
        logger.info(f"nan count: {nan_count}")
        if nan_count > 0:
            df = df.fillna(0)
        return df

    def transpose_df(self, df):
        # Step 5b: transpose df into samples x genes
        dft = df.T
        dft.columns = dft.iloc[0]
        dft.drop(dft.index[0], inplace=True)
        dft.rename_axis('sample', inplace=True)
        return dft 

    def DownloadDataset(self, request, context):
        """
        Download an RNA-seq counts file from NASA OSDR, store it as a
        dataset, and return a ValidationResult just like ValidateDataset.
        """
        logger.info(f"request in DownloadDataset: {request}")
        try:
            osd_id   = request.osd_id
            patterns = list(request.patterns) or ["Unnormalized", "RSEM"]
            osd_key  = f"OSD-{osd_id}"
            factor_name = request.factor_name
            factor_values = list(request.factor_values)
            min_features = request.min_features
            exclude_columns = list(request.exclude_columns)
            cv_step = float(request.cv_step)

            # Generate cache key based on download parameters
            cache_key = self._generate_cache_key(
                osd_id, patterns, factor_name, factor_values, exclude_columns, min_features, cv_step
            )
        
            # Check if we've already downloaded this exact dataset
            if cache_key in self.download_cache:
                cached_dataset_id = self.download_cache[cache_key]
            
                # Check if the cached dataset still exists
                cached_file = self.dataset_path / f"{cached_dataset_id}.parquet"
                if cached_file.exists():
                    logger.info(f"✓ Found cached dataset for OSD-{osd_id}: {cached_dataset_id}")
                    logger.info(f"  Cache key: {cache_key}")
                    logger.info(f"  Reusing existing dataset instead of re-downloading")
                
                    # Load from disk if not in memory
                    if cached_dataset_id not in self.datasets:
                        df = pd.read_parquet(cached_file)
                        self.datasets[cached_dataset_id] = df
                        logger.info(f"  Loaded cached dataset into memory: {df.shape}")
                        logger.info(f"  Loaded cached dataset into memory: {df.head()}")
                    logger.info("right before reading in df from cache") 
                    df = self.datasets[cached_dataset_id]
                    logger.info("right after reading in df from cache") 
                    # Verify df is valid before building info
                    if df is None or df.empty:
                        logger.error("Cached dataset is None or empty")
                        context.abort(grpc.StatusCode.INTERNAL, "Cached dataset is invalid")
                
                    # build dataset info
                    dataset_info=self._build_dataset_info(cached_dataset_id, df)
                    logger.info("just built the dataset info")
                    # Return the cached dataset
                    logger.info("returning cached dataset")
                    return data_service_pb2.ValidationResult(
                        is_valid=True,
                        dataset_id=cached_dataset_id,
                        errors=[],
                        warnings=["Using cached dataset from previous download"],
                        dataset_info=dataset_info
                    )
                else:
                    # Cache entry exists but file is gone - remove from cache
                    logger.warning(f"Cached dataset file not found, will re-download")
                    del self.download_cache[cache_key]
                    self._save_download_cache()
        
            # Not in cache or cache invalid - proceed with download
            logger.info(f"Downloading OSD-{osd_id} (cache key: {cache_key})")

            # Step 1: fetch file listing
            files = self._fetch_json(NASA_FILES_URL.format(osd_id=osd_id))

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

            # Step 4: download file content
            download_url = NASA_DOWNLOAD_URL.format(osd_id=osd_id, filename=rna_file)
            rna_seq_text = self._fetch_text(download_url)

            # Step 5: parse into DataFrame (tab-separated counts files are common)
            sep = "\t" if "\t" in rna_seq_text.split("\n")[0] else ","
            df = pd.read_csv(StringIO(rna_seq_text), sep=sep)
            logger.info(f"Downloaded to shape: {df.shape[0]} by {df.shape[1]}")

            # replace nans with 0's
            df = self.check_for_nans(df)
            logger.info(f"shape after replace nans: {df.shape[0]} by {df.shape[1]}")

            # transpose df into samples x genes
            df = self.transpose_df(df)
            logger.info(f"shape after transpose: {df.shape[0]} by {df.shape[1]}")

            # remove name of columns
            df.columns.name = None


            # set dtype for entire dataframe
            df = df.astype(float)

            # filter low CVS
            logger.info(f"using cv_step: {cv_step}")
            logger.info(f"shape before filter_cvs: {df.shape}")
            #df = self._filter_cvs(df, start=1, step=0.25, min_features=min_features)
            df = self._filter_cvs(df, start=1, step=cv_step, min_features=min_features)
            logger.info(f"shape after filter_cvs: {df.shape}")


            # Step 5c: find metadata
            matches = self._find_file(files, osd_key, ['metadata', 'zip'])
            if not matches:
                return data_service_pb2.ValidationResult(
                    is_valid=False,
                    errors=[f"No files matched patterns ['metadata', 'zip'] in {osd_key}"]
                )
            logger.info(f"Matched files: {matches}")
            meta_file = matches[0]

            # Step 5d: download metadata
            download_url = NASA_DOWNLOAD_URL.format(osd_id=osd_id, filename=meta_file)
            meta_zip_data = self._fetch_bytes(download_url)
            with open(meta_file, 'wb') as f:
                f.write(meta_zip_data)
            f.close()

            # STEP 5e: unzip metadata file
            dest_dir='TMP'
            os.makedirs(dest_dir, exist_ok=True)
            with zipfile.ZipFile(meta_file, 'r') as zip_ref:
                zip_ref.extractall(dest_dir) 

            # STEP 5f: read in metadata
            metadata_file = 's_OSD-' + osd_id + '.txt'
            meta = pd.read_csv(dest_dir + '/' + metadata_file, sep='\t', header=0) 
            logger.info(meta.head())
            #metadata = meta[['Sample Name', 'Factor Value[Spaceflight]']]
            metadata = meta[['Sample Name', factor_name]] 
            logger.info(metadata)

            # Step 5g: combine condition with expr
            conditions = list()
            logger.info(f"samples include: {list(df.index)}")
            for sample in list(df.index): 
                logger.info(f"examing sample: {sample}")
                condition = metadata[metadata['Sample Name'] == sample][factor_name].values[0] 
                # TODO add factor_values here
                if 'flight' in condition.lower():
                    logger.info(f"appending {condition.lower()} to 1")
                    conditions.append(1)
                else:
                    logger.info(f"appending {condition.lower()} to 0")
                    conditions.append(0) 
            logger.info(f"conditions are: {conditions}")
            df[factor_name] = conditions

            # Step 5h: remove metadata
            #os.removedirs(dest_dir)

            # Step 6: validate and store
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
                self._save_dataset_to_disk(dataset_id, df)  # ADD THIS
        
                # Add to download cache
                self.download_cache[cache_key] = dataset_id
                self._save_download_cache()
        
                logger.info("in DownloadDatasets()")
                logger.info(f"✓ Downloaded and cached OSD-{osd_id} as {dataset_id}")
                logger.info(f"  Future downloads with same parameters will reuse this dataset")
        
                logger.info(f"✓ Stored downloaded dataset {dataset_id} ({len(df)} rows) from {rna_file}")
                logger.info(f"  Total datasets in memory: {len(self.datasets)}")

            dataset_info = self._build_dataset_info(dataset_id, df)

            return data_service_pb2.ValidationResult(
                is_valid=len(errors) == 0,
                dataset_id=dataset_id,
                errors=errors,
                warnings=warnings,
                dataset_info=dataset_info
            )

        except requests.HTTPError as e:
            msg = f"HTTP error downloading from NASA OSDR: {e}"
            logger.error(msg)
            return data_service_pb2.ValidationResult(is_valid=False, errors=[msg])
        except KeyError as e:
            msg = f"Unexpected file-listing structure (missing key {e}) for {osd_key}"
            logger.error(msg)
            return data_service_pb2.ValidationResult(is_valid=False, errors=[msg])
        except Exception as e:
            msg = f"DownloadDataset failed: {e}"
            logger.error(msg, exc_info=True)
            return data_service_pb2.ValidationResult(is_valid=False, errors=[msg])

    # ------------------------------------------------------------------
    # Existing RPCs (unchanged)
    # ------------------------------------------------------------------

    def UploadDataset(self, request, context):
        """Accept raw file bytes, parse, store and return metadata"""
        try:
            # Get exclude_columns from request
            exclude_columns = list(request.exclude_columns) if request.exclude_columns else []

            cv_step = float(request.cv_step) if request.cv_step else 0.25
        
            if request.format == "csv":
                df = pd.read_csv(BytesIO(request.file_content))
            elif request.format == "json":
                df = pd.read_json(BytesIO(request.file_content))
            else:
                return data_service_pb2.ValidationResult(
                    is_valid=False,
                    errors=[f"Unsupported format: {request.format}"]
                )

            # replace nans with 0's
            df = self.check_for_nans(df)

            # remove sample column from df
            for col in exclude_columns:
                if col in list(df.columns):
                    df.drop(columns=[col], inplace=True)

            # set dtype for entire dataframe
            df = df.astype(float)

            errors = []
            warnings = []

            if df.empty:
                errors.append("Dataset is empty")
            if len(df.columns) == 0:
                errors.append("Dataset has no columns")
            logger.warn(f"errors: {errors}")
            null_percentage = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
            if null_percentage > 0.5:
                warnings.append(f"Dataset has {null_percentage:.1%} missing values")

            dataset_id = request.dataset_id or str(uuid.uuid4())
            if not errors:
                self.datasets[dataset_id] = df
                self._save_dataset_to_disk(dataset_id, df)  # ADD THIS
                logger.info("in UploadDataset()")
                logger.info(f"Uploaded dataset {dataset_id} ({len(df)} rows)")
                logger.info(f"  Total datasets in memory: {len(self.datasets)}")

            logger.info("building dataset info")
            dataset_info = self._build_dataset_info(dataset_id, df)
            logger.info("dataset info built")
            logger.info(f"length of errors = {len(errors)}")

            return data_service_pb2.ValidationResult(
                is_valid=len(errors) == 0,
                dataset_id = dataset_id,
                errors=errors,
                warnings=warnings,
                dataset_info=dataset_info
            )

        except Exception as e:
            logger.critical(f"encountered exception str({e})")
            return data_service_pb2.ValidationResult(
                is_valid=False,
                dataset_id = dataset_id,
                errors=[f"Failed to upload dataset: {str(e)}"],
                warnings=warnings,
                dataset_info=dataset_info
            )


    def ValidateDataset(self, request, context):
        """Validate dataset and return metadata"""
        try:
            dataset_id = request.dataset_id or str(uuid.uuid4())

            if request.format == "csv":
                df = pd.read_csv(BytesIO(request.dataset_content))
            elif request.format == "json":
                df = pd.read_json(BytesIO(request.dataset_content))
            else:
                return data_service_pb2.ValidationResult(
                    is_valid=False,
                    errors=[f"Unsupported format: {request.format}"]
                )
                '''context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"Unsupported format: {request.format}"
                )'''

     
            # Exclude columns if specified
            exclude_columns = list(request.exclude_columns) if request.exclude_columns else []
            if exclude_columns:
                df = df.drop(columns=exclude_columns, errors='ignore')
                logger.info(f"Excluded columns: {exclude_columns}")
        

            errors = []
            warnings = []

            if df.empty:
                errors.append("Dataset is empty")
            if len(df.columns) == 0:
                errors.append("Dataset has no columns")

            null_percentage = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
            if null_percentage > 0.5:
                warnings.append(f"Dataset has {null_percentage:.1%} missing values")

            if not errors:
                self.datasets[dataset_id] = df
                self._save_dataset_to_disk(dataset_id, df)  # ADD THIS

                logger.info("in ValidateDataset()")
                logger.info(f"✓ Stored dataset {dataset_id} ({len(df)} rows)")
                logger.info(f"  Total datasets in memory: {len(self.datasets)}")

            # Build dataset info
            dataset_info = self._build_dataset_info(dataset_id, df)
        
            return data_service_pb2.ValidationResult(
                is_valid=is_valid,
                dataset_id=dataset_id if is_valid else "",
                errors=errors,
                warnings=warnings,
                dataset_info=dataset_info
            )

        except Exception as e:
            logger.error(f"Validation error: {e}", exc_info=True)
            context.abort(grpc.StatusCode.INTERNAL, str(e))

    def ApplyTransformation(self, request, context):
        """Apply transformations to dataset"""
        try:
            if request.dataset_id not in self.datasets:
                return data_service_pb2.TransformationResult(
                    success=False,
                    error_message=f"Dataset not found: {request.dataset_id}"
                )

            df = self.datasets[request.dataset_id].copy()

            for transform in request.transformations:
                columns = list(transform.columns)

                if transform.type == "log":
                    df = self.transformer.log_transform(df, columns)
                elif transform.type == "standardize":
                    df = self.transformer.standardize(df, columns)
                elif transform.type == "normalize":
                    df = self.transformer.normalize(df, columns)
                elif transform.type == "one_hot_encode":
                    df = self.transformer.one_hot_encode(df, columns)
                elif transform.type == "tpm":
                    df = self.transformer.tpm_transform(df, columns)
                else:
                    return data_service_pb2.TransformationResult(
                        success=False,
                        error_message=f"Unknown transformation: {transform.type}"
                    )

            new_id = f"{request.dataset_id}_transformed_{uuid.uuid4().hex[:8]}"
            self.datasets[new_id] = df
            self._save_dataset_to_disk(new_id, df)  # ADD THIS

            logger.info(f"StreamDataset called for: {request.dataset_id}")
            logger.info(f"Available datasets: {list(self.datasets.keys())}")

            dataset_info = self._build_dataset_info(new_id, df)

            return data_service_pb2.TransformationResult(
                transformed_dataset_id=new_id,
                success=True,
                transformed_info=dataset_info
            )

        except Exception as e:
            return data_service_pb2.TransformationResult(
                success=False,
                error_message=f"Transformation failed: {str(e)}"
            )

    def StreamDataset(self, request, context):
        """Stream dataset in chunks"""
        try:
            if request.dataset_id not in self.datasets:
                logger.error(f"Dataset not found: {request.dataset_id}")
                logger.error(f"Available datasets: {list(self.datasets.keys())}")
                context.abort(
                    grpc.StatusCode.NOT_FOUND,
                    f"Dataset {request.dataset_id} not found. Available: {len(self.datasets)} datasets"
                )
                return

            df = self.datasets[request.dataset_id]
            chunk_size = request.chunk_size or 1000
            num_chunks = (len(df) + chunk_size - 1) // chunk_size
            logger.info(f"Streaming dataset {request.dataset_id}: {len(df)} rows in {num_chunks} chunks")

            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, len(df))
                chunk_df = df.iloc[start_idx:end_idx]

                csv_buffer = StringIO()
                chunk_df.to_csv(csv_buffer, index=False)
                csv_bytes = csv_buffer.getvalue().encode("utf-8")

                yield data_service_pb2.DataChunk(
                    chunk_number=i,
                    data=csv_bytes,
                    is_final=(i == num_chunks - 1)
                )
            logger.info(f"Finished streaming dataset {request.dataset_id}")

        except Exception as e:
            logger.error(f"Streaming failed: {e}", exc_info=True)
            context.abort(grpc.StatusCode.INTERNAL, f"Streaming failed: {str(e)}")

    def GetDatasetInfo(self, request, context):
        """Get metadata about a dataset"""
        if request.dataset_id not in self.datasets:
            logger.error(f"Dataset not found: {request.dataset_id}")
            logger.error(f"Available datasets: {list(self.datasets.keys())}")
            context.abort(grpc.StatusCode.NOT_FOUND, "Dataset not found",
                          f"Dataset not found. Available datasets: {list(self.datasets.keys())}")

        df = self.datasets[request.dataset_id]
        return self._build_dataset_info(request.dataset_id, df)

    def _build_dataset_info(self, dataset_id: str, df: pd.DataFrame):
        """Helper to build DatasetInfo message"""
        columns_info = []

        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                dtype = "numeric"
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                dtype = "datetime"
            elif pd.api.types.is_categorical_dtype(df[col]) or df[col].dtype == "object":
                dtype = "categorical"
            else:
                dtype = "text"

            sample_values = df[col].dropna().astype(str).head(3).tolist()

            columns_info.append(data_service_pb2.ColumnInfo(
                name=col,
                dtype=dtype,
                null_count=int(df[col].isnull().sum()),
                sample_values=sample_values
            ))

        return data_service_pb2.DatasetInfo(
            dataset_id=dataset_id,
            num_rows=int(len(df)),
            num_columns=int(len(df.columns)),
            columns=columns_info,
            size_bytes=int(df.memory_usage(deep=True).sum())
        )
