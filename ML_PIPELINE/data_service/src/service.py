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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NASA_FILES_URL  = "https://visualization.osdr.nasa.gov/biodata/api/v2/dataset/OSD-{osd_id}/files/"
NASA_DOWNLOAD_URL = "https://osdr.nasa.gov/geode-py/ws/studies/OSD-{osd_id}/download?source=datamanager&file={filename}"


class DataServiceImpl(data_service_pb2_grpc.DataServiceServicer):
    """gRPC service implementation for data operations"""

    def __init__(self):
        # In-memory storage (in production, use Redis/S3/database)
        self.datasets: Dict[str, pd.DataFrame] = {}
        self.transformer = DataTransformer()

    # ------------------------------------------------------------------
    # NASA OSDR helpers
    # ------------------------------------------------------------------

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

    def _filter_cvs(self, df, thresh):
        # calculate coefficient of variation
        # assumes samples x genes
        keep_columns = list()
        for col in list(df.columns):
            m = np.mean(df[col])
            sd = np.std(df[col])
            if m != 0 and sd/m > thresh:
               keep_columns.append(col)
        return df[keep_columns]
   

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

    def DownloadDataset(self, request, context):
        """
        Download an RNA-seq counts file from NASA OSDR, store it as a
        dataset, and return a ValidationResult just like ValidateDataset.
        """
        osd_id   = request.osd_id
        patterns = list(request.patterns) or ["Unnormalized", "RSEM"]
        osd_key  = f"OSD-{osd_id}"

        try:
            # Step 1: fetch file listing
            logger.info(f"Fetching file list for {osd_key}")
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

            # check for nans
            nan_count = df.isna().sum().sum()
            logger.info(f"nan count: {nan_count}")
            if nan_count > 0:
                df = df.fillna(0)


            # Step 5b: transform df into samples x genes
            dft = df.T
            dft.columns = dft.iloc[0]
            dft.drop(dft.index[0], inplace=True)
            dft.rename_axis('sample', inplace=True)
            df = dft
            logger.info(df.head())

            # remove name of columns
            df.columns.name = None

            # set dtype for entire dataframe
            df = df.astype(float)

            # filter low CVS
            df = self._filter_cvs(df, 2)
            logger.info(f"shape after filter_cvs: {df.shape}")


            # check type of data
            #data_type = df['ENSMUSG00002076992'].dtype
            #logger.info(f"dtype of gene col: {data_type}")

            # Step 5c: filter out genes
            

            # Step 5c: download metadata
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
            metadata = meta[['Sample Name', 'Factor Value[Spaceflight]']]
            logger.info(metadata)

            # Step 5g: combine condition with expr
            conditions = list()
            for sample in df.index: 
                condition = metadata[metadata['Sample Name'] == sample]['Factor Value[Spaceflight]'].values[0] 
                if 'flight' in condition.lower():
                    conditions.append(1)
                else:
                    conditions.append(0) 
            logger.info(conditions)
            df['condition'] = conditions

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
                logger.info(f"✓ Stored downloaded dataset {dataset_id} ({len(df)} rows) from {rna_file}")
                logger.info(f"  Total datasets in memory: {len(self.datasets)}")

            dataset_info = self._build_dataset_info(dataset_id, df)

            return data_service_pb2.ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                info=dataset_info
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

    def ValidateDataset(self, request, context):
        """Validate dataset and return metadata"""
        try:
            if request.format == "csv":
                df = pd.read_csv(BytesIO(request.dataset_content))
            elif request.format == "json":
                df = pd.read_json(BytesIO(request.dataset_content))
            else:
                return data_service_pb2.ValidationResult(
                    is_valid=False,
                    errors=[f"Unsupported format: {request.format}"]
                )

            errors = []
            warnings = []

            if df.empty:
                errors.append("Dataset is empty")
            if len(df.columns) == 0:
                errors.append("Dataset has no columns")

            null_percentage = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
            if null_percentage > 0.5:
                warnings.append(f"Dataset has {null_percentage:.1%} missing values")

            dataset_id = request.dataset_id or str(uuid.uuid4())
            if not errors:
                self.datasets[dataset_id] = df
                logger.info(f"✓ Stored dataset {dataset_id} ({len(df)} rows)")
                logger.info(f"  Total datasets in memory: {len(self.datasets)}")

            dataset_info = self._build_dataset_info(dataset_id, df)

            return data_service_pb2.ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                info=dataset_info
            )

        except Exception as e:
            return data_service_pb2.ValidationResult(
                is_valid=False,
                errors=[f"Failed to parse dataset: {str(e)}"]
            )

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
                    df = self.transformer.tpm(df, columns)
                else:
                    return data_service_pb2.TransformationResult(
                        success=False,
                        error_message=f"Unknown transformation: {transform.type}"
                    )

            new_id = f"{request.dataset_id}_transformed_{uuid.uuid4().hex[:8]}"
            self.datasets[new_id] = df
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
            num_rows=len(df),
            num_columns=len(df.columns),
            columns=columns_info,
            size_bytes=df.memory_usage(deep=True).sum()
        )
