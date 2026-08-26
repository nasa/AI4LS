#!/usr/bin/env python3
"""
DataServiceClient - Updated for Merged Multi-Dataset Service

The multi_dataset_service definitions are merged into data_service.proto,
so we import from data_service_pb2, not a separate multi_dataset_service_pb2
"""

import grpc
import logging
from typing import List, Dict, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataServiceClient:
    """Client for data service with multi-dataset support"""
    
    def __init__(self, host='localhost', port=50051, service_url=None):
        """
        Initialize data service client
        
        Args:
            host: Service host (default localhost)
            port: Service port (default 50051)
            service_url: Full service URL (e.g., "localhost:50051")
        """
        if service_url:
            self.service_url = service_url
        else:
            if ':' in host:
                self.service_url = host
            else:
                self.service_url = f"{host}:{port}"
        
        # Import here to avoid issues if protobuf files not yet generated
        try:
            #from data_service.generated import data_service_pb2, data_service_pb2_grpc
            from generated import data_service_pb2, data_service_pb2_grpc
            self.data_service_pb2 = data_service_pb2
            self.data_service_pb2_grpc = data_service_pb2_grpc
        except ImportError as e:
            logger.error(f"Failed to import protobuf modules: {e}")
            logger.error("Make sure to regenerate data_service.proto:")
            logger.error("  python regenerate_data_service_proto.py")
            raise
        
        self.channel = grpc.insecure_channel(self.service_url)
        self.multi_stub = self.data_service_pb2_grpc.MultiDatasetServiceStub(self.channel)
    
    # ============================================================================
    # MULTI-DATASET SERVICE METHODS
    # ============================================================================
    
    def get_osd_ids_for_tissue(self, tissue_name: str) -> List[str]:
        """
        Get OSD IDs for a tissue type
        
        Args:
            tissue_name: Name of tissue (e.g., "liver", "muscle")
        
        Returns:
            List of OSD IDs for that tissue
        """
        try:
            request = self.data_service_pb2.GetOSDIDsRequest(tissue_name=tissue_name)
            response = self.multi_stub.GetOSDIDsForTissue(request)
            
            if not response.success:
                raise Exception(response.error_message)
            
            logger.info(f"Found {len(response.osd_ids)} OSD IDs for {tissue_name}")
            return list(response.osd_ids)
        except Exception as e:
            logger.error(f"Error getting OSD IDs for tissue {tissue_name}: {e}")
            raise
    
    def download_multiple_datasets(
        self,
        osd_ids: List[str],
        patterns: List[str] = None,
        factor_name: str = None,
        factor_values: List[str] = None,
        exclude_columns: List[str] = None,
        min_features: int = 1000,
        cv_step: float = 0.25
    ) -> Dict[str, str]:
        """
        Download multiple datasets
        
        Args:
            osd_ids: List of OSD IDs to download
            patterns: Optional list of patterns to filter samples
            factor_name: Name of factor for filtering
            factor_values: List of factor values to include
            exclude_columns: List of columns to exclude
            min_features: Minimum features after CV filtering
            cv_step: CV filtering threshold
        
        Returns:
            Dict mapping OSD ID -> dataset_id
        """
        try:
            if patterns is None:
                patterns = []
            if factor_values is None:
                factor_values = []
            if exclude_columns is None:
                exclude_columns = []
            
            request = self.data_service_pb2.DownloadMultipleDatasetsRequest(
                osd_ids=osd_ids,
                patterns=patterns,
                factor_name=factor_name or "",
                factor_values=factor_values,
                exclude_columns=exclude_columns,
                min_features=min_features,
                cv_step=cv_step
            )
            response = self.multi_stub.DownloadMultipleDatasets(request)
            
            if not response.success:
                raise Exception(response.error_message)
            
            logger.info(f"Downloaded {len(response.dataset_ids)} datasets")
            return dict(response.dataset_ids)
        except Exception as e:
            logger.error(f"Error downloading multiple datasets: {e}")
            raise
    
    def find_common_genes(self, dataset_ids: List[str]) -> List[str]:
        """
        Find genes common to all datasets
        
        Args:
            dataset_ids: List of dataset IDs to analyze
        
        Returns:
            List of common gene names
        """
        try:
            request = self.data_service_pb2.FindCommonGenesRequest(
                dataset_ids=dataset_ids
            )
            response = self.multi_stub.FindCommonGenes(request)
            
            if not response.success:
                raise Exception(response.error_message)
            
            logger.info(f"Found {response.count} common genes")
            return list(response.common_genes)
        except Exception as e:
            logger.error(f"Error finding common genes: {e}")
            raise
    
    def combine_datasets(
        self,
        dataset_ids: List[str],
        common_genes: List[str] = None,
        output_name: str = None
    ) -> Tuple[str, Dict[str, int], str]:
        """
        Combine multiple datasets into one
        
        Args:
            dataset_ids: List of dataset IDs to combine
            common_genes: Optional list of genes to keep (if None, will compute)
            output_name: Optional name for combined dataset
        
        Returns:
            Tuple of (combined_dataset_id, samples_per_source, condition_column)
        """
        try:
            if common_genes is None:
                common_genes = []
            
            logger.info(f"here is the number of common genes {len(common_genes)}") 
            logger.info(f"here is the list of dataset ids {dataset_ids}") 
            logger.info(f"here is the output name {output_name}") 
            request = self.data_service_pb2.CombineDatasetsRequest(
                dataset_ids=dataset_ids,
                common_genes=common_genes,
                output_name=output_name or ""
            )
            logger.info(f"here is the request for self.data_service_pb2.CombineDatasetsRequest {request}") 
            # JC this is the call that is failing
            response = self.multi_stub.CombineDatasets(request)
            logger.info(f"here is the response from self.multi_stub.CombineDatasets {response}") 
            if not response.success:
                raise Exception(response.error_message)
            
            logger.info(f"Combined dataset created: {response.combined_dataset_id}")
            logger.info(f"  Total samples: {response.total_samples}")
            logger.info(f"  Total genes: {response.total_genes}")
            
            return response.combined_dataset_id, dict(response.samples_per_source), response.condition_column
        except Exception as e:
            logger.error(f"Error combining datasets: {e}")
            raise
    
    def combine_by_tissue(
        self,
        tissue_name: str,
        patterns: List[str] = None,
        factor_name: str = None,
        factor_values: List[str] = None,
        min_features: int = 1000,
        cv_step: float = 0.25,
        output_name: str = None
    ) -> Tuple[str, Dict[str, int], str]:
        """
        Combine all datasets for a tissue in one call (convenience method)
        
        Args:
            tissue_name: Name of tissue to combine
            patterns: Optional patterns to filter samples
            factor_name: Name of factor for filtering
            factor_values: List of factor values to include
            min_features: Minimum features after CV filtering
            cv_step: CV filtering threshold
            output_name: Optional name for combined dataset
        
        Returns:
            Tuple of (combined_dataset_id, samples_per_source, condition_column)
        """
        try:
            if patterns is None:
                patterns = []
            if factor_values is None:
                factor_values = []
            
            request = self.data_service_pb2.CombineByTissueRequest(
                tissue_name=tissue_name,
                patterns=patterns,
                factor_name=factor_name or "",
                factor_values=factor_values,
                min_features=min_features,
                cv_step=cv_step,
                output_name=output_name or ""
            )
            response = self.multi_stub.CombineByTissue(request)
            
            if not response.success:
                raise Exception(response.error_message)
            
            logger.info(f"Combined {tissue_name} datasets: {response.combined_dataset_id}")
            logger.info(f"  Total samples: {response.total_samples}")
            logger.info(f"  Total genes: {response.total_genes}")
            
            return response.combined_dataset_id, dict(response.samples_per_source), response.condition_column
        except Exception as e:
            logger.error(f"Error combining by tissue: {e}")
            raise
    
    def close(self):
        """Close the channel"""
        self.channel.close()

    def get_dataset(self, dataset_id):
        """Load a dataset from disk"""
        try:
            from pathlib import Path
            import pandas as pd

            dataset_path = Path("./datasets") / f"{dataset_id}.parquet"

            if not dataset_path.exists():
                raise FileNotFoundError(f"Dataset not found: {dataset_path}")

            df = pd.read_parquet(dataset_path)
            logger.info(f"✓ Loaded dataset {dataset_id}: {df.shape[0]} samples × {df.shape[1]} features")

            return df

        except Exception as e:
            logger.error(f"Error loading dataset {dataset_id}: {e}")
            raise


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    # Initialize client
    client = DataServiceClient('localhost', 50051)
    
    # Example 1: Get OSD IDs for liver tissue
    try:
        liver_osds = client.get_osd_ids_for_tissue("liver")
        print(f"Liver datasets: {liver_osds}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 2: Combine liver datasets in one call
    try:
        combined_id, samples_per_source, condition_col = client.combine_by_tissue(
            tissue_name="liver",
            factor_name="Factor Value[Spaceflight]",
            factor_values=["Ground Control", "Space Flight"],
            min_features=500
        )
        print(f"Combined dataset: {combined_id}")
        print(f"Samples per source: {samples_per_source}")
    except Exception as e:
        print(f"Error: {e}")
    
    client.close()
