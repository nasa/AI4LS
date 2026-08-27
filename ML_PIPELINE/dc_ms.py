#!/usr/bin/env python3
"""
Data Service Client for Multi-Service Architecture

Use this in any service that needs to call the data service.
Assumes protobuf files are in ./generated/ directory.

Example usage:
    client = DataServiceClient("localhost:50051")
    dataset = client.download_dataset("47", patterns=["unnormalized", "RSEM"])
"""

import grpc
import logging
from pathlib import Path
import sys

# Import protobuf files from local generated directory
from generated import data_service_pb2, data_service_pb2_grpc

logger = logging.getLogger(__name__)


class DataServiceClient:
    """Client for communicating with Data Service"""
    
    def __init__(self, service_url="localhost:50051"):
        """
        Initialize data service client
        
        Args:
            service_url: Address of data service (host:port)
        """
        self.service_url = service_url
        self.channel = None
        self.stub = None
        self.multi_stub = None
        self._connect()
    
    def _connect(self):
        """Connect to data service"""
        try:
            logger.info(f"Connecting to Data Service at {self.service_url}...")
            
            self.channel = grpc.aio.secure_channel(
                self.service_url,
                grpc.ssl_channel_credentials()
            ) if self.service_url.endswith(":50052") else grpc.insecure_channel(
                self.service_url
            )
            
            # Use insecure channel for local development
            self.channel = grpc.insecure_channel(self.service_url)
            
            self.stub = data_service_pb2_grpc.DataServiceStub(self.channel)
            self.multi_stub = data_service_pb2_grpc.MultiDatasetServiceStub(self.channel)
            
            logger.info(f"✓ Connected to Data Service")
        
        except Exception as e:
            logger.error(f"✗ Failed to connect to Data Service: {e}")
            raise
    
    def download_dataset(self, osd_id, dataset_id=None, patterns=None, 
                        factor_name=None, factor_values=None, 
                        exclude_columns=None, min_features=1000, cv_step=0.25):
        """
        Download a single dataset from data service
        
        Args:
            osd_id: OSD ID
            dataset_id: Dataset ID (if different from OSD ID)
            patterns: List of patterns to match in filenames
            factor_name: Factor column name
            factor_values: List of factor values to include
            exclude_columns: List of columns to exclude
            min_features: Minimum number of features
            cv_step: Cross-validation step size
        
        Returns:
            Response object with is_valid, dataset_info, etc.
        """
        try:
            if patterns is None:
                patterns = []
            if factor_values is None:
                factor_values = []
            if exclude_columns is None:
                exclude_columns = []
            
            request = data_service_pb2.DownloadDatasetRequest(
                osd_id=str(osd_id),
                dataset_id=str(dataset_id or osd_id),
                patterns=patterns,
                factor_name=factor_name or "",
                factor_values=factor_values,
                exclude_columns=exclude_columns,
                min_features=min_features,
                cv_step=cv_step
            )
            
            logger.info(f"Downloading dataset {osd_id}...")
            response = self.stub.DownloadDataset(request)
            
            logger.info(f"✓ Download response: {response.is_valid}")
            return response
        
        except Exception as e:
            logger.error(f"Error downloading dataset: {e}")
            raise
    
    def download_multiple_datasets(self, osd_ids, patterns=None, factor_name=None,
                                  factor_values=None, exclude_columns=None,
                                  min_features=1000, cv_step=0.25):
        """
        Download multiple datasets at once
        
        Args:
            osd_ids: List of OSD IDs
            patterns: List of patterns to match
            factor_name: Factor column name
            factor_values: List of factor values
            exclude_columns: List of columns to exclude
            min_features: Minimum features
            cv_step: CV step size
        
        Returns:
            Dictionary mapping OSD ID to dataset ID
        """
        try:
            if patterns is None:
                patterns = []
            if factor_values is None:
                factor_values = []
            if exclude_columns is None:
                exclude_columns = []
            
            request = data_service_pb2.DownloadMultipleDatasetsRequest(
                osd_ids=osd_ids,
                patterns=patterns,
                factor_name=factor_name or "",
                factor_values=factor_values,
                exclude_columns=exclude_columns,
                min_features=min_features,
                cv_step=cv_step
            )
            
            logger.info(f"Downloading {len(osd_ids)} datasets...")
            response = self.multi_stub.DownloadMultipleDatasets(request)
            
            # Parse response into dictionary
            dataset_map = {}
            for osd_id, dataset_id in response.dataset_ids.items():
                dataset_map[osd_id] = dataset_id
            
            logger.info(f"✓ Downloaded {len(dataset_map)} datasets")
            return dataset_map
        
        except Exception as e:
            logger.error(f"Error downloading multiple datasets: {e}")
            raise
    
    def find_common_genes(self, dataset_ids):
        """
        Find genes common across multiple datasets
        
        Args:
            dataset_ids: List of dataset IDs to compare
        
        Returns:
            List of common gene names
        """
        try:
            request = data_service_pb2.FindCommonGenesRequest(
                dataset_ids=dataset_ids
            )
            
            logger.info(f"Finding common genes across {len(dataset_ids)} datasets...")
            response = self.stub.FindCommonGenes(request)
            
            if response.error_message:
                logger.error(f"Error finding common genes: {response.error_message}")
                raise Exception(response.error_message)
            
            logger.info(f"✓ Found {len(response.common_genes)} common genes")
            return list(response.common_genes)
        
        except Exception as e:
            logger.error(f"Error finding common genes: {e}")
            raise
    
    def get_available_datasets(self):
        """Get list of available datasets on data service"""
        try:
            request = data_service_pb2.GetAvailableDatasetsRequest()
            response = self.stub.GetAvailableDatasets(request)
            
            datasets = []
            for dataset_id, count in response.dataset_counts.items():
                datasets.append({
                    'dataset_id': dataset_id,
                    'sample_count': count
                })
            
            logger.info(f"✓ Found {len(datasets)} available datasets")
            return datasets
        
        except Exception as e:
            logger.error(f"Error getting available datasets: {e}")
            raise
    
    def close(self):
        """Close connection to data service"""
        if self.channel:
            self.channel.close()
            logger.info("✓ Data service connection closed")


if __name__ == "__main__":
    # Test the client
    logging.basicConfig(level=logging.INFO)
    
    client = DataServiceClient("localhost:50051")
    
    try:
        # List available datasets
        datasets = client.get_available_datasets()
        print(f"\nAvailable datasets: {len(datasets)}")
        for ds in datasets[:5]:
            print(f"  - {ds['dataset_id']}: {ds['sample_count']} samples")
    
    finally:
        client.close()
