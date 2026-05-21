# ml-service/src/data_client.py
import grpc
import pandas as pd
from io import BytesIO
from io import StringIO
import sys
from pathlib import Path
import logging

# Add path to data-service generated code
data_service_path = Path(__file__).parent.parent.parent / "data-service" / "generated"
sys.path.insert(0, str(data_service_path))

from generated import data_service_pb2, data_service_pb2_grpc
from generated.data_service_pb2 import StreamDatasetRequest
from generated.data_service_pb2_grpc import DataServiceStub

logger = logging.getLogger(__name__)

class DataServiceClient:
    """Client to fetch datasets from Data Service"""
    
    def __init__(self, service_url: str = "data-service:50051"):
        self.service_url = service_url
        #self.channel = grpc.insecure_channel(service_url)
        self.channel = grpc.insecure_channel(
            self.service_url,
            options=[
                ('grpc.max_send_message_length', 50 * 1024 * 1024),
                ('grpc.max_receive_message_length', 50 * 1024 * 1024),
             ]
        )
        self.stub = DataServiceStub(self.channel)
        logger.info(f"Connected to Data Service at {service_url}")
    
    def get_dataset(self, dataset_id: str) -> pd.DataFrame:
     """Fetch a dataset by streaming it from Data Service"""
     try:
         logger.info(f"Fetching dataset {dataset_id} from Data Service...")

         request = StreamDatasetRequest(
             dataset_id=dataset_id,
             chunk_size=10000
         )

         frames = []
         for chunk in self.stub.StreamDataset(request):
             chunk_df = pd.read_csv(StringIO(chunk.data.decode("utf-8")), index_col=0)
             frames.append(chunk_df)

         if not frames:
             logger.error(f"No data received for dataset {dataset_id}")
             return None

         df = pd.concat(frames, ignore_index=False)
         logger.info(f"Dataset {dataset_id} loaded: {len(df)} rows, {len(df.columns)} columns")
         return df

     except grpc.RpcError as e:
         logger.error(f"gRPC error fetching dataset: {e.code()} - {e.details()}")
         return None
     except Exception as e:
         logger.error(f"Error fetching dataset: {e}")
         return None 

     def close(self):
        """Close the gRPC channel"""
        self.channel.close()
