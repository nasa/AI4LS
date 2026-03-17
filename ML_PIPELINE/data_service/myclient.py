# Example client usage
import asyncio
from data_client import DataServiceClient

async def main():
    client = DataServiceClient()
    
    # Read dataset
    with open("sample_data.csv", "rb") as f:
        content = f.read()
    
    # Validate
    validation = await client.validate_dataset(content, "csv")
    print(f"Valid: {validation['is_valid']}")
    print(f"Rows: {validation['num_rows']}, Columns: {validation['num_columns']}")
    
    if validation['is_valid']:
        dataset_id = validation['dataset_id']
        
        # Apply transformations
        result = await client.apply_transformations(
            dataset_id=dataset_id,
            transformations=[
                {"type": "log", "columns": ["income", "age"]},
                {"type": "standardize", "columns": ["income", "age"]},
                {"type": "one_hot_encode", "columns": ["category"]}
            ]
        )
        
        print(f"Transformed ID: {result['transformed_dataset_id']}")
        print(f"New shape: {result['num_rows']} x {result['num_columns']}")
    
    client.close()

if __name__ == "__main__":
    asyncio.run(main())
