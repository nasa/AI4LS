#!/bin/bash

# Health check
curl http://localhost:8000/health

# Upload and validate dataset
curl -X POST http://localhost:8000/api/datasets/validate \
  -F "file=@/Users/jcasalet/Desktop/NASA/FOUNDATION_MODEL/ML_PIPELINE/MICRO/data-service/sample_data.csv"

# Get dataset info
dataset_id="e7c72e2c-0ec4-4d75-9836-78f9ac48bd43"
curl http://localhost:8000/api/datasets/{dataset_id}

# Transform dataset
curl -X POST http://localhost:8000/api/datasets/{dataset_id}/transform \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": dataset_id, 
    "transformations": [
      {
        "type": "log",
        "columns": ["income"],
        "params": {}
      },
      {
        "type": "standardize",
        "columns": ["income", "age"],
        "params": {}
      }
    ]
  }'

# Run full pipeline
curl -X POST http://localhost:8000/api/pipeline/run \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": dataset_id, 
    "config": {
      "transformations": [
        {
          "type": "standardize",
          "columns": ["age", "income"],
          "params": {}
        }
      ],
      "algorithm": "random_forest",
      "hyperparameters": {},
      "metrics": ["accuracy", "f1_score"],
      "test_size": 0.2
    }
  }'
