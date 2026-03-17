#!/bin/bash

# Build and start all services
docker-compose up --build

# Test the data service directly
curl -X POST http://localhost:8000/api/validate-dataset \
  -F "file=@sample_data.csv"

# Run full pipeline
curl -X POST http://localhost:8000/api/pipeline/dataset-123 \
  -H "Content-Type: application/json" \
  -d '{
    "transformations": [
      {"type": "log", "columns": ["price"]},
      {"type": "standardize", "columns": ["price", "quantity"]}
    ],
    "algorithm": "random_forest",
    "metrics": ["accuracy", "f1_score"]
  }'
