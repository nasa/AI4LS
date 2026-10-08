
# Summary 
This software represents a dockerized microservice implementation of a complete pipeline that may be used to leverage machine learning for analyzing transcriptomic data from OSDR.

## Clone the repo.
1. Clone this github repository to your local system. 

```console
git clone https://github.com/nasa/AI4LS
```

2. Change directory to the `AI4LS` directory. 

```console
cd AI4LS 
```

3. Checkout the `mlpipe` branch.

```console
git checkout mlpipe
```

4. Create a conda environment with Python version 3.10.

```console
conda create -n mlpipe python=3.10 -c conda-forge --override-channels
```

5. Activate the environment. 

```console
conda activate mlpipe 
```

6. Install the Python requirements. 

```console
pip install -r requirements.txt
```

## Install docker (if not already installed)

For Mac users, follow [these steps](https://docs.docker.com/desktop/setup/install/mac-install/).

For Windows users, follow [these steps](https://docs.docker.com/desktop/setup/install/windows-install/)

For Linux users, follow [these steps](https://docs.docker.com/desktop/setup/install/linux/)

Make sure the Docker Desktop service is running 

```console
docker image list
```

## Create the microservice docker containers

1. Change directory to the `ML_PIPELINE` directory.

```console
cd ML_PIPELINE
```

2. Build the microservice containers

```console
docker-compose build
```

3. Verify that the images are built.

```console
docker image list
```

4. Start the microservice containers in daemon mode.

```console
docker-compose up -d
```

5. Verify the containers are running.

```console
docker container list
```

6. View the log files from the containers.

```console
docker-compose logs -f
```

## Run a classification algorithm against OSD-48 

1. Open another terminal window.

2. Change directory to the ML_PIPELINE directory

```console
cd AI4LS/ML_PIPELINE
```
3. Run the `run_docker_pipeline.py` script against the data in OSD-48 to do classification using the random_forest algorithm.

```console
python run_docker_pipeline.py --operation=download --osd_id=48 --target_column='Factor Value[Spaceflight]'  --task_type=classification --algorithm=random_forest --test_size=0.2 --trans_list=t,s --fi_methods=built_in,rfe -pv 0.05 -qv 0.05 -fc 1 --dgea=True
```
4. Check the results directory for the CSV and PNG files 
 
```console
ls -R results/
```

5. Run the `run_docker_pipeline.py` script against the data in OSD-137 to do classification using the logistic_regression algorithm.

```console
python run_docker_pipeline.py --operation=download --osd_id=137 --target_column='Factor Value[Spaceflight]'  --task_type=classification --algorithm=logistic_regression --test_size=0.2 --trans_list=l,t --fi_methods=pfi,rfe --patterns=unnormalized -pv 0.05 -qv 0.05 --dgea=True

6. Check the results directory for the CSV and PNG files 
 
```console
ls -R results/
```


## Run a classification algorithm against an uploaded data file. 

1. Open another terminal window.

2. Change directory to the ML_PIPELINE directory

```console
cd AI4LS/ML_PIPELINE
```

3. Run the `run_docker_pipeline.py` script to upload a CSV file and do classification using the neural_network algorithm.

```console
python run_docker_pipeline.py   -op upload   -tt classification   -al neural_network -if DATA/X_hne_class_nosample_100.csv -tc "Factor Value[Spaceflight]" -ec sample -sc sample -pv 0.05 -qv 0.05 -fi rfe --dgea=True 
```

4. Check the results directory for the CSV and PNG files 

```console
ls -R results/
```

# Manage artifacts

## List all artifacts

```console
python utils/cleanup.py list experiments
python utils/cleanup.py list models
python utils/cleanup.py list datasets
```

## Preview deletions (dry-run)

```console
python utils/cleanup.py delete-experiment exp_abc123 --dry-run
python utils/cleanup.py delete-model model_f4d97fe38159 --dry-run
python utils/cleanup.py delete-dataset 65a3ddc6-b4e8 --dry-run
```

## Actually delete

```console
python utils/cleanup.py delete-experiment exp_abc123
python utils/cleanup.py delete-model model_f4d97fe38159
python utils/cleanup.py delete-dataset 65a3ddc6-b4e8
python utils/cleanup.py delete-importance model_f4d97fe38159
python utils/cleanup.py delete-kegg --analysis-id model_61abe71dee68
```

# multi-dataset experiments
## Combine all muscle tissue datasets
```console
python new_multi_pipeline.py --tissue muscle
```

## Other available tissues: bone, liver, kidney, heart, brain, blood, skin, intestine, lung
```console
python new_multi_pipeline.py --tissue brain
```

## Combine specific datasets
```console
python new_multi_pipeline.py --osd-ids "OSD-48,OSD-51,OSD-71"
```

## Single dataset still works
```console
python new_multi_pipeline.py --osd-ids "OSD-48"
```

## Customize the pipeline
```console
python new_multi_pipeline.py \
  --tissue muscle \
  --task_type classification \
  --algorithm random_forest \
  --test_size 0.2 \
  --cv_step 0.25 \
  --min_features 1000 \
  --fi_methods "built_in,permutation"
```

## Custom factor values
```console
python new_multi_pipeline.py \
  --osd-ids "OSD-48,OSD-51" \
  --factor_name "Factor Value[Spaceflight]" \
  --factor_values "Ground Control,Space Flight" \
  --algorithm logistic_regression
```

# Full pipeline with all steps
python multi_dataset_pipeline_full.py --tissue liver

# With custom transformations
python multi_dataset_pipeline_full.py \
  --tissue liver \
  --trans_list "t,l,s" \
  --cv_step 0.25 \
  --min_features 1000

# Skip transformations (raw filtered data)
python multi_dataset_pipeline_full.py \
  --tissue liver \
  --trans_list ""

# Skip specific steps
python multi_dataset_pipeline_full.py \
  --tissue liver \
  --no-ensemble \
  --no-kegg


# ML Bioinformatics Pipeline

A distributed, containerized machine learning pipeline for bioinformatics data analysis. Built with microservices, gRPC, and Docker Compose for scalability and ease of deployment.

## Features

- **Multi-dataset support**: Download, combine, and analyze multiple datasets simultaneously
- **CV-based feature filtering**: Automatically select top features by coefficient of variation
- **Ensemble learning**: Train multiple algorithms in parallel with consensus predictions
- **Feature importance**: Compute feature importance using permutation and recursive methods
- **Streaming gRPC**: Efficient data transfer between microservices
- **Interactive Jupyter frontend**: Analyze results and visualize metrics
- **Reproducible results**: Full configuration control via JSON payloads
- **Docker containerization**: One-command setup and deployment

## Architecture

### Microservices

| Service | Port | Purpose |
|---------|------|---------|
| **data_service** | 50051 | Dataset management, combining, filtering, transformations |
| **ml_service** | 50052 | Model training, ensemble learning, LOOCV metrics |
| **feature_importance_service** | 50053 | Feature importance computation |
| **bioinformatics_service** | 50054 | Bioinformatics-specific analysis (DESeq2, KEGG) |
| **orchestration_service** | 8000 | HTTP API, pipeline orchestration, request handling |

### Data Flow

```
User (Jupyter/CLI)
        ↓
Orchestration Service (HTTP)
        ↓
Data Service (Download → Combine → Filter by CV)
        ↓
ML Service (Train Ensemble → LOOCV → Metrics)
        ↓
Feature Importance Service (Compute Importance)
        ↓
Results JSON/CSV/PDF Report
```

## Quick Start

### Prerequisites

- Docker & Docker Compose (v1.29+)
- Python 3.10+ (for local development)
- 8GB+ RAM recommended

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd ML_PIPELINE
```

2. **Start all services**
```bash
docker-compose up --build -d
```

3. **Verify services are healthy**
```bash
docker-compose ps
```

All services should show `(healthy)` status.

4. **Access the Jupyter notebook**
```bash
# The notebook is at: ./ML_Pipeline_Frontend.ipynb
# Copy it to your preferred location and open in Jupyter
jupyter notebook ML_Pipeline_Frontend.ipynb
```

## Usage

### Via Jupyter Notebook

The `ML_Pipeline_Frontend.ipynb` provides an interactive interface:

1. **Load the notebook**
```python
# In the first cell, import and configure
import requests
import json
import pandas as pd

ORCHESTRATION_URL = "http://localhost:8000"
```

2. **Define your pipeline configuration**
```python
payload = {
    "osd_ids": "47,48,137,168",  # Comma-separated dataset IDs
    "config": {
        "target_column": "Factor Value[Spaceflight]",
        "task_type": "classification",
        "ensemble_algorithms": [
            "random_forest",
            "logistic_regression",
            "xgboost",
            "svm"
        ],
        "trans_list": "s,l",  # Standardization, Log transformation
        "test_size": 0.2,
        "min_features": 100,  # CV-filter to top 100 genes
        "random_state": 42,
        "metrics": ["accuracy", "precision", "recall", "f1_score"],
        "fi_methods": ["permutation", "recursive"],
        "factor_name": "Factor Value[Spaceflight]",
        "factor_values": ["Ground Control", "Space Flight"]
    }
}
```

3. **Run the pipeline**
```python
response = requests.post(
    f"{ORCHESTRATION_URL}/api/pipeline/run",
    json=payload,
    stream=True
)

# Process streaming results
for line in response.iter_lines():
    if line:
        result = json.loads(line)
        print(f"Status: {result['status']}")
        print(f"Progress: {result['progress_percent']}%")
```

4. **Analyze results**
```python
# Training metrics
print(result['training_results']['test_metrics'])

# Feature importance
for model_id, importance in result['feature_importance'].items():
    print(f"\n{model_id}:")
    for method, data in importance.items():
        print(f"  {method}: {len(data['features'])} features")
```

### Via Command Line / Python Script

```python
import requests
import json

def run_pipeline(osd_ids, target_column, algorithms, min_features=100):
    """Run ML pipeline via orchestration service"""
    
    payload = {
        "osd_ids": osd_ids,
        "config": {
            "target_column": target_column,
            "task_type": "classification",
            "ensemble_algorithms": algorithms,
            "trans_list": "s,l",
            "test_size": 0.2,
            "min_features": min_features,
            "random_state": 42,
            "metrics": ["accuracy", "precision", "recall", "f1_score"],
            "fi_methods": ["permutation", "recursive"],
            "factor_name": target_column,
            "factor_values": ["Ground Control", "Space Flight"]
        }
    }
    
    response = requests.post(
        "http://localhost:8000/api/pipeline/run",
        json=payload,
        stream=True
    )
    
    results = []
    for line in response.iter_lines():
        if line:
            results.append(json.loads(line))
    
    return results[-1]  # Return final result

# Run example
result = run_pipeline(
    osd_ids="47,48,137,168",
    target_column="Factor Value[Spaceflight]",
    algorithms=["random_forest", "logistic_regression", "xgboost", "svm"],
    min_features=100
)

print(json.dumps(result, indent=2))
```

## Configuration

### Dataset IDs (Tissue Registry)

Available tissue types and their dataset IDs:

```python
TISSUE_REGISTRY = {
    'muscle': ['48', '51', '71', '97'],
    'bone': ['179', '180', '181'],
    'liver': ['47', '48', '137', '168', '463', '379', '245', '173'],
    'kidney': ['123', '124'],
    'heart': ['142', '143'],
    'brain': ['165', '166'],
    'blood': ['200', '201'],
    'skin': ['220', '221'],
    'intestine': ['250', '251'],
    'lung': ['275', '276'],
}
```

### Pipeline Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `osd_ids` | string | Required | Comma-separated dataset IDs |
| `target_column` | string | Required | Target variable for prediction |
| `task_type` | string | "classification" | "classification" or "regression" |
| `ensemble_algorithms` | list | Required | ML algorithms to train |
| `trans_list` | string | "" | Transformations: "s"=standardize, "l"=log |
| `test_size` | float | 0.2 | Train/test split ratio |
| `min_features` | int | 1000 | Top N features to select by CV |
| `random_state` | int | 42 | Random seed for reproducibility |
| `metrics` | list | ["accuracy"] | Metrics to compute |
| `fi_methods` | list | ["permutation"] | Feature importance methods |

### Supported Algorithms

- `random_forest` - Random Forest (classification & regression)
- `logistic_regression` - Logistic Regression (classification)
- `gradient_boosting` - Gradient Boosting (classification & regression)
- `xgboost` - XGBoost (classification & regression)
- `svm` - Support Vector Machine (classification)
- `neural_network` - Multi-layer Perceptron (classification)
- `linear_regression` - Linear Regression (regression)
- `ridge_regression` - Ridge Regression (regression)

### Feature Importance Methods

- `permutation` - Permutation feature importance
- `recursive` - Recursive feature elimination

## Output

### JSON Response

```json
{
  "pipeline_id": "unique-pipeline-id",
  "status": "completed",
  "message": "Pipeline completed successfully",
  "progress_percent": 100,
  "config": {
    "osd_ids": "47,48",
    "algorithms": ["random_forest", "logistic_regression"],
    "target_column": "Factor Value[Spaceflight]",
    "test_size": 0.2,
    "min_features": 100
  },
  "training_results": {
    "status": "completed",
    "message": "Ensemble training completed: 2 models",
    "test_metrics": {
      "num_models": 2,
      "models": [
        {
          "model_id": "model_abc123",
          "algorithm": "random_forest",
          "accuracy": 0.95,
          "precision": 0.94,
          "recall": 0.96,
          "f1_score": 0.95
        }
      ],
      "mean_accuracy": 0.925
    }
  },
  "feature_importance": {
    "model_abc123": {
      "permutation": {
        "features": [
          {
            "feature_name": "gene_001",
            "importance": 0.125,
            "rank": 1
          }
        ]
      }
    }
  },
  "transformed_dataset_id": "filtered_xxxxxxxx"
}
```

## Development

### Project Structure

```
ML_PIPELINE/
├── data_service/
│   ├── src/
│   │   ├── service.py          # DataService implementation
│   │   ├── model_store.py      # Model persistence
│   │   └── generated/          # Protobuf generated files
│   ├── proto/
│   │   └── data_service.proto  # Service definition
│   └── Dockerfile
├── ml_service/
│   ├── src/
│   │   ├── service.py          # ML training logic
│   │   ├── trainers.py         # Model training
│   │   ├── data_client.py      # Data service client
│   │   └── generated/
│   ├── proto/
│   │   └── ml_service.proto
│   └── Dockerfile
├── feature_importance_service/
│   ├── src/
│   │   ├── service.py
│   │   ├── importance_methods.py
│   │   └── generated/
│   ├── proto/
│   │   └── feature_importance_service.proto
│   └── Dockerfile
├── orchestration_service/
│   ├── src/
│   │   ├── main.py             # HTTP API & orchestration
│   │   ├── config.py           # Configuration
│   │   ├── models.py           # Pydantic models
│   │   ├── clients/            # Service clients
│   │   └── generated/
│   └── Dockerfile
├── docker-compose.yml
└── ML_Pipeline_Frontend.ipynb
```

### Running Locally (Without Docker)

For development, you can run services individually:

```bash
# Terminal 1: Data Service
cd data_service
python -m src.server

# Terminal 2: ML Service
cd ml_service
python -m src.server

# Terminal 3: Feature Importance Service
cd feature_importance_service
python -m src.server

# Terminal 4: Orchestration Service
cd orchestration_service
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

### Adding New Algorithms

1. Implement in `ml_service/src/trainers.py`
2. Add to `VALID_ALGORITHMS` in `orchestration_service/src/main.py`
3. Add proto definition if needed
4. Test via Jupyter notebook

### Testing

```bash
# Run tests for a service
cd ml_service
pytest tests/

# Check service health
curl http://localhost:8000/health
curl http://localhost:50051/health  # Data service (gRPC)
```

## Troubleshooting

### Services won't start

```bash
# Check logs
docker-compose logs ml-pipeline-orchestration

# Rebuild from scratch
docker-compose down -v
docker-compose up --build -d
```

### Dataset not found errors

```bash
# Verify datasets exist
docker exec ml-pipeline-data_service ls -la /app/datasets/

# Check dataset IDs in tissue registry
# Make sure osd_ids are valid
```

### Out of memory errors

```bash
# Increase Docker memory limit
# Or reduce min_features parameter
# Or use smaller datasets
```

### gRPC connection refused

```bash
# Ensure service names resolve correctly
# Use service_name:port (not localhost) in Docker

# Verify all services are running
docker-compose ps

# Check network
docker network ls
```

### Feature importance fails

```bash
# Check that ml_service completed training first
# Verify dataset_id exists in data_service
# Check feature_importance_service logs
docker-compose logs ml-pipeline-feature-importance
```

## Performance Tips

1. **Feature filtering**: Use `min_features` to reduce dimensionality
2. **Dataset size**: Start with smaller datasets, scale up gradually
3. **Algorithms**: Random Forest is faster than XGBoost for large datasets
4. **Parallel training**: Ensemble naturally parallelizes across algorithms
5. **Memory**: Monitor with `docker stats`

## API Endpoints

### Orchestration Service

```
POST /api/pipeline/run
  - Run full ML pipeline
  - Request: JSON payload with osd_ids, config
  - Response: Streaming NDJSON with progress updates

GET /api/models
  - List all trained models
  - Query params: algorithm, task_type, limit

GET /api/models/{model_id}
  - Get model details and metrics

GET /health
  - Health check
```

## Future Enhancements

- [ ] DESeq2 integration for RNA-seq analysis
- [ ] KEGG enrichment analysis
- [ ] Web dashboard for results visualization
- [ ] Model serving/prediction API
- [ ] Hyperparameter optimization (Optuna)
- [ ] Cross-validation improvements
- [ ] Database backend (PostgreSQL) instead of in-memory
- [ ] Distributed training (Ray/Dask)
- [ ] Model versioning and registry

## Known Limitations

- In-memory dataset storage (restart loses data)
- Single-machine deployment (no distributed computation yet)
- Limited hyperparameter tuning
- No model persistence across restarts (models saved to disk but not auto-loaded)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{ml_pipeline_2024,
  title={ML Bioinformatics Pipeline},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/ML_PIPELINE}
}
```

## Contact & Support

- **Issues**: Create a GitHub issue for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Email**: your-email@example.com

## Acknowledgments

- Built with gRPC, FastAPI, scikit-learn, XGBoost
- Inspired by modern ML ops practices
- Thanks to the bioinformatics community for feedback

---

**Last Updated**: October 2024  
**Version**: 1.0.0
