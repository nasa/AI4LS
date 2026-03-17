# test_integration.py
"""
Integration tests for the ML Pipeline services.

Requires all services to be running:
    docker-compose up -d

Usage:
    pytest test_integration.py -v
    pytest test_integration.py -v -k "data_service"
    pytest test_integration.py -v -k "orchestration"
"""

import pytest
import grpc
import requests
import pandas as pd
import io
import uuid
import json

from data_service.generated import data_service_pb2, data_service_pb2_grpc
from ml_service.generated import ml_service_pb2, ml_service_pb2_grpc

# ── Connection config ─────────────────────────────────────────────────────────

GRPC_HOST      = "localhost:50051"
ML_GRPC_HOST   = "localhost:50052"
REST_BASE_URL  = "http://localhost:8000"

GRPC_OPTIONS = [
    ('grpc.max_send_message_length',    50 * 1024 * 1024),
    ('grpc.max_receive_message_length', 50 * 1024 * 1024),
]

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def grpc_stub():
    """gRPC stub connected directly to data-service."""
    channel = grpc.insecure_channel(GRPC_HOST, options=GRPC_OPTIONS)
    yield data_service_pb2_grpc.DataServiceStub(channel)
    channel.close()


@pytest.fixture(scope="session")
def small_csv_bytes():
    """Minimal valid CSV with numeric + categorical columns."""
    df = pd.DataFrame({
        "sample":    ["s1", "s2", "s3", "s4", "s5"],
        "gene_a":    [1.0,  2.0,  3.0,  4.0,  5.0],
        "gene_b":    [5.0,  4.0,  3.0,  2.0,  1.0],
        "condition": ["A",  "B",  "A",  "B",  "A"],
    })
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


@pytest.fixture(scope="session")
def regression_csv_bytes():
    """CSV suitable for regression (numeric target)."""
    df = pd.DataFrame({
        "sample":  [f"s{i}" for i in range(20)],
        "gene_a":  [float(i)       for i in range(20)],
        "gene_b":  [float(20 - i)  for i in range(20)],
        "score":   [float(i) * 1.5 for i in range(20)],
    })
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


@pytest.fixture(scope="session")
def validated_dataset_id(grpc_stub, small_csv_bytes):
    """Upload a small CSV via gRPC and return its dataset_id."""
    request = data_service_pb2.ValidateRequest(
        dataset_content=small_csv_bytes,
        format="csv",
    )
    response = grpc_stub.ValidateDataset(request)
    assert response.is_valid, f"Fixture setup failed: {list(response.errors)}"
    return response.info.dataset_id


@pytest.fixture(scope="session")
def rest_dataset_id(small_csv_bytes):
    """Upload a small CSV via the REST gateway and return its dataset_id."""
    response = requests.post(
        f"{REST_BASE_URL}/api/datasets/validate",
        files={"file": ("data.csv", small_csv_bytes, "text/csv")},
    )
    assert response.status_code == 200
    result = response.json()
    assert result["is_valid"]
    return result["dataset_info"]["dataset_id"]


@pytest.fixture(scope="session")
def rest_regression_dataset_id(regression_csv_bytes):
    """Upload regression CSV via REST and return its dataset_id."""
    response = requests.post(
        f"{REST_BASE_URL}/api/datasets/validate",
        files={"file": ("data.csv", regression_csv_bytes, "text/csv")},
    )
    assert response.status_code == 200
    result = response.json()
    assert result["is_valid"]
    return result["dataset_info"]["dataset_id"]


# ── Data-service: ValidateDataset ─────────────────────────────────────────────

class TestGrpcValidateDataset:

    def test_valid_csv(self, grpc_stub, small_csv_bytes):
        request = data_service_pb2.ValidateRequest(
            dataset_content=small_csv_bytes,
            format="csv",
        )
        result = grpc_stub.ValidateDataset(request)
        assert result.is_valid
        assert result.info.num_rows == 5
        assert result.info.num_columns == 4
        assert not result.errors

    def test_returns_dataset_id(self, grpc_stub, small_csv_bytes):
        request = data_service_pb2.ValidateRequest(
            dataset_content=small_csv_bytes,
            format="csv",
        )
        result = grpc_stub.ValidateDataset(request)
        assert result.info.dataset_id != ""

    def test_preserves_custom_dataset_id(self, grpc_stub, small_csv_bytes):
        custom_id = f"test-{uuid.uuid4().hex[:8]}"
        request = data_service_pb2.ValidateRequest(
            dataset_content=small_csv_bytes,
            format="csv",
            dataset_id=custom_id,
        )
        result = grpc_stub.ValidateDataset(request)
        assert result.is_valid
        assert result.info.dataset_id == custom_id

    def test_empty_csv_is_invalid(self, grpc_stub):
        request = data_service_pb2.ValidateRequest(
            dataset_content=b"",
            format="csv",
        )
        result = grpc_stub.ValidateDataset(request)
        assert not result.is_valid
        assert result.errors

    def test_unsupported_format_is_invalid(self, grpc_stub, small_csv_bytes):
        request = data_service_pb2.ValidateRequest(
            dataset_content=small_csv_bytes,
            format="parquet",
        )
        result = grpc_stub.ValidateDataset(request)
        assert not result.is_valid

    def test_column_dtypes_detected(self, grpc_stub, small_csv_bytes):
        request = data_service_pb2.ValidateRequest(
            dataset_content=small_csv_bytes,
            format="csv",
        )
        result = grpc_stub.ValidateDataset(request)
        dtype_map = {col.name: col.dtype for col in result.info.columns}
        assert dtype_map["gene_a"] == "numeric"
        assert dtype_map["gene_b"] == "numeric"
        assert dtype_map["condition"] == "categorical"


# ── Data-service: ApplyTransformation ────────────────────────────────────────

class TestGrpcApplyTransformation:

    def test_log_transform(self, grpc_stub, validated_dataset_id):
        request = data_service_pb2.TransformRequest(
            dataset_id=validated_dataset_id,
            transformations=[
                data_service_pb2.Transformation(
                    type="log",
                    columns=["gene_a", "gene_b"],
                )
            ],
        )
        result = grpc_stub.ApplyTransformation(request)
        assert result.success
        assert result.transformed_dataset_id != ""

    def test_normalize_transform(self, grpc_stub, validated_dataset_id):
        request = data_service_pb2.TransformRequest(
            dataset_id=validated_dataset_id,
            transformations=[
                data_service_pb2.Transformation(
                    type="normalize",
                    columns=["gene_a", "gene_b"],
                )
            ],
        )
        result = grpc_stub.ApplyTransformation(request)
        assert result.success

    def test_standardize_transform(self, grpc_stub, validated_dataset_id):
        request = data_service_pb2.TransformRequest(
            dataset_id=validated_dataset_id,
            transformations=[
                data_service_pb2.Transformation(
                    type="standardize",
                    columns=["gene_a", "gene_b"],
                )
            ],
        )
        result = grpc_stub.ApplyTransformation(request)
        assert result.success

    def test_chained_transforms_produce_new_id(self, grpc_stub, validated_dataset_id):
        request = data_service_pb2.TransformRequest(
            dataset_id=validated_dataset_id,
            transformations=[
                data_service_pb2.Transformation(type="log",         columns=["gene_a"]),
                data_service_pb2.Transformation(type="standardize", columns=["gene_a"]),
            ],
        )
        result = grpc_stub.ApplyTransformation(request)
        assert result.success
        assert result.transformed_dataset_id != validated_dataset_id

    def test_unknown_transform_fails(self, grpc_stub, validated_dataset_id):
        request = data_service_pb2.TransformRequest(
            dataset_id=validated_dataset_id,
            transformations=[
                data_service_pb2.Transformation(type="fft", columns=["gene_a"])
            ],
        )
        result = grpc_stub.ApplyTransformation(request)
        assert not result.success
        assert result.error_message != ""

    def test_missing_dataset_id_fails(self, grpc_stub):
        request = data_service_pb2.TransformRequest(
            dataset_id="does-not-exist",
            transformations=[
                data_service_pb2.Transformation(type="log", columns=["gene_a"])
            ],
        )
        result = grpc_stub.ApplyTransformation(request)
        assert not result.success


# ── Data-service: StreamDataset ───────────────────────────────────────────────

class TestGrpcStreamDataset:

    def test_streams_all_rows(self, grpc_stub, validated_dataset_id):
        request = data_service_pb2.StreamRequest(
            dataset_id=validated_dataset_id,
            chunk_size=2,
        )
        chunks = list(grpc_stub.StreamDataset(request))
        assert len(chunks) > 0
        assert chunks[-1].is_final

        # Count data rows across all chunks (subtract header per chunk)
        total_rows = sum(
            max(len(c.data.decode("utf-8").strip().split("\n")) - 1, 0)
            for c in chunks
        )
        assert total_rows == 5

    def test_chunk_numbers_are_sequential(self, grpc_stub, validated_dataset_id):
        request = data_service_pb2.StreamRequest(
            dataset_id=validated_dataset_id,
            chunk_size=2,
        )
        chunks = list(grpc_stub.StreamDataset(request))
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_number == i

    def test_missing_dataset_raises_not_found(self, grpc_stub):
        request = data_service_pb2.StreamRequest(dataset_id="no-such-id")
        with pytest.raises(grpc.RpcError) as exc_info:
            list(grpc_stub.StreamDataset(request))
        assert exc_info.value.code() == grpc.StatusCode.NOT_FOUND


# ── Data-service: DownloadDataset ─────────────────────────────────────────────

class TestGrpcDownloadDataset:

    def test_invalid_osd_id_returns_error(self, grpc_stub):
        request = data_service_pb2.DownloadRequest(
            osd_id="999999",
            patterns=["Unnormalized", "RSEM"],
        )
        result = grpc_stub.DownloadDataset(request)
        assert not result.is_valid
        assert result.errors

    def test_no_pattern_match_returns_error(self, grpc_stub):
        request = data_service_pb2.DownloadRequest(
            osd_id="379",
            patterns=["ZZZNOMATCH"],
        )
        result = grpc_stub.DownloadDataset(request)
        assert not result.is_valid
        assert result.errors

    def test_custom_dataset_id_preserved(self, grpc_stub):
        """Uses a known-bad OSD ID — just checks the ID round-trip on error path."""
        custom_id = "my-custom-id-xyz"
        request = data_service_pb2.DownloadRequest(
            osd_id="999999",
            patterns=["Unnormalized", "RSEM"],
            dataset_id=custom_id,
        )
        result = grpc_stub.DownloadDataset(request)
        # Service returns error but should not crash
        assert not result.is_valid

    @pytest.mark.slow
    def test_real_download_osd_379(self, grpc_stub):
        """Live download from NASA OSDR — requires internet access."""
        request = data_service_pb2.DownloadRequest(
            osd_id="379",
            patterns=["Unnormalized", "RSEM"],
        )
        result = grpc_stub.DownloadDataset(request)
        assert result.is_valid, f"Download failed: {list(result.errors)}"
        assert result.info.num_rows > 0
        assert result.info.num_columns > 0
        assert "sample" in [col.name for col in result.info.columns]


# ── Orchestration: Health ─────────────────────────────────────────────────────

class TestRestHealth:

    def test_health_returns_200(self):
        response = requests.get(f"{REST_BASE_URL}/health")
        assert response.status_code == 200

    def test_health_reports_healthy(self):
        result = requests.get(f"{REST_BASE_URL}/health").json()
        assert result["status"] == "healthy"

    def test_health_includes_service_status(self):
        result = requests.get(f"{REST_BASE_URL}/health").json()
        assert "data_service" in result["services"]
        assert "ml_service"   in result["services"]
        assert result["services"]["data_service"] is True
        assert result["services"]["ml_service"]   is True


# ── Orchestration: Validate dataset ──────────────────────────────────────────

class TestRestValidateDataset:

    def test_valid_upload_returns_200(self, small_csv_bytes):
        response = requests.post(
            f"{REST_BASE_URL}/api/datasets/validate",
            files={"file": ("data.csv", small_csv_bytes, "text/csv")},
        )
        assert response.status_code == 200

    def test_valid_upload_is_valid(self, small_csv_bytes):
        result = requests.post(
            f"{REST_BASE_URL}/api/datasets/validate",
            files={"file": ("data.csv", small_csv_bytes, "text/csv")},
        ).json()
        assert result["is_valid"]
        assert result["dataset_info"]["num_rows"] == 5
        assert result["dataset_info"]["num_columns"] == 4

    def test_empty_file_is_invalid(self):
        response = requests.post(
            f"{REST_BASE_URL}/api/datasets/validate",
            files={"file": ("data.csv", b"", "text/csv")},
        )
        result = response.json()
        assert not result["is_valid"]


# ── Orchestration: Download dataset ──────────────────────────────────────────

class TestRestDownloadDataset:

    def test_invalid_osd_returns_200_with_error(self):
        response = requests.post(
            f"{REST_BASE_URL}/api/datasets/download",
            json={"osd_id": "999999", "patterns": ["Unnormalized", "RSEM"]},
        )
        assert response.status_code == 200
        result = response.json()
        assert not result["is_valid"]
        assert result["errors"]

    def test_no_pattern_match_returns_error(self):
        response = requests.post(
            f"{REST_BASE_URL}/api/datasets/download",
            json={"osd_id": "379", "patterns": ["ZZZNOMATCH"]},
        )
        assert response.status_code == 200
        result = response.json()
        assert not result["is_valid"]

    @pytest.mark.slow
    def test_real_download_osd_379(self):
        """Live download — requires internet access."""
        response = requests.post(
            f"{REST_BASE_URL}/api/datasets/download",
            json={"osd_id": "379", "patterns": ["Unnormalized", "RSEM"]},
        )
        assert response.status_code == 200
        result = response.json()
        assert result["is_valid"], f"Download failed: {result['errors']}"
        assert result["dataset_info"]["num_rows"] > 0


# ── Orchestration: Transform dataset ─────────────────────────────────────────

class TestRestTransformDataset:

    def test_log_transform(self, rest_dataset_id):
        response = requests.post(
            f"{REST_BASE_URL}/api/datasets/{rest_dataset_id}/transform",
            json={"transformations": [{"type": "log", "columns": ["gene_a", "gene_b"], "params": {}}]},
        )
        assert response.status_code == 200
        result = response.json()
        assert result["success"]
        assert result["transformed_dataset_id"]

    def test_normalize_transform(self, rest_dataset_id):
        response = requests.post(
            f"{REST_BASE_URL}/api/datasets/{rest_dataset_id}/transform",
            json={"transformations": [{"type": "normalize", "columns": ["gene_a", "gene_b"], "params": {}}]},
        )
        assert response.status_code == 200
        assert response.json()["success"]

    def test_standardize_transform(self, rest_dataset_id):
        response = requests.post(
            f"{REST_BASE_URL}/api/datasets/{rest_dataset_id}/transform",
            json={"transformations": [{"type": "standardize", "columns": ["gene_a", "gene_b"], "params": {}}]},
        )
        assert response.status_code == 200
        assert response.json()["success"]

    def test_invalid_dataset_id_fails(self):
        response = requests.post(
            f"{REST_BASE_URL}/api/datasets/no-such-id/transform",
            json={"transformations": [{"type": "log", "columns": ["gene_a"], "params": {}}]},
        )
        result = response.json()
        assert not result["success"]


# ── Orchestration: Pipeline run ───────────────────────────────────────────────

class TestRestPipelineRun:

    def _run_pipeline(self, dataset_id, target, task_type, algorithm, metrics,
                      sample_col=None, transformations=None):
        """Helper to POST to /api/pipeline/run and collect streamed lines."""
        columns_response = requests.get(f"{REST_BASE_URL}/api/datasets/{dataset_id}")
        all_columns = [c["name"] for c in columns_response.json()["columns"]]

        feature_columns = [
            c for c in all_columns
            if c not in ([target] + ([sample_col] if sample_col else []))
        ]

        payload = {
            "dataset_id": dataset_id,
            "config": {
                "target_column": target,
                "task_type": task_type,
                "feature_columns": feature_columns,
                "transformations": transformations or [],
                "algorithm": algorithm,
                "hyperparameters": {},
                "metrics": metrics,
                "test_size": 0.2,
                "random_state": 42,
            },
        }

        response = requests.post(
            f"{REST_BASE_URL}/api/pipeline/run",
            json=payload,
            stream=True,
        )
        assert response.status_code == 200

        events = []
        for line in response.iter_lines():
            if line:
                import json
                events.append(json.loads(line))
        return events

    def test_classification_random_forest(self, rest_dataset_id):
        events = self._run_pipeline(
            dataset_id=rest_dataset_id,
            target="condition",
            task_type="classification",
            algorithm="random_forest",
            metrics=["accuracy", "f1_score"],
            sample_col="sample",
        )
        final = events[-1]
        assert final["status"] == "completed"
        assert final["model_id"]

    def test_regression_ridge(self, rest_regression_dataset_id):
        events = self._run_pipeline(
            dataset_id=rest_regression_dataset_id,
            target="score",
            task_type="regression",
            algorithm="Ridge",
            metrics=["rmse", "r2_score"],
            sample_col="sample",
        )
        final = events[-1]
        assert final["status"] == "completed"
        assert final["model_id"]

    def test_pipeline_with_transformations(self, rest_dataset_id):
        events = self._run_pipeline(
            dataset_id=rest_dataset_id,
            target="condition",
            task_type="classification",
            algorithm="random_forest",
            metrics=["accuracy", "f1_score"],
            sample_col="sample",
            transformations=[
                {"type": "log",         "columns": ["gene_a", "gene_b"], "params": {}},
                {"type": "standardize", "columns": ["gene_a", "gene_b"], "params": {}},
            ],
        )
        final = events[-1]
        assert final["status"] == "completed"

    def test_pipeline_streams_progress(self, rest_dataset_id):
        events = self._run_pipeline(
            dataset_id=rest_dataset_id,
            target="condition",
            task_type="classification",
            algorithm="random_forest",
            metrics=["accuracy", "f1_score"],
            sample_col="sample",
        )
        percents = [e.get("progress_percent", 0) for e in events]
        assert percents[-1] == 100
        assert percents == sorted(percents)   # progress only goes forward

    def test_invalid_dataset_id_fails(self):
        payload = {
            "dataset_id": "no-such-id",
            "config": {
                "target_column": "condition",
                "task_type": "classification",
                "feature_columns": [],
                "transformations": [],
                "algorithm": "random_forest",
                "hyperparameters": {},
                "metrics": ["accuracy"],
                "test_size": 0.2,
                "random_state": 42,
            },
        }
        response = requests.post(
            f"{REST_BASE_URL}/api/pipeline/run",
            json=payload,
            stream=True,
        )
        import json
        events = [json.loads(l) for l in response.iter_lines() if l]
        assert any(e.get("status") == "failed" for e in events)


# ── Orchestration: Models ─────────────────────────────────────────────────────

class TestRestModels:

    @pytest.fixture(scope="class")
    def trained_model_id(self, rest_dataset_id):
        """Train a model and return its ID for model-info tests."""
        payload = {
            "dataset_id": rest_dataset_id,
            "config": {
                "target_column": "condition",
                "task_type": "classification",
                "feature_columns": ["gene_a", "gene_b"],
                "transformations": [],
                "algorithm": "random_forest",
                "hyperparameters": {},
                "metrics": ["accuracy", "f1_score"],
                "test_size": 0.2,
                "random_state": 42,
            },
        }
        import json
        response = requests.post(
            f"{REST_BASE_URL}/api/pipeline/run", json=payload, stream=True
        )
        events = [json.loads(l) for l in response.iter_lines() if l]
        final = events[-1]
        assert final["status"] == "completed"
        return final["model_id"]

    def test_list_models_returns_200(self):
        response = requests.get(f"{REST_BASE_URL}/api/models?limit=10")
        assert response.status_code == 200

    def test_list_models_has_expected_fields(self):
        result = requests.get(f"{REST_BASE_URL}/api/models?limit=10").json()
        assert "total_count" in result
        assert "models" in result

    def test_get_model_info(self, trained_model_id):
        response = requests.get(f"{REST_BASE_URL}/api/models/{trained_model_id}")
        assert response.status_code == 200
        result = response.json()
        assert result["model_id"] == trained_model_id
        assert result["algorithm"] == "random_forest"
        assert "test_metrics" in result

    def test_get_nonexistent_model_returns_404(self):
        response = requests.get(f"{REST_BASE_URL}/api/models/no-such-model")
        assert response.status_code == 404


# ── ML-service fixtures ───────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def ml_grpc_stub():
    """gRPC stub connected directly to ml-service."""
    channel = grpc.insecure_channel(ML_GRPC_HOST, options=GRPC_OPTIONS)
    yield ml_service_pb2_grpc.MLServiceStub(channel)
    channel.close()


@pytest.fixture(scope="session")
def classification_dataset_id(grpc_stub):
    """Upload a classification CSV directly to data-service and return dataset_id."""
    df = pd.DataFrame({
        "sample":    [f"s{i}" for i in range(30)],
        "gene_a":    [float(i)       for i in range(30)],
        "gene_b":    [float(30 - i)  for i in range(30)],
        "gene_c":    [float(i % 5)   for i in range(30)],
        "condition": (["A", "B"] * 15),
    })
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    request = data_service_pb2.ValidateRequest(
        dataset_content=buf.getvalue(), format="csv"
    )
    response = grpc_stub.ValidateDataset(request)
    assert response.is_valid, f"Fixture setup failed: {list(response.errors)}"
    return response.info.dataset_id


@pytest.fixture(scope="session")
def regression_dataset_id_grpc(grpc_stub):
    """Upload a regression CSV directly to data-service and return dataset_id."""
    df = pd.DataFrame({
        "sample":  [f"s{i}" for i in range(30)],
        "gene_a":  [float(i)       for i in range(30)],
        "gene_b":  [float(30 - i)  for i in range(30)],
        "score":   [float(i) * 1.5 for i in range(30)],
    })
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    request = data_service_pb2.ValidateRequest(
        dataset_content=buf.getvalue(), format="csv"
    )
    response = grpc_stub.ValidateDataset(request)
    assert response.is_valid
    return response.info.dataset_id


@pytest.fixture(scope="session")
def trained_classification_model_id(ml_grpc_stub, classification_dataset_id):
    """Train a classification model and return its model_id."""
    request = ml_service_pb2.TrainRequest(
        dataset_id=classification_dataset_id,
        algorithm="random_forest",
        task_type="classification",
        target_column="condition",
        feature_columns=["gene_a", "gene_b", "gene_c"],
        test_size=0.2,
        random_state=42,
    )
    events = list(ml_grpc_stub.TrainModel(request))
    final = events[-1]
    assert final.status == "completed", f"Training failed: {final.error_message}"
    return final.model_id


@pytest.fixture(scope="session")
def trained_regression_model_id(ml_grpc_stub, regression_dataset_id_grpc):
    """Train a regression model and return its model_id."""
    request = ml_service_pb2.TrainRequest(
        dataset_id=regression_dataset_id_grpc,
        algorithm="Ridge",
        task_type="regression",
        target_column="score",
        feature_columns=["gene_a", "gene_b"],
        test_size=0.2,
        random_state=42,
    )
    events = list(ml_grpc_stub.TrainModel(request))
    final = events[-1]
    assert final.status == "completed", f"Training failed: {final.error_message}"
    return final.model_id


# ── ML-service: TrainModel ────────────────────────────────────────────────────

class TestGrpcTrainModel:

    def test_classification_random_forest_completes(self, ml_grpc_stub, classification_dataset_id):
        request = ml_service_pb2.TrainRequest(
            dataset_id=classification_dataset_id,
            algorithm="random_forest",
            task_type="classification",
            target_column="condition",
            feature_columns=["gene_a", "gene_b", "gene_c"],
            test_size=0.2,
            random_state=42,
        )
        events = list(ml_grpc_stub.TrainModel(request))
        final = events[-1]
        assert final.status == "completed"
        assert final.model_id != ""
        assert final.progress_percent == 100

    def test_regression_ridge_completes(self, ml_grpc_stub, regression_dataset_id_grpc):
        request = ml_service_pb2.TrainRequest(
            dataset_id=regression_dataset_id_grpc,
            algorithm="Ridge",
            task_type="regression",
            target_column="score",
            feature_columns=["gene_a", "gene_b"],
            test_size=0.2,
            random_state=42,
        )
        events = list(ml_grpc_stub.TrainModel(request))
        final = events[-1]
        assert final.status == "completed"
        assert final.progress_percent == 100

    def test_streams_progress_in_order(self, ml_grpc_stub, classification_dataset_id):
        request = ml_service_pb2.TrainRequest(
            dataset_id=classification_dataset_id,
            algorithm="random_forest",
            task_type="classification",
            target_column="condition",
            feature_columns=["gene_a", "gene_b", "gene_c"],
            test_size=0.2,
            random_state=42,
        )
        events = list(ml_grpc_stub.TrainModel(request))
        percents = [e.progress_percent for e in events]
        assert percents == sorted(percents), "Progress should be non-decreasing"
        assert percents[0] == 0
        assert percents[-1] == 100

    def test_training_produces_test_metrics(self, ml_grpc_stub, classification_dataset_id):
        request = ml_service_pb2.TrainRequest(
            dataset_id=classification_dataset_id,
            algorithm="random_forest",
            task_type="classification",
            target_column="condition",
            feature_columns=["gene_a", "gene_b", "gene_c"],
            test_size=0.2,
            random_state=42,
        )
        events = list(ml_grpc_stub.TrainModel(request))
        final = events[-1]
        assert len(final.test_metrics) > 0

    def test_missing_dataset_id_fails(self, ml_grpc_stub):
        request = ml_service_pb2.TrainRequest(
            dataset_id="no-such-dataset",
            algorithm="random_forest",
            task_type="classification",
            target_column="condition",
            test_size=0.2,
            random_state=42,
        )
        events = list(ml_grpc_stub.TrainModel(request))
        final = events[-1]
        assert final.status == "failed"
        assert final.error_message != ""

    def test_missing_target_column_fails(self, ml_grpc_stub, classification_dataset_id):
        request = ml_service_pb2.TrainRequest(
            dataset_id=classification_dataset_id,
            algorithm="random_forest",
            task_type="classification",
            target_column="no_such_column",
            feature_columns=["gene_a", "gene_b"],
            test_size=0.2,
            random_state=42,
        )
        events = list(ml_grpc_stub.TrainModel(request))
        final = events[-1]
        assert final.status == "failed"

    def test_empty_feature_columns_uses_all(self, ml_grpc_stub, classification_dataset_id):
        """Passing no feature_columns should default to all non-target columns."""
        request = ml_service_pb2.TrainRequest(
            dataset_id=classification_dataset_id,
            algorithm="random_forest",
            task_type="classification",
            target_column="condition",
            feature_columns=[],   # empty — use all
            test_size=0.2,
            random_state=42,
        )
        events = list(ml_grpc_stub.TrainModel(request))
        final = events[-1]
        assert final.status == "completed"


# ── ML-service: GetModelInfo ──────────────────────────────────────────────────

class TestGrpcGetModelInfo:

    def test_returns_correct_model_id(self, ml_grpc_stub, trained_classification_model_id):
        request = ml_service_pb2.ModelInfoRequest(model_id=trained_classification_model_id)
        result = ml_grpc_stub.GetModelInfo(request)
        assert result.model_id == trained_classification_model_id

    def test_returns_correct_algorithm(self, ml_grpc_stub, trained_classification_model_id):
        request = ml_service_pb2.ModelInfoRequest(model_id=trained_classification_model_id)
        result = ml_grpc_stub.GetModelInfo(request)
        assert result.algorithm == "random_forest"

    def test_returns_feature_columns(self, ml_grpc_stub, trained_classification_model_id):
        request = ml_service_pb2.ModelInfoRequest(model_id=trained_classification_model_id)
        result = ml_grpc_stub.GetModelInfo(request)
        assert len(result.feature_columns) > 0

    def test_returns_test_metrics(self, ml_grpc_stub, trained_classification_model_id):
        request = ml_service_pb2.ModelInfoRequest(model_id=trained_classification_model_id)
        result = ml_grpc_stub.GetModelInfo(request)
        assert len(result.test_metrics) > 0

    def test_nonexistent_model_raises_not_found(self, ml_grpc_stub):
        request = ml_service_pb2.ModelInfoRequest(model_id="no-such-model")
        with pytest.raises(grpc.RpcError) as exc_info:
            ml_grpc_stub.GetModelInfo(request)
        assert exc_info.value.code() == grpc.StatusCode.NOT_FOUND


# ── ML-service: ListModels ────────────────────────────────────────────────────

class TestGrpcListModels:

    def test_returns_models_after_training(self, ml_grpc_stub, trained_classification_model_id):
        request = ml_service_pb2.ListModelsRequest(limit=10)
        result = ml_grpc_stub.ListModels(request)
        assert result.total_count > 0
        assert len(result.models) > 0

    def test_model_ids_are_unique(self, ml_grpc_stub):
        request = ml_service_pb2.ListModelsRequest(limit=50)
        result = ml_grpc_stub.ListModels(request)
        ids = [m.model_id for m in result.models]
        assert len(ids) == len(set(ids))

    def test_filter_by_algorithm(self, ml_grpc_stub, trained_classification_model_id):
        request = ml_service_pb2.ListModelsRequest(algorithm="random_forest", limit=10)
        result = ml_grpc_stub.ListModels(request)
        for model in result.models:
            assert model.algorithm == "random_forest"

    def test_filter_by_task_type(self, ml_grpc_stub, trained_classification_model_id):
        request = ml_service_pb2.ListModelsRequest(task_type="classification", limit=10)
        result = ml_grpc_stub.ListModels(request)
        for model in result.models:
            assert model.task_type == "classification"

    def test_limit_is_respected(self, ml_grpc_stub, trained_classification_model_id):
        request = ml_service_pb2.ListModelsRequest(limit=1)
        result = ml_grpc_stub.ListModels(request)
        assert len(result.models) <= 1


# ── ML-service: Predict ───────────────────────────────────────────────────────

class TestGrpcPredict:

    def test_classification_predict_succeeds(self, ml_grpc_stub, trained_classification_model_id):
        df = pd.DataFrame({
            "gene_a": [1.0, 2.0],
            "gene_b": [5.0, 4.0],
            "gene_c": [0.0, 1.0],
        })
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        request = ml_service_pb2.PredictRequest(
            model_id=trained_classification_model_id,
            input_data=buf.getvalue(),
            format="csv",
        )
        result = ml_grpc_stub.Predict(request)
        assert result.success
        assert len(result.predictions) == 2

    def test_classification_returns_probabilities(self, ml_grpc_stub, trained_classification_model_id):
        df = pd.DataFrame({
            "gene_a": [1.0],
            "gene_b": [5.0],
            "gene_c": [0.0],
        })
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        request = ml_service_pb2.PredictRequest(
            model_id=trained_classification_model_id,
            input_data=buf.getvalue(),
            format="csv",
        )
        result = ml_grpc_stub.Predict(request)
        assert result.success
        assert len(result.probabilities) > 0

    def test_regression_predict_succeeds(self, ml_grpc_stub, trained_regression_model_id):
        df = pd.DataFrame({
            "gene_a": [5.0, 10.0],
            "gene_b": [25.0, 20.0],
        })
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        request = ml_service_pb2.PredictRequest(
            model_id=trained_regression_model_id,
            input_data=buf.getvalue(),
            format="csv",
        )
        result = ml_grpc_stub.Predict(request)
        assert result.success
        assert len(result.predictions) == 2

    def test_nonexistent_model_returns_error(self, ml_grpc_stub):
        df = pd.DataFrame({"gene_a": [1.0], "gene_b": [2.0]})
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        request = ml_service_pb2.PredictRequest(
            model_id="no-such-model",
            input_data=buf.getvalue(),
            format="csv",
        )
        result = ml_grpc_stub.Predict(request)
        assert not result.success
        assert result.error_message != ""


# ── Data-service: _filter_cvs and NaN handling ───────────────────────────────

class TestDataServiceInternals:
    """
    Tests for internal data-service logic that can be exercised by uploading
    crafted CSVs and inspecting what comes back through ValidateDataset /
    StreamDataset — no direct method calls needed.
    """

    def _upload(self, grpc_stub, df):
        """Helper: upload a DataFrame as CSV and return (dataset_id, response)."""
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        request = data_service_pb2.ValidateRequest(
            dataset_content=buf.getvalue(), format="csv"
        )
        response = grpc_stub.ValidateDataset(request)
        return response.info.dataset_id, response

    def _stream_to_df(self, grpc_stub, dataset_id):
        """Helper: stream a dataset back and reassemble into a DataFrame."""
        chunks = list(grpc_stub.StreamDataset(
            data_service_pb2.StreamRequest(dataset_id=dataset_id, chunk_size=1000)
        ))
        frames = []
        for i, chunk in enumerate(chunks):
            text = chunk.data.decode("utf-8")
            frames.append(pd.read_csv(io.StringIO(text)))
        return pd.concat(frames, ignore_index=True)

    # -- NaN handling --

    def test_dataset_with_nans_is_valid(self, grpc_stub):
        """A dataset with some NaNs should still be accepted (filled to 0)."""
        df = pd.DataFrame({
            "sample": ["s1", "s2", "s3"],
            "gene_a": [1.0, None, 3.0],
            "gene_b": [4.0, 5.0, None],
            "condition": [0, 1, 0],
        })
        _, response = self._upload(grpc_stub, df)
        assert response.is_valid

    def test_dataset_over_50pct_nans_warns(self, grpc_stub):
        """A dataset with >50% NaNs should include a warning."""
        df = pd.DataFrame({
            "gene_a": [1.0, None, None, None],
            "gene_b": [None, None, None, 2.0],
        })
        _, response = self._upload(grpc_stub, df)
        assert any("missing" in w.lower() for w in response.warnings)

    # -- _filter_cvs: coefficient of variation filter --

    def test_high_cv_columns_retained_after_download(self, grpc_stub):
        """
        Columns with CV > threshold should survive _filter_cvs.
        We verify indirectly: after a successful download the returned
        column count should be less than the raw gene count (some filtered).
        This test only runs if a live download succeeds.
        """
        request = data_service_pb2.DownloadRequest(
            osd_id="379",
            patterns=["Unnormalized", "RSEM"],
        )
        result = grpc_stub.DownloadDataset(request)
        if not result.is_valid:
            pytest.skip("Live NASA download unavailable — skipping CV filter check")
        # condition column + at least some gene columns should remain
        assert result.info.num_columns >= 2
        col_names = [c.name for c in result.info.columns]
        assert "condition" in col_names

    # -- Metadata merge --

    def test_download_includes_condition_column(self, grpc_stub):
        """After a successful download, 'condition' column must be present."""
        request = data_service_pb2.DownloadRequest(
            osd_id="379",
            patterns=["Unnormalized", "RSEM"],
        )
        result = grpc_stub.DownloadDataset(request)
        if not result.is_valid:
            pytest.skip("Live NASA download unavailable")
        col_names = [c.name for c in result.info.columns]
        assert "condition" in col_names

    def test_condition_column_is_numeric(self, grpc_stub):
        """condition should be 0/1 integers after the spaceflight label mapping."""
        request = data_service_pb2.DownloadRequest(
            osd_id="379",
            patterns=["Unnormalized", "RSEM"],
        )
        result = grpc_stub.DownloadDataset(request)
        if not result.is_valid:
            pytest.skip("Live NASA download unavailable")
        condition_col = next(
            (c for c in result.info.columns if c.name == "condition"), None
        )
        assert condition_col is not None
        assert condition_col.dtype == "numeric"
        # sample values should all be "0" or "1"
        assert all(v in ("0", "1", "0.0", "1.0") for v in condition_col.sample_values)

    # -- _build_dataset_info dtype detection --

    def test_numeric_columns_detected(self, grpc_stub):
        df = pd.DataFrame({
            "int_col":   [1, 2, 3],
            "float_col": [1.1, 2.2, 3.3],
            "str_col":   ["a", "b", "c"],
        })
        _, response = self._upload(grpc_stub, df)
        dtype_map = {c.name: c.dtype for c in response.info.columns}
        assert dtype_map["int_col"]   == "numeric"
        assert dtype_map["float_col"] == "numeric"
        assert dtype_map["str_col"]   == "categorical"

    def test_null_count_reported_correctly(self, grpc_stub):
        df = pd.DataFrame({
            "gene_a": [1.0, None, 3.0, None, 5.0],
            "gene_b": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        _, response = self._upload(grpc_stub, df)
        null_map = {c.name: c.null_count for c in response.info.columns}
        assert null_map["gene_a"] == 2
        assert null_map["gene_b"] == 0

    def test_sample_values_populated(self, grpc_stub):
        df = pd.DataFrame({
            "gene_a": [10.0, 20.0, 30.0, 40.0],
        })
        _, response = self._upload(grpc_stub, df)
        col = response.info.columns[0]
        assert len(col.sample_values) == 3   # head(3) in _build_dataset_info

    # -- StreamDataset round-trip --

    def test_streamed_data_matches_original(self, grpc_stub):
        """Data streamed back should round-trip cleanly."""
        df = pd.DataFrame({
            "sample":    ["s1", "s2", "s3", "s4", "s5"],
            "gene_a":    [1.0, 2.0, 3.0, 4.0, 5.0],
            "gene_b":    [5.0, 4.0, 3.0, 2.0, 1.0],
            "condition": [0, 1, 0, 1, 0],
        })
        dataset_id, response = self._upload(grpc_stub, df)
        assert response.is_valid

        streamed_df = self._stream_to_df(grpc_stub, dataset_id)
        assert len(streamed_df) == 5
        assert set(streamed_df.columns) == set(df.columns)
        assert list(streamed_df["gene_a"]) == [1.0, 2.0, 3.0, 4.0, 5.0]
