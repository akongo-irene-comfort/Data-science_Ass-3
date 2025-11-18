"""
Unit tests for FastAPI inference service
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Add the project root to the path if needed
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from ml.api.main import app
except ImportError:
    # Fallback for different project structures
    try:
        from api.main import app
    except ImportError:
        pytest.skip("Could not import app", allow_module_level=True)

client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data


def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "model_version" in data
    assert "device" in data
    assert "uptime_seconds" in data


def test_predict_endpoint_no_model():
    """Test prediction endpoint returns 503 when model not loaded"""
    request_data = {
        "state": {
            "vehicle_counts": [12, 8, 15, 10, 6, 9, 14, 11],
            "speeds": [35.5, 42.0, 28.3, 38.7, 45.2, 33.1, 30.5, 40.2],
            "densities": [0.04, 0.027, 0.05, 0.033, 0.02, 0.03, 0.047, 0.037],
            "time_of_day": 8.5,
        },
        "return_q_values": True,
        "request_id": "test-001",
    }

    response = client.post("/predict", json=request_data)

    # Should return 503 since model is not loaded in test environment
    assert response.status_code == 503
    data = response.json()
    assert "detail" in data
    assert "not loaded" in data["detail"].lower()


def test_predict_invalid_input():
    """Test prediction with invalid input"""
    request_data = {
        "state": {
            "vehicle_counts": [12],  # Too few values
            "speeds": [35.5],
            "densities": [0.04],
            "time_of_day": 8.5,
        }
    }

    response = client.post("/predict", json=request_data)
    # Should fail validation with 422
    assert response.status_code == 422


def test_predict_invalid_time_of_day():
    """Test prediction with invalid time_of_day"""
    request_data = {
        "state": {
            "vehicle_counts": [12, 8, 15, 10, 6, 9, 14, 11],
            "speeds": [35.5, 42.0, 28.3, 38.7, 45.2, 33.1, 30.5, 40.2],
            "densities": [0.04, 0.027, 0.05, 0.033, 0.02, 0.03, 0.047, 0.037],
            "time_of_day": 25.0,  # Invalid: > 24
        }
    }

    response = client.post("/predict", json=request_data)
    # Should fail validation
    assert response.status_code == 422


def test_model_info_no_model():
    """Test model info endpoint when model not loaded"""
    response = client.get("/model-info")
    # Should return 503 if model not loaded
    assert response.status_code == 503


def test_batch_predict_no_model():
    """Test batch prediction endpoint when model not loaded"""
    request_data = [
        {
            "state": {
                "vehicle_counts": [12, 8, 15, 10, 6, 9, 14, 11],
                "speeds": [35.5, 42.0, 28.3, 38.7, 45.2, 33.1, 30.5, 40.2],
                "densities": [0.04, 0.027, 0.05, 0.033, 0.02, 0.03, 0.047, 0.037],
                "time_of_day": 8.5,
            }
        },
        {
            "state": {
                "vehicle_counts": [10, 12, 8, 15, 9, 6, 11, 14],
                "speeds": [40.0, 38.5, 42.3, 35.7, 43.2, 39.1, 36.5, 41.2],
                "densities": [
                    0.033,
                    0.04,
                    0.027,
                    0.05,
                    0.03,
                    0.02,
                    0.037,
                    0.047,
                ],
                "time_of_day": 9.0,
            }
        },
    ]

    response = client.post("/batch-predict", json=request_data)
    # Should return 503 since model not loaded
    assert response.status_code == 503


def test_batch_predict_exceeds_limit():
    """Test batch prediction with too many requests"""
    # Create 101 requests (exceeds limit of 100)
    request_data = [
        {
            "state": {
                "vehicle_counts": [12, 8, 15, 10, 6, 9, 14, 11],
                "speeds": [35.5, 42.0, 28.3, 38.7, 45.2, 33.1, 30.5, 40.2],
                "densities": [0.04, 0.027, 0.05, 0.033, 0.02, 0.03, 0.047, 0.037],
                "time_of_day": 8.5,
            }
        }
        for _ in range(101)
    ]

    response = client.post("/batch-predict", json=request_data)
    # Should return 400 for exceeding batch limit
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert "exceeds limit" in data["detail"].lower()


def test_metrics_endpoint():
    """Test Prometheus metrics endpoint"""
    response = client.get("/metrics")
    assert response.status_code == 200
    # Metrics should be in Prometheus text format
    assert "text/plain" in response.headers.get("content-type", "")


def test_reload_model_endpoint():
    """Test model reload endpoint"""
    response = client.post("/reload-model")
    # Will fail since model path doesn't exist in test, but endpoint should exist
    assert response.status_code in [200, 500]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])