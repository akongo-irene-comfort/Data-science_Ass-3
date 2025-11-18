"""
Unit tests for FastAPI inference service
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

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
    assert "message" in response.json()


def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "model_version" in data


def test_predict_endpoint():
    """Test prediction endpoint with valid input"""
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

    # May return 503 if model not loaded in test environment
    assert response.status_code in [200, 503]

    if response.status_code == 200:
        data = response.json()
        assert "action" in data
        assert "confidence" in data
        assert "inference_time_ms" in data
        assert 0 <= data["action"] < 8


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
    # Should fail validation
    assert response.status_code in [422, 503]


def test_model_info():
    """Test model info endpoint"""
    response = client.get("/model-info")
    # May return 503 if model not loaded
    assert response.status_code in [200, 503]

    if response.status_code == 200:
        data = response.json()
        assert "version" in data
        assert "action_space" in data


def test_batch_predict():
    """Test batch prediction endpoint"""
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
                "densities": [0.033, 0.04, 0.027, 0.05, 0.03, 0.02, 0.037, 0.047],
                "time_of_day": 9.0,
            }
        },
    ]

    response = client.post("/batch-predict", json=request_data)
    assert response.status_code in [200, 503]

    if response.status_code == 200:
        data = response.json()
        assert "predictions" in data
        assert "batch_size" in data
        assert data["batch_size"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])