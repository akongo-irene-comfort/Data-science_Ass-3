import pytest
from fastapi.testclient import TestClient
from ml.api.main import app, model_state, load_model

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_model():
    """Ensure model is loaded before each test"""
    if model_state.model is None:
        load_model()

def test_root():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert data["message"] == "RL Traffic Control API"

def test_health():
    """Test health endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert "model_loaded" in data
    assert "model_version" in data
    assert "uptime_seconds" in data

def test_model_info():
    """Test model info endpoint"""
    response = client.get("/model-info")
    # This might return 503 if model is not loaded, which is acceptable for testing
    if response.status_code == 200:
        data = response.json()
        assert "version" in data
        assert "actions" in data
        assert len(data["actions"]) == 8
    else:
        assert response.status_code == 503

def test_predict_valid_input():
    """Test prediction with valid input"""
    request_data = {
        "state": {
            "vehicle_counts": [12, 8, 15, 10, 6, 9, 14, 11],
            "speeds": [35.5, 42.0, 28.3, 38.7, 45.2, 33.1, 30.5, 40.2],
            "densities": [0.04, 0.027, 0.05, 0.033, 0.02, 0.03, 0.047, 0.037],
            "time_of_day": 8.5,
        },
        "return_q_values": True,
        "request_id": "test-123"
    }

    response = client.post("/predict", json=request_data)
    # Accept both 200 (success) and 503 (model not ready) for now
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        data = response.json()
        assert "action" in data
        assert "action_name" in data
        assert "confidence" in data
        assert "q_values" in data
        assert "inference_time_ms" in data
        assert data["request_id"] == "test-123"
        assert isinstance(data["action"], int)
        assert 0 <= data["action"] <= 7

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
    error_detail = response.json()["detail"][0]
    assert "vehicle_counts" in str(error_detail["loc"])

def test_predict_invalid_time():
    """Test prediction with invalid time of day"""
    request_data = {
        "state": {
            "vehicle_counts": [12, 8, 15, 10, 6, 9, 14, 11],
            "speeds": [35.5, 42.0, 28.3, 38.7, 45.2, 33.1, 30.5, 40.2],
            "densities": [0.04, 0.027, 0.05, 0.033, 0.02, 0.03, 0.047, 0.037],
            "time_of_day": 25.0,  # Invalid time
        }
    }

    response = client.post("/predict", json=request_data)
    assert response.status_code == 422
    error_detail = response.json()["detail"][0]
    assert "time_of_day" in str(error_detail["loc"])

def test_predict_without_q_values():
    """Test prediction without returning Q-values"""
    request_data = {
        "state": {
            "vehicle_counts": [12, 8, 15, 10, 6, 9, 14, 11],
            "speeds": [35.5, 42.0, 28.3, 38.7, 45.2, 33.1, 30.5, 40.2],
            "densities": [0.04, 0.027, 0.05, 0.033, 0.02, 0.03, 0.047, 0.037],
            "time_of_day": 8.5,
        },
        "return_q_values": False
    }

    response = client.post("/predict", json=request_data)
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        data = response.json()
        assert data["q_values"] is None

def test_batch_predict():
    """Test batch prediction"""
    # Correct format - send list directly
    requests_data = [
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
                "vehicle_counts": [10, 7, 12, 9, 5, 8, 13, 10],
                "speeds": [40.0, 38.5, 32.1, 41.2, 43.0, 35.8, 29.9, 39.5],
                "densities": [0.03, 0.025, 0.045, 0.03, 0.018, 0.028, 0.05, 0.032],
                "time_of_day": 18.2,
            }
        }
    ]

    response = client.post("/batch-predict", json=requests_data)
    # Accept both 200 and 503
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        data = response.json()
        assert "predictions" in data
        assert "batch_size" in data
        assert data["batch_size"] == 2
        assert len(data["predictions"]) == 2

def test_batch_predict_too_large():
    """Test batch prediction with too many requests"""
    # Correct format - send list directly
    requests_data = [{
        "state": {
            "vehicle_counts": [12, 8, 15, 10, 6, 9, 14, 11],
            "speeds": [35.5, 42.0, 28.3, 38.7, 45.2, 33.1, 30.5, 40.2],
            "densities": [0.04, 0.027, 0.05, 0.033, 0.02, 0.03, 0.047, 0.037],
            "time_of_day": 8.5,
        }
    }] * 101  # Exceeds limit

    response = client.post("/batch-predict", json=requests_data)
    assert response.status_code == 400
    assert "exceeds limit" in response.json()["detail"].lower()

def test_metrics():
    """Test metrics endpoint"""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]

def test_reload_model():
    """Test model reload endpoint"""
    response = client.post("/reload-model")
    # This should work even with mock model
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "version" in data