# RL Traffic Control - MLOps Pipeline

Complete MLOps implementation for training, deploying, and monitoring a Deep Q-Network (DQN) agent for traffic signal optimization.

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Docker & Kubernetes
- GitHub account (for CI/CD)
- SUMO traffic simulator
- (Optional) Cloud provider account (AWS/GCP/Azure)

### Installation

```bash
# Clone repository
git clone https://github.com/YOUR_ORG/rl-traffic-control.git
cd rl-traffic-control

# Install dependencies
cd ml
pip install -r requirements.txt

# Install SUMO (Ubuntu/Debian)
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update
sudo apt-get install sumo sumo-tools sumo-doc
```

## 📋 Pipeline Overview

### 1. **Training Pipeline**

Train the DQN agent locally or in CI/CD:

```bash
# Local training
python ml/train.py \
  --episodes 1000 \
  --batch-size 64 \
  --learning-rate 0.001 \
  --gamma 0.95 \
  --output-dir models/

# With Weights & Biases tracking
python ml/train.py --use-wandb
```

**Output artifacts:**
- `models/dqn_final.pth` - PyTorch state dict
- `models/dqn_final_traced.pt` - TorchScript for production
- `models/training_metrics.json` - Training statistics
- `models/dqn_best.pth` - Best performing checkpoint

### 2. **Model Evaluation**

Evaluate trained model performance:

```bash
python ml/evaluate.py \
  --model-path models/dqn_final.pth \
  --episodes 10 \
  --output metrics/evaluation.json
```

**Key metrics:**
- Average episode reward
- Mean waiting time (target: <60s)
- Queue length
- Throughput (vehicles/hour)

### 3. **Model Serving API**

FastAPI server with low-latency inference:

```bash
# Run locally
cd ml
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Test endpoint
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "state": {
      "vehicle_counts": [12, 8, 15, 10, 6, 9, 14, 11],
      "speeds": [35.5, 42.0, 28.3, 38.7, 45.2, 33.1, 30.5, 40.2],
      "densities": [0.04, 0.027, 0.05, 0.033, 0.02, 0.03, 0.047, 0.037],
      "time_of_day": 8.5
    }
  }'
```

**API endpoints:**
- `GET /health` - Health check
- `POST /predict` - Single prediction
- `POST /batch-predict` - Batch predictions
- `GET /metrics` - Prometheus metrics
- `GET /model-info` - Model metadata

### 4. **Containerization**

Build and run Docker container:

```bash
# Build image
docker build -t rl-traffic-api:latest -f ml/Dockerfile ml/

# Run container
docker run -p 8000:8000 rl-traffic-api:latest

# Push to registry
docker tag rl-traffic-api:latest ghcr.io/YOUR_ORG/rl-traffic-api:latest
docker push ghcr.io/YOUR_ORG/rl-traffic-api:latest
```

## 🔄 CI/CD Pipeline (GitHub Actions)

The pipeline automatically:
1. **Tests** - Linting, unit tests, coverage
2. **Trains** - DQN model training with SUMO
3. **Evaluates** - Performance benchmarking
4. **Builds** - Docker image creation
5. **Deploys** - Render deployment
6. **Monitors** - Prometheus & Grafana setup

### Workflow Triggers

- **Push to main** - Full pipeline (train → build → deploy)
- **Pull requests** - Tests only
- **Manual dispatch** - Custom environment selection

### Setup GitHub Actions

1. **Add secrets** (Settings → Secrets):
   ```
   KUBE_CONFIG      - Kubernetes cluster config
   WANDB_API_KEY    - Weights & Biases API key (optional)
   GITHUB_TOKEN     - Auto-provided by GitHub
   ```

2. **Push to repository**:
   ```bash
   git push origin main
   ```

3. **Monitor workflow**:
   - Go to Actions tab
   - View pipeline execution
   - Download artifacts

## ☸️ Render Deployment

### Local Deployment (k3s)

```bash
# Install k3s
curl -sfL https://get.k3s.io | sh -

# Apply manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml

# Verify deployment
kubectl get pods -n rl-traffic
kubectl get svc -n rl-traffic

# Port forward for testing
kubectl port-forward -n rl-traffic svc/rl-traffic-api 8080:80
```

### Cloud Deployment

**AWS SageMaker:**
```bash
# Build and push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com
docker tag rl-traffic-api:latest <account>.dkr.ecr.us-east-1.amazonaws.com/rl-traffic-api:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/rl-traffic-api:latest

# Deploy to SageMaker Endpoint
# (Use SageMaker SDK or AWS Console)
```

**GCP Vertex AI:**
```bash
# Build and push to GCR
gcloud auth configure-docker
docker tag rl-traffic-api:latest gcr.io/PROJECT_ID/rl-traffic-api:latest
docker push gcr.io/PROJECT_ID/rl-traffic-api:latest

# Deploy to Vertex AI
# (Use gcloud AI Platform or Console)
```

**Azure ML:**
```bash
# Build and push to ACR
az acr login --name <registry-name>
docker tag rl-traffic-api:latest <registry-name>.azurecr.io/rl-traffic-api:latest
docker push <registry-name>.azurecr.io/rl-traffic-api:latest

# Deploy to Azure ML
# (Use Azure ML SDK or Portal)
```

## 📊 Monitoring & Observability

### Prometheus Metrics

Deploy monitoring stack:

```bash
kubectl apply -f k8s/monitoring/prometheus-config.yaml
kubectl apply -f k8s/monitoring/prometheus-deployment.yaml
kubectl apply -f k8s/monitoring/grafana-deployment.yaml
kubectl apply -f k8s/monitoring/model-monitor-deployment.yaml

# Access Prometheus
kubectl port-forward -n rl-traffic svc/prometheus 9090:9090

# Access Grafana
kubectl port-forward -n rl-traffic svc/grafana 3000:3000
# Default credentials: admin / changeme123
```

### Key Monitoring Metrics

**Performance Metrics:**
- `inference_requests_total` - Request count by status
- `inference_latency_seconds` - Request latency histogram
- `model_confidence` - Average Q-value confidence
- `active_requests` - Concurrent request count

**Model Health:**
- `data_drift_score` - Input distribution drift (KL divergence)
- `model_drift_score` - Prediction pattern drift
- `avg_episode_reward` - Production environment rewards
- `prediction_entropy` - Action selection entropy

**Alerts:**
- High error rate (>5% for 5min)
- High latency (P95 >100ms)
- Data drift detected (score >0.3)
- Model drift detected (score >0.3)
- Low confidence (<0.5)
- Insufficient pods (<2)

### Grafana Dashboards

Pre-configured dashboards:
- **Model Performance** - Inference metrics, latency, errors
- **Drift Detection** - Data/model drift over time
- **System Health** - Pod status, resource usage
- **Business Metrics** - Traffic improvements, rewards

## 🧪 Testing

```bash
# Run all tests
pytest ml/tests/ -v

# Run specific test
pytest ml/tests/test_api.py::test_predict_endpoint -v

# With coverage
pytest ml/tests/ --cov=ml --cov-report=html
```

## 📈 Performance Benchmarks

**Inference Latency:**
- P50: <10ms
- P95: <25ms
- P99: <50ms

**Throughput:**
- Single instance: ~100 req/s
- With HPA (10 pods): ~1000 req/s

**Model Performance:**
- Waiting time reduction: 20-30% vs fixed-time control
- Throughput increase: 15-25%
- Queue length reduction: 25-35%

## 🛠️ Troubleshooting

### Common Issues

**Model not loading:**
```bash
# Check model file exists
kubectl exec -it -n rl-traffic deployment/rl-traffic-api -- ls -la /app/models/

# Check logs
kubectl logs -n rl-traffic deployment/rl-traffic-api
```

**High latency:**
```bash
# Check resource limits
kubectl top pods -n rl-traffic

# Scale deployment
kubectl scale deployment/rl-traffic-api -n rl-traffic --replicas=5
```

**Drift alerts:**
```bash
# Check monitoring service logs
kubectl logs -n rl-traffic deployment/model-monitor

# Retrain model with new data
# (Trigger CI/CD pipeline)
```

## 📚 Additional Resources

- [SUMO Documentation](https://sumo.dlr.de/docs/)
- [DQN Paper](https://www.nature.com/articles/nature14236)
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [Kubernetes Best Practices](https://kubernetes.io/docs/concepts/)
- [Prometheus Monitoring](https://prometheus.io/docs/introduction/overview/)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

MIT License - see LICENSE file for details

## 📧 Contact

For questions or support, please open an issue or contact the ML team.

---

**Built with ❤️ for intelligent traffic management**
