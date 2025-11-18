import { Card } from "@/components/ui/card";
import { Cloud, GitBranch, Activity, Container } from "lucide-react";

export const DeploymentSection = () => {
  return (
    <section id="deployment" className="py-20 px-4 bg-muted/30">
      <div className="container mx-auto max-w-6xl">
        <div className="text-center space-y-4 mb-16">
          <h2 className="text-3xl sm:text-4xl font-bold">Deployment & MLOps</h2>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            Production-grade deployment with CI/CD automation and continuous monitoring
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-12">
          <Card className="p-6 space-y-4">
            <div className="flex items-center gap-3">
              <div className="p-3 bg-primary/10 rounded-lg">
                <Container className="h-6 w-6 text-primary" />
              </div>
              <h3 className="text-xl font-semibold">Containerization</h3>
            </div>
            <div className="space-y-3">
              <p className="text-sm text-muted-foreground">
                Docker containers ensure consistent environments across development, testing, and production.
              </p>
              <div className="bg-muted/50 p-4 rounded-lg font-mono text-xs space-y-1">
                <div className="text-muted-foreground"># Dockerfile</div>
                <div>FROM python:3.9-slim</div>
                <div>WORKDIR /app</div>
                <div>COPY requirements.txt .</div>
                <div>RUN pip install -r requirements.txt</div>
                <div>COPY . .</div>
                <div>EXPOSE 8000</div>
                <div>CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0"]</div>
              </div>
              <div className="bg-muted/50 p-4 rounded-lg font-mono text-xs space-y-1">
                <div className="text-muted-foreground"># Docker Compose</div>
                <div>services:</div>
                <div className="ml-2">api:</div>
                <div className="ml-4">build: .</div>
                <div className="ml-4">ports: ["8000:8000"]</div>
                <div className="ml-2">redis:</div>
                <div className="ml-4">image: redis:alpine</div>
              </div>
            </div>
          </Card>

          <Card className="p-6 space-y-4">
            <div className="flex items-center gap-3">
              <div className="p-3 bg-primary/10 rounded-lg">
                <GitBranch className="h-6 w-6 text-primary" />
              </div>
              <h3 className="text-xl font-semibold">CI/CD Pipeline</h3>
            </div>
            <div className="space-y-3">
              <p className="text-sm text-muted-foreground">
                Automated testing, building, and deployment using GitHub Actions
              </p>
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-primary"></div>
                  <span className="text-sm font-medium">Continuous Integration</span>
                </div>
                <ul className="ml-4 space-y-1 text-sm text-muted-foreground">
                  <li>• Run unit tests on every commit</li>
                  <li>• Lint code with flake8 and black</li>
                  <li>• Type checking with mypy</li>
                  <li>• Security scanning with bandit</li>
                </ul>
              </div>
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-primary"></div>
                  <span className="text-sm font-medium">Continuous Deployment</span>
                </div>
                <ul className="ml-4 space-y-1 text-sm text-muted-foreground">
                  <li>• Build and push Docker images</li>
                  <li>• Deploy to staging environment</li>
                  <li>• Run integration tests</li>
                  <li>• Deploy to production on approval</li>
                </ul>
              </div>
            </div>
          </Card>

          <Card className="p-6 space-y-4 lg:col-span-2">
            <div className="flex items-center gap-3">
              <div className="p-3 bg-primary/10 rounded-lg">
                <Cloud className="h-6 w-6 text-primary" />
              </div>
              <h3 className="text-xl font-semibold">Cloud Deployment Options</h3>
            </div>
            <div className="grid sm:grid-cols-3 gap-4">
              <div className="space-y-2">
                <h4 className="font-medium text-sm text-primary">AWS</h4>
                <ul className="space-y-1 text-sm text-muted-foreground">
                  <li>• <strong>SageMaker:</strong> Model training & hosting</li>
                  <li>• <strong>ECS/EKS:</strong> Container orchestration</li>
                  <li>• <strong>Lambda:</strong> Serverless inference</li>
                  <li>• <strong>S3:</strong> Model artifact storage</li>
                  <li>• <strong>CloudWatch:</strong> Monitoring & logging</li>
                </ul>
              </div>
              <div className="space-y-2">
                <h4 className="font-medium text-sm text-primary">Google Cloud</h4>
                <ul className="space-y-1 text-sm text-muted-foreground">
                  <li>• <strong>Vertex AI:</strong> End-to-end ML platform</li>
                  <li>• <strong>GKE:</strong> Kubernetes engine</li>
                  <li>• <strong>Cloud Run:</strong> Serverless containers</li>
                  <li>• <strong>GCS:</strong> Storage for models & data</li>
                  <li>• <strong>Cloud Monitoring:</strong> Observability</li>
                </ul>
              </div>
              <div className="space-y-2">
                <h4 className="font-medium text-sm text-primary">Azure / Local</h4>
                <ul className="space-y-1 text-sm text-muted-foreground">
                  <li>• <strong>Azure ML:</strong> Model management</li>
                  <li>• <strong>AKS:</strong> Kubernetes service</li>
                  <li>• <strong>Kubernetes/k3s:</strong> Self-hosted</li>
                  <li>• <strong>Docker Swarm:</strong> Lightweight orchestration</li>
                  <li>• <strong>Bare Metal:</strong> On-premise deployment</li>
                </ul>
              </div>
            </div>
          </Card>

          <Card className="p-6 space-y-4 lg:col-span-2">
            <div className="flex items-center gap-3">
              <div className="p-3 bg-primary/10 rounded-lg">
                <Activity className="h-6 w-6 text-primary" />
              </div>
              <h3 className="text-xl font-semibold">Monitoring & Observability</h3>
            </div>
            <div className="grid sm:grid-cols-2 gap-6">
              <div className="space-y-3">
                <h4 className="font-medium text-sm text-primary">Model Performance</h4>
                <ul className="space-y-1 text-sm text-muted-foreground">
                  <li>• <strong>Reward tracking:</strong> Monitor episode rewards over time</li>
                  <li>• <strong>Model drift:</strong> Detect performance degradation</li>
                  <li>• <strong>Prediction latency:</strong> Ensure real-time response</li>
                  <li>• <strong>Action distribution:</strong> Track policy behavior changes</li>
                </ul>
              </div>
              <div className="space-y-3">
                <h4 className="font-medium text-sm text-primary">Data & Infrastructure</h4>
                <ul className="space-y-1 text-sm text-muted-foreground">
                  <li>• <strong>Data drift:</strong> Monitor input distribution shifts</li>
                  <li>• <strong>System metrics:</strong> CPU, memory, GPU utilization</li>
                  <li>• <strong>API metrics:</strong> Request rate, error rate, uptime</li>
                  <li>• <strong>Alerts:</strong> Automated notifications for anomalies</li>
                </ul>
              </div>
            </div>
            <div className="bg-muted/50 p-4 rounded-lg">
              <h4 className="font-medium text-sm mb-2">Monitoring Stack</h4>
              <div className="flex flex-wrap gap-2">
                <span className="px-3 py-1 bg-background rounded-md text-xs font-mono">Prometheus</span>
                <span className="px-3 py-1 bg-background rounded-md text-xs font-mono">Grafana</span>
                <span className="px-3 py-1 bg-background rounded-md text-xs font-mono">MLflow</span>
                <span className="px-3 py-1 bg-background rounded-md text-xs font-mono">Weights & Biases</span>
                <span className="px-3 py-1 bg-background rounded-md text-xs font-mono">TensorBoard</span>
              </div>
            </div>
          </Card>
        </div>

        <Card className="p-8 bg-primary/5 border-primary/20">
          <h3 className="text-xl font-semibold mb-4">MLOps Best Practices</h3>
          <div className="grid sm:grid-cols-2 gap-6 text-sm text-muted-foreground">
            <ul className="space-y-2">
              <li className="flex items-start gap-2">
                <span className="text-primary mt-1">✓</span>
                <span><strong>Version Control:</strong> Track code, data, and model versions</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-primary mt-1">✓</span>
                <span><strong>Reproducibility:</strong> Deterministic training with seed control</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-primary mt-1">✓</span>
                <span><strong>Automated Testing:</strong> Unit, integration, and performance tests</span>
              </li>
            </ul>
            <ul className="space-y-2">
              <li className="flex items-start gap-2">
                <span className="text-primary mt-1">✓</span>
                <span><strong>Continuous Monitoring:</strong> Real-time performance tracking</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-primary mt-1">✓</span>
                <span><strong>Model Registry:</strong> Centralized model management</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-primary mt-1">✓</span>
                <span><strong>Rollback Strategy:</strong> Quick reversion on failures</span>
              </li>
            </ul>
          </div>
        </Card>
      </div>
    </section>
  );
};
