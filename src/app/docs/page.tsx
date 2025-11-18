import { BookOpen, Code, Database, Zap, GitBranch, Cloud, Terminal, Package, Activity } from "lucide-react";

export default function DocsPage() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-muted/20">
      <div className="mx-auto max-w-5xl px-6 py-12 lg:px-8">
        {/* Header */}
        <div className="mb-8 text-center">
          <div className="mb-3 inline-flex items-center gap-2 rounded-full border border-blue-500/20 bg-blue-500/5 px-4 py-2">
            <BookOpen className="h-4 w-4 text-blue-600" />
            <span className="text-sm font-medium text-blue-600">Complete Documentation</span>
          </div>
          <h1 className="mb-3 text-4xl font-bold tracking-tight">Project Documentation</h1>
          <p className="mx-auto max-w-2xl text-muted-foreground">
            Complete technical documentation including MLOps pipeline for Traffic Congestion Optimization
          </p>
        </div>

        {/* Quick Links */}
        <div className="mb-8 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <a href="#setup" className="rounded-lg border border-border bg-card p-3 transition-colors hover:bg-accent">
            <Terminal className="mb-2 h-5 w-5 text-primary" />
            <div className="text-sm font-semibold">Setup Guide</div>
          </a>
          <a href="#architecture" className="rounded-lg border border-border bg-card p-3 transition-colors hover:bg-accent">
            <Code className="mb-2 h-5 w-5 text-primary" />
            <div className="text-sm font-semibold">Architecture</div>
          </a>
          <a href="#training" className="rounded-lg border border-border bg-card p-3 transition-colors hover:bg-accent">
            <Zap className="mb-2 h-5 w-5 text-primary" />
            <div className="text-sm font-semibold">Training</div>
          </a>
          <a href="#mlops" className="rounded-lg border border-border bg-card p-3 transition-colors hover:bg-accent">
            <Activity className="mb-2 h-5 w-5 text-primary" />
            <div className="text-sm font-semibold">MLOps Pipeline</div>
          </a>
        </div>

        {/* Setup Guide */}
        <section id="setup" className="mb-8 rounded-xl border border-border bg-card p-6">
          <div className="mb-4 flex items-center gap-3">
            <Terminal className="h-6 w-6 text-blue-600" />
            <h2 className="text-2xl font-bold">Installation & Setup</h2>
          </div>

          <div className="space-y-4">
            <div>
              <h3 className="mb-2 text-lg font-semibold">Prerequisites</h3>
              <ul className="ml-4 space-y-1 text-sm text-muted-foreground">
                <li>• Python 3.10+ with pip</li>
                <li>• SUMO Traffic Simulator 1.19+</li>
                <li>• CUDA 11.0+ (for GPU training)</li>
                <li>• Docker & Kubernetes (for deployment)</li>
              </ul>
            </div>

            <div>
              <h3 className="mb-2 text-lg font-semibold">Install Dependencies</h3>
              <pre className="rounded-lg bg-slate-900 p-3 text-sm text-slate-100">
cd ml
pip install -r requirements.txt</pre>
              <p className="mt-2 text-sm text-muted-foreground">Key packages: torch, gymnasium, traci, fastapi, prometheus-client</p>
            </div>

            <div>
              <h3 className="mb-2 text-lg font-semibold">Configure SUMO</h3>
              <pre className="rounded-lg bg-slate-900 p-3 text-sm text-slate-100">
export SUMO_HOME="/usr/share/sumo"
echo 'export SUMO_HOME="/usr/share/sumo"' &gt;&gt; ~/.bashrc</pre>
            </div>
          </div>
        </section>

        {/* Architecture */}
        <section id="architecture" className="mb-8 rounded-xl border border-border bg-card p-6">
          <div className="mb-4 flex items-center gap-3">
            <Code className="h-6 w-6 text-purple-600" />
            <h2 className="text-2xl font-bold">System Architecture</h2>
          </div>

          <div className="space-y-4">
            <div>
              <h3 className="mb-2 text-lg font-semibold">DQN Network Architecture</h3>
              <div className="rounded-lg bg-blue-500/5 p-4 text-sm">
                <p className="mb-2 font-semibold text-blue-600">Input Layer:</p>
                <p className="mb-3 text-muted-foreground">State vector (25 dimensions): 8 lanes × 3 features + time of day</p>
                
                <p className="mb-2 font-semibold text-purple-600">Hidden Layers:</p>
                <p className="mb-3 text-muted-foreground">Dense(256) → ReLU → BatchNorm → Dense(256) → ReLU → Dropout(0.2) → Dense(128) → ReLU</p>
                
                <p className="mb-2 font-semibold text-green-600">Output Layer:</p>
                <p className="text-muted-foreground">Q-values for 8 discrete actions (traffic signal phases)</p>
              </div>
            </div>
          </div>
        </section>

        {/* Training */}
        <section id="training" className="mb-8 rounded-xl border border-border bg-card p-6">
          <div className="mb-4 flex items-center gap-3">
            <Zap className="h-6 w-6 text-orange-600" />
            <h2 className="text-2xl font-bold">Training Guide</h2>
          </div>

          <div className="space-y-4">
            <div>
              <h3 className="mb-2 text-lg font-semibold">Basic Training</h3>
              <pre className="rounded-lg bg-slate-900 p-3 text-sm text-slate-100">
{`python ml/train.py \\
  --episodes 1000 \\
  --batch-size 64 \\
  --learning-rate 0.001 \\
  --gamma 0.95 \\
  --use-wandb`}
              </pre>
            </div>

            <div>
              <h3 className="mb-2 text-lg font-semibold">Hyperparameters</h3>
              <div className="grid gap-3 sm:grid-cols-2">
                <div className="rounded-lg bg-muted/50 p-3 text-sm">
                  <p className="mb-1 font-semibold">Learning Rate:</p>
                  <p className="text-muted-foreground">0.001 (Adam optimizer)</p>
                </div>
                <div className="rounded-lg bg-muted/50 p-3 text-sm">
                  <p className="mb-1 font-semibold">Discount Factor (γ):</p>
                  <p className="text-muted-foreground">0.95</p>
                </div>
                <div className="rounded-lg bg-muted/50 p-3 text-sm">
                  <p className="mb-1 font-semibold">Replay Buffer:</p>
                  <p className="text-muted-foreground">100,000 transitions</p>
                </div>
                <div className="rounded-lg bg-muted/50 p-3 text-sm">
                  <p className="mb-1 font-semibold">Target Update:</p>
                  <p className="text-muted-foreground">Every 1000 steps</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* MLOps Pipeline - MAIN SECTION */}
        <section id="mlops" className="mb-8 rounded-xl border-2 border-primary/20 bg-gradient-to-br from-orange-500/5 to-pink-500/5 p-6">
          <div className="mb-4 flex items-center gap-3">
            <Activity className="h-7 w-7 text-orange-600" />
            <h2 className="text-2xl font-bold">MLOps Pipeline</h2>
          </div>

          <div className="space-y-4">
            {/* CI/CD */}
            <div className="rounded-xl border border-border bg-card p-4">
              <h3 className="mb-3 text-lg font-semibold text-orange-600">GitHub Actions CI/CD</h3>
              <p className="mb-3 text-sm text-muted-foreground">
                Automated pipeline: Test → Train → Build → Deploy → Monitor
              </p>
              <div className="rounded-lg bg-muted/30 p-3 text-sm">
                <strong>Pipeline Stages:</strong>
                <ul className="ml-4 mt-2 space-y-1 text-xs text-muted-foreground">
                  <li>• <strong>Test:</strong> Linting (flake8, black), unit tests, coverage reports</li>
                  <li>• <strong>Train:</strong> DQN training with SUMO (100 episodes for CI)</li>
                  <li>• <strong>Evaluate:</strong> Performance benchmarking on test episodes</li>
                  <li>• <strong>Build:</strong> Docker image creation and push to GHCR</li>
                  <li>• <strong>Deploy:</strong> Kubernetes deployment with rolling updates</li>
                  <li>• <strong>Monitor:</strong> Prometheus and Grafana stack deployment</li>
                </ul>
              </div>
              <p className="mt-2 text-xs text-muted-foreground">
                Workflow file: <code>.github/workflows/train-and-deploy.yml</code>
              </p>
            </div>

            {/* Model Serving */}
            <div className="rounded-xl border border-border bg-card p-4">
              <h3 className="mb-3 text-lg font-semibold text-green-600">FastAPI Model Serving</h3>
              <div className="space-y-3">
                <div className="rounded-lg bg-muted/30 p-3 text-sm">
                  <strong>Inference API:</strong> Low-latency endpoint accepting state observations
                  <pre className="mt-2 text-xs">POST /predict - P95 latency: ~25ms</pre>
                </div>
                <div className="grid gap-2 sm:grid-cols-2 text-xs">
                  <div className="rounded-lg bg-green-500/5 p-3">
                    <strong>Throughput:</strong> ~100 req/s per instance
                  </div>
                  <div className="rounded-lg bg-green-500/5 p-3">
                    <strong>Scalability:</strong> Auto-scaling 2-10 replicas
                  </div>
                </div>
              </div>
            </div>

            {/* Containerization */}
            <div className="rounded-xl border border-border bg-card p-4">
              <h3 className="mb-3 text-lg font-semibold text-indigo-600">Docker Containerization</h3>
              <pre className="rounded-lg bg-slate-900 p-3 text-xs text-slate-100">
{`# Build & run
docker build -t rl-traffic-api:latest -f ml/Dockerfile ml/
docker run -p 8000:8000 rl-traffic-api:latest`}
              </pre>
              <p className="mt-2 text-xs text-muted-foreground">Multi-stage build with security best practices</p>
            </div>

            {/* Kubernetes */}
            <div className="rounded-xl border border-border bg-card p-4">
              <h3 className="mb-3 text-lg font-semibold text-cyan-600">Kubernetes Deployment</h3>
              <div className="space-y-3">
                <div className="rounded-lg bg-muted/30 p-3 text-sm">
                  <strong>Local (k3s):</strong>
                  <pre className="mt-2 text-xs">
{`kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/hpa.yaml`}
                  </pre>
                </div>
                <div className="grid gap-2 sm:grid-cols-3 text-xs">
                  <div className="rounded-lg bg-cyan-500/5 p-2">
                    <strong>AWS SageMaker</strong>
                    <p className="mt-1 text-muted-foreground">Managed ML endpoints</p>
                  </div>
                  <div className="rounded-lg bg-cyan-500/5 p-2">
                    <strong>GCP Vertex AI</strong>
                    <p className="mt-1 text-muted-foreground">Serverless deployment</p>
                  </div>
                  <div className="rounded-lg bg-cyan-500/5 p-2">
                    <strong>Azure ML</strong>
                    <p className="mt-1 text-muted-foreground">ACR integration</p>
                  </div>
                </div>
              </div>
            </div>

            {/* Monitoring */}
            <div className="rounded-xl border border-border bg-card p-4">
              <h3 className="mb-3 text-lg font-semibold text-pink-600">Production Monitoring</h3>
              <div className="space-y-3">
                <div className="grid gap-3 sm:grid-cols-2 text-sm">
                  <div className="rounded-lg bg-muted/30 p-3">
                    <strong>Prometheus Metrics:</strong>
                    <ul className="ml-4 mt-2 space-y-1 text-xs text-muted-foreground">
                      <li>• inference_requests_total</li>
                      <li>• inference_latency_seconds</li>
                      <li>• model_confidence</li>
                      <li>• data_drift_score</li>
                      <li>• model_drift_score</li>
                      <li>• avg_episode_reward</li>
                    </ul>
                  </div>
                  <div className="rounded-lg bg-muted/30 p-3">
                    <strong>Grafana Dashboards:</strong>
                    <ul className="ml-4 mt-2 space-y-1 text-xs text-muted-foreground">
                      <li>• Model performance metrics</li>
                      <li>• Request latency & throughput</li>
                      <li>• Error rates by status code</li>
                      <li>• Data/model drift visualization</li>
                      <li>• System health & resources</li>
                    </ul>
                  </div>
                </div>
                <div className="rounded-lg bg-pink-500/5 p-3 text-xs">
                  <strong>Drift Detection:</strong> Real-time monitoring using KL divergence
                  <ul className="ml-4 mt-2 space-y-1 text-muted-foreground">
                    <li>• <strong>Data Drift:</strong> Input distribution changes (alert &gt;0.3)</li>
                    <li>• <strong>Model Drift:</strong> Prediction pattern shifts (alert &gt;0.3)</li>
                    <li>• <strong>Reward Tracking:</strong> Production environment performance</li>
                  </ul>
                </div>
                <div className="rounded-lg bg-muted/30 p-3 text-xs">
                  <strong>Automated Alerts:</strong> High error rate, latency, drift detection, low confidence
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Model Serving Details */}
        <section id="serving" className="mb-8 rounded-xl border border-border bg-card p-6">
          <div className="mb-4 flex items-center gap-3">
            <Package className="h-6 w-6 text-green-600" />
            <h2 className="text-2xl font-bold">Model Serving API</h2>
          </div>

          <div className="space-y-4">
            <div>
              <h3 className="mb-2 text-lg font-semibold">API Endpoints</h3>
              <div className="space-y-2 text-sm">
                <div className="flex gap-3 rounded-lg bg-muted/50 p-3">
                  <code className="font-semibold text-green-600">POST</code>
                  <div className="flex-1">
                    <p className="font-semibold">/predict</p>
                    <p className="text-xs text-muted-foreground">Single inference (P95: ~25ms)</p>
                  </div>
                </div>
                <div className="flex gap-3 rounded-lg bg-muted/50 p-3">
                  <code className="font-semibold text-green-600">POST</code>
                  <div className="flex-1">
                    <p className="font-semibold">/batch-predict</p>
                    <p className="text-xs text-muted-foreground">Batch inference (up to 100 requests)</p>
                  </div>
                </div>
                <div className="flex gap-3 rounded-lg bg-muted/50 p-3">
                  <code className="font-semibold text-blue-600">GET</code>
                  <div className="flex-1">
                    <p className="font-semibold">/health</p>
                    <p className="text-xs text-muted-foreground">Health check for K8s probes</p>
                  </div>
                </div>
                <div className="flex gap-3 rounded-lg bg-muted/50 p-3">
                  <code className="font-semibold text-blue-600">GET</code>
                  <div className="flex-1">
                    <p className="font-semibold">/metrics</p>
                    <p className="text-xs text-muted-foreground">Prometheus metrics endpoint</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Evaluation */}
        <section id="evaluation" className="mb-8 rounded-xl border border-border bg-card p-6">
          <div className="mb-4 flex items-center gap-3">
            <Database className="h-6 w-6 text-pink-600" />
            <h2 className="text-2xl font-bold">Evaluation Metrics</h2>
          </div>

          <div className="grid gap-4 sm:grid-cols-2">
            <div className="rounded-lg bg-muted/50 p-4">
              <h3 className="mb-2 font-semibold">Primary Metrics</h3>
              <ul className="space-y-1 text-sm text-muted-foreground">
                <li>• Average waiting time per vehicle</li>
                <li>• Total throughput (vehicles/hour)</li>
                <li>• Queue length distribution</li>
                <li>• Episode reward</li>
              </ul>
            </div>
            <div className="rounded-lg bg-muted/50 p-4">
              <h3 className="mb-2 font-semibold">Comparison Baselines</h3>
              <ul className="space-y-1 text-sm text-muted-foreground">
                <li>• Fixed-time traffic lights (60s cycle)</li>
                <li>• Actuated control (sensor-based)</li>
                <li>• Random policy</li>
                <li>• Max-pressure heuristic</li>
              </ul>
            </div>
          </div>

          <div className="mt-4 rounded-lg bg-green-500/10 p-4">
            <h3 className="mb-2 font-semibold text-green-600">Expected Performance</h3>
            <p className="text-sm text-muted-foreground">
              DQN agent achieves <strong>20-30% reduction</strong> in average waiting time compared to fixed-time control,
              with <strong>15-25% higher throughput</strong> during peak hours.
            </p>
          </div>
        </section>

        {/* Resources */}
        <section className="rounded-xl border border-border bg-gradient-to-r from-blue-500/10 via-purple-500/10 to-pink-500/10 p-6">
          <div className="mb-4 flex items-center gap-3">
            <GitBranch className="h-6 w-6" />
            <h2 className="text-2xl font-bold">Additional Resources</h2>
          </div>

          <div className="grid gap-3 sm:grid-cols-2">
            <a href="https://github.com/yourusername/traffic-rl" className="rounded-lg border border-border bg-card p-4 transition-colors hover:bg-accent">
              <p className="mb-1 font-semibold">GitHub Repository</p>
              <p className="text-sm text-muted-foreground">Complete source code & MLOps pipeline</p>
            </a>
            <a href="/README_MLOPS.md" className="rounded-lg border border-border bg-card p-4 transition-colors hover:bg-accent">
              <p className="mb-1 font-semibold">MLOps Guide</p>
              <p className="text-sm text-muted-foreground">Detailed deployment instructions</p>
            </a>
            <a href="https://sumo.dlr.de/docs/" className="rounded-lg border border-border bg-card p-4 transition-colors hover:bg-accent">
              <p className="mb-1 font-semibold">SUMO Documentation</p>
              <p className="text-sm text-muted-foreground">Traffic simulator reference</p>
            </a>
            <a href="https://www.nature.com/articles/nature14236" className="rounded-lg border border-border bg-card p-4 transition-colors hover:bg-accent">
              <p className="mb-1 font-semibold">DQN Paper (Nature 2015)</p>
              <p className="text-sm text-muted-foreground">Original deep RL research</p>
            </a>
          </div>
        </section>
      </div>
    </div>
  );
}
