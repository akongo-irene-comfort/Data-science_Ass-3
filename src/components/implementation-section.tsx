import { Card } from "@/components/ui/card";
import { Code, Server, Package, FileCode } from "lucide-react";

export const ImplementationSection = () => {
  return (
    <section id="implementation" className="py-20 px-4">
      <div className="container mx-auto max-w-6xl">
        <div className="text-center space-y-4 mb-16">
          <h2 className="text-3xl sm:text-4xl font-bold">Implementation & Architecture</h2>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            Production-ready code structure with model serving and inference API
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-12">
          <Card className="p-6 space-y-4">
            <div className="flex items-center gap-3">
              <div className="p-3 bg-primary/10 rounded-lg">
                <Code className="h-6 w-6 text-primary" />
              </div>
              <h3 className="text-xl font-semibold">Project Structure</h3>
            </div>
            <div className="bg-muted/50 p-4 rounded-lg font-mono text-xs space-y-1">
              <div>traffic-rl/</div>
              <div className="ml-3">├── src/</div>
              <div className="ml-6">├── agents/         # PPO agent implementation</div>
              <div className="ml-6">├── envs/           # SUMO environment wrapper</div>
              <div className="ml-6">├── models/         # Neural network architectures</div>
              <div className="ml-6">├── utils/          # Helper functions</div>
              <div className="ml-6">└── train.py        # Training script</div>
              <div className="ml-3">├── api/</div>
              <div className="ml-6">├── app.py          # FastAPI inference server</div>
              <div className="ml-6">├── schemas.py      # Request/response models</div>
              <div className="ml-6">└── inference.py    # Model loading & prediction</div>
              <div className="ml-3">├── configs/        # Hyperparameter configs</div>
              <div className="ml-3">├── notebooks/      # Jupyter analysis notebooks</div>
              <div className="ml-3">├── tests/          # Unit and integration tests</div>
              <div className="ml-3">├── docker/         # Containerization files</div>
              <div className="ml-3">└── requirements.txt</div>
            </div>
          </Card>

          <Card className="p-6 space-y-4">
            <div className="flex items-center gap-3">
              <div className="p-3 bg-primary/10 rounded-lg">
                <Package className="h-6 w-6 text-primary" />
              </div>
              <h3 className="text-xl font-semibold">Technology Stack</h3>
            </div>
            <div className="space-y-3">
              <div>
                <h4 className="font-medium text-sm mb-2">Training</h4>
                <ul className="space-y-1 text-sm text-muted-foreground">
                  <li>• Python 3.9+</li>
                  <li>• PyTorch 2.0 (Deep Learning)</li>
                  <li>• Stable-Baselines3 (RL algorithms)</li>
                  <li>• SUMO 1.15+ (Traffic simulation)</li>
                  <li>• Gym / Gymnasium (Environment interface)</li>
                </ul>
              </div>
              <div>
                <h4 className="font-medium text-sm mb-2">Serving & Deployment</h4>
                <ul className="space-y-1 text-sm text-muted-foreground">
                  <li>• FastAPI (REST API)</li>
                  <li>• Docker & Docker Compose</li>
                  <li>• Redis (Caching)</li>
                  <li>• Prometheus + Grafana (Monitoring)</li>
                </ul>
              </div>
            </div>
          </Card>

          <Card className="p-6 space-y-4 lg:col-span-2">
            <div className="flex items-center gap-3">
              <div className="p-3 bg-primary/10 rounded-lg">
                <FileCode className="h-6 w-6 text-primary" />
              </div>
              <h3 className="text-xl font-semibold">Model Saving & Loading</h3>
            </div>
            <div className="grid sm:grid-cols-3 gap-4">
              <div className="space-y-2">
                <h4 className="font-medium text-sm text-primary">PyTorch Save</h4>
                <div className="bg-muted/50 p-3 rounded-lg font-mono text-xs">
                  <div>torch.save({"{"}</div>
                  <div className="ml-2">'policy': model.state_dict(),</div>
                  <div className="ml-2">'optimizer': opt.state_dict()</div>
                  <div>{"}"}, 'model.pt')</div>
                </div>
                <p className="text-xs text-muted-foreground">Standard PyTorch checkpoint</p>
              </div>
              <div className="space-y-2">
                <h4 className="font-medium text-sm text-primary">TorchScript JIT</h4>
                <div className="bg-muted/50 p-3 rounded-lg font-mono text-xs">
                  <div>traced = torch.jit.trace(</div>
                  <div className="ml-2">model, example_input</div>
                  <div>)</div>
                  <div>traced.save('model_jit.pt')</div>
                </div>
                <p className="text-xs text-muted-foreground">Optimized for inference</p>
              </div>
              <div className="space-y-2">
                <h4 className="font-medium text-sm text-primary">ONNX Export</h4>
                <div className="bg-muted/50 p-3 rounded-lg font-mono text-xs">
                  <div>torch.onnx.export(</div>
                  <div className="ml-2">model, example_input,</div>
                  <div className="ml-2">'model.onnx'</div>
                  <div>)</div>
                </div>
                <p className="text-xs text-muted-foreground">Cross-platform compatibility</p>
              </div>
            </div>
          </Card>

          <Card className="p-6 space-y-4 lg:col-span-2">
            <div className="flex items-center gap-3">
              <div className="p-3 bg-primary/10 rounded-lg">
                <Server className="h-6 w-6 text-primary" />
              </div>
              <h3 className="text-xl font-semibold">Inference API</h3>
            </div>
            <div className="space-y-4">
              <div>
                <h4 className="font-medium text-sm mb-2">FastAPI Endpoint Example</h4>
                <div className="bg-muted/50 p-4 rounded-lg font-mono text-xs space-y-1">
                  <div className="text-muted-foreground"># POST /api/predict</div>
                  <div>@app.post("/api/predict")</div>
                  <div>async def predict(state: TrafficState):</div>
                  <div className="ml-4">obs = preprocess(state)</div>
                  <div className="ml-4">action = model.predict(obs)</div>
                  <div className="ml-4">return {"{"}"action": action, "confidence": conf{"}"}</div>
                </div>
              </div>
              <div className="grid sm:grid-cols-2 gap-4">
                <div>
                  <h4 className="font-medium text-sm mb-2">Request Schema</h4>
                  <div className="bg-muted/50 p-3 rounded-lg font-mono text-xs">
                    <div>{"{"}</div>
                    <div className="ml-2">"intersection_id": "int01",</div>
                    <div className="ml-2">"queue_lengths": [12, 8, 5, 10],</div>
                    <div className="ml-2">"avg_speeds": [25.3, 30.1, ...],</div>
                    <div className="ml-2">"time_of_day": "peak"</div>
                    <div>{"}"}</div>
                  </div>
                </div>
                <div>
                  <h4 className="font-medium text-sm mb-2">Response Schema</h4>
                  <div className="bg-muted/50 p-3 rounded-lg font-mono text-xs">
                    <div>{"{"}</div>
                    <div className="ml-2">"action": "green_phase_2",</div>
                    <div className="ml-2">"duration": 45,</div>
                    <div className="ml-2">"confidence": 0.87,</div>
                    <div className="ml-2">"latency_ms": 12</div>
                    <div>{"}"}</div>
                  </div>
                </div>
              </div>
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <CheckCircle2 className="h-4 w-4 text-primary" />
                <span>Target latency: &lt;50ms for real-time traffic control</span>
              </div>
            </div>
          </Card>
        </div>
      </div>
    </section>
  );
};

const CheckCircle2 = ({ className }: { className?: string }) => (
  <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
  </svg>
);
