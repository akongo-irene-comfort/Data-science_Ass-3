import { Card } from "@/components/ui/card";
import { Settings, BarChart3, CheckCircle2, Play } from "lucide-react";

export const SimulationSection = () => {
  return (
    <section id="simulation" className="py-20 px-4 bg-muted/30">
      <div className="container mx-auto max-w-6xl">
        <div className="text-center space-y-4 mb-16">
          <h2 className="text-3xl sm:text-4xl font-bold">Simulation & Training</h2>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            Robust training pipeline with SUMO environment and comprehensive evaluation metrics
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-12">
          <Card className="p-6 space-y-4">
            <div className="flex items-center gap-3">
              <div className="p-3 bg-primary/10 rounded-lg">
                <Play className="h-6 w-6 text-primary" />
              </div>
              <h3 className="text-xl font-semibold">Environment Setup</h3>
            </div>
            <div className="space-y-3">
              <div>
                <h4 className="font-medium text-sm mb-2">SUMO (Simulation of Urban MObility)</h4>
                <ul className="space-y-1 text-sm text-muted-foreground">
                  <li>• Open-source traffic simulation platform</li>
                  <li>• Realistic vehicle dynamics and routing</li>
                  <li>• TraCI API for Python integration</li>
                  <li>• Customizable road networks and traffic patterns</li>
                </ul>
              </div>
              <div>
                <h4 className="font-medium text-sm mb-2">Gym Integration</h4>
                <ul className="space-y-1 text-sm text-muted-foreground">
                  <li>• OpenAI Gym wrapper for SUMO</li>
                  <li>• Standardized observation and action spaces</li>
                  <li>• Easy integration with RL libraries</li>
                  <li>• Support for multi-agent scenarios</li>
                </ul>
              </div>
            </div>
            <div className="bg-muted/50 p-4 rounded-lg font-mono text-xs">
              env = SumoEnvironment(net_file, route_file, ...)
            </div>
          </Card>

          <Card className="p-6 space-y-4">
            <div className="flex items-center gap-3">
              <div className="p-3 bg-primary/10 rounded-lg">
                <Settings className="h-6 w-6 text-primary" />
              </div>
              <h3 className="text-xl font-semibold">Hyperparameter Tuning</h3>
            </div>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Learning Rate (α)</span>
                <span className="font-mono">3e-4</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Discount Factor (γ)</span>
                <span className="font-mono">0.99</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Clip Parameter (ε)</span>
                <span className="font-mono">0.2</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Batch Size</span>
                <span className="font-mono">64</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Update Epochs</span>
                <span className="font-mono">10</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">GAE Lambda (λ)</span>
                <span className="font-mono">0.95</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Entropy Coefficient</span>
                <span className="font-mono">0.01</span>
              </div>
            </div>
            <p className="text-xs text-muted-foreground mt-4">
              Tuned using grid search and Optuna for optimal convergence
            </p>
          </Card>

          <Card className="p-6 space-y-4 lg:col-span-2">
            <div className="flex items-center gap-3">
              <div className="p-3 bg-primary/10 rounded-lg">
                <BarChart3 className="h-6 w-6 text-primary" />
              </div>
              <h3 className="text-xl font-semibold">Evaluation Metrics</h3>
            </div>
            <div className="grid sm:grid-cols-3 gap-6">
              <div className="space-y-2">
                <h4 className="font-medium text-sm text-primary">Performance Metrics</h4>
                <ul className="space-y-1 text-sm text-muted-foreground">
                  <li>• Average episode reward</li>
                  <li>• Total waiting time (seconds)</li>
                  <li>• Average queue length</li>
                  <li>• Vehicle throughput (vehicles/hour)</li>
                  <li>• Average vehicle speed (km/h)</li>
                </ul>
              </div>
              <div className="space-y-2">
                <h4 className="font-medium text-sm text-primary">Learning Metrics</h4>
                <ul className="space-y-1 text-sm text-muted-foreground">
                  <li>• Policy loss</li>
                  <li>• Value loss</li>
                  <li>• Entropy coefficient</li>
                  <li>• Convergence speed (episodes)</li>
                  <li>• Training stability (variance)</li>
                </ul>
              </div>
              <div className="space-y-2">
                <h4 className="font-medium text-sm text-primary">Comparison Baselines</h4>
                <ul className="space-y-1 text-sm text-muted-foreground">
                  <li>• Fixed-time control</li>
                  <li>• Actuated control (simple logic)</li>
                  <li>• Max pressure algorithm</li>
                  <li>• DQN baseline</li>
                  <li>• A3C baseline</li>
                </ul>
              </div>
            </div>
          </Card>

          <Card className="p-6 space-y-4 lg:col-span-2">
            <div className="flex items-center gap-3">
              <div className="p-3 bg-primary/10 rounded-lg">
                <CheckCircle2 className="h-6 w-6 text-primary" />
              </div>
              <h3 className="text-xl font-semibold">Expected Training Results</h3>
            </div>
            <div className="grid sm:grid-cols-2 gap-6">
              <div className="space-y-3">
                <div className="flex items-baseline gap-2">
                  <div className="text-3xl font-bold text-primary">30-40%</div>
                  <div className="text-sm text-muted-foreground">Reduction in waiting time</div>
                </div>
                <div className="flex items-baseline gap-2">
                  <div className="text-3xl font-bold text-primary">20-25%</div>
                  <div className="text-sm text-muted-foreground">Increase in throughput</div>
                </div>
              </div>
              <div className="space-y-3">
                <div className="flex items-baseline gap-2">
                  <div className="text-3xl font-bold text-primary">15-20%</div>
                  <div className="text-sm text-muted-foreground">Improvement in avg speed</div>
                </div>
                <div className="flex items-baseline gap-2">
                  <div className="text-3xl font-bold text-primary">5000</div>
                  <div className="text-sm text-muted-foreground">Episodes to convergence</div>
                </div>
              </div>
            </div>
            <p className="text-sm text-muted-foreground">
              Results compared against fixed-time baseline on a 4x4 grid network with realistic traffic patterns
            </p>
          </Card>
        </div>
      </div>
    </section>
  );
};
