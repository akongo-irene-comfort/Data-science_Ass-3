import { Card } from "@/components/ui/card";
import { Brain, Target, Zap, GitBranch } from "lucide-react";

export const ProblemSection = () => {
  return (
    <section id="problem" className="py-20 px-4 bg-muted/30">
      <div className="container mx-auto max-w-6xl">
        <div className="text-center space-y-4 mb-16">
          <h2 className="text-3xl sm:text-4xl font-bold">Problem Formulation as MDP</h2>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            Translating real-world traffic congestion into a formal Markov Decision Process
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-12">
          <Card className="p-6 space-y-4">
            <div className="flex items-center gap-3">
              <div className="p-3 bg-primary/10 rounded-lg">
                <Brain className="h-6 w-6 text-primary" />
              </div>
              <h3 className="text-xl font-semibold">State Space (S)</h3>
            </div>
            <ul className="space-y-2 text-muted-foreground">
              <li className="flex items-start gap-2">
                <span className="text-primary mt-1">•</span>
                <span>Number of cars at each intersection and lane</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-primary mt-1">•</span>
                <span>Average speed on each road segment</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-primary mt-1">•</span>
                <span>Time of day (peak/off-peak hours)</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-primary mt-1">•</span>
                <span>Queue lengths at traffic lights</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-primary mt-1">•</span>
                <span>Lane density and occupancy rates</span>
              </li>
            </ul>
          </Card>

          <Card className="p-6 space-y-4">
            <div className="flex items-center gap-3">
              <div className="p-3 bg-primary/10 rounded-lg">
                <Target className="h-6 w-6 text-primary" />
              </div>
              <h3 className="text-xl font-semibold">Action Space (A)</h3>
            </div>
            <ul className="space-y-2 text-muted-foreground">
              <li className="flex items-start gap-2">
                <span className="text-primary mt-1">•</span>
                <span>Duration of green/red light phases</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-primary mt-1">•</span>
                <span>Sequencing order of traffic light signals</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-primary mt-1">•</span>
                <span>Dynamic phase adjustments per intersection</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-primary mt-1">•</span>
                <span>Optional: Open/close specific lanes</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-primary mt-1">•</span>
                <span>Optional: Speed limit adjustments</span>
              </li>
            </ul>
          </Card>

          <Card className="p-6 space-y-4">
            <div className="flex items-center gap-3">
              <div className="p-3 bg-primary/10 rounded-lg">
                <Zap className="h-6 w-6 text-primary" />
              </div>
              <h3 className="text-xl font-semibold">Reward Function (R)</h3>
            </div>
            <ul className="space-y-2 text-muted-foreground">
              <li className="flex items-start gap-2">
                <span className="text-primary mt-1">•</span>
                <span><strong>Minimize:</strong> Total waiting time across all vehicles</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-primary mt-1">•</span>
                <span><strong>Minimize:</strong> Queue lengths at intersections</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-primary mt-1">•</span>
                <span><strong>Maximize:</strong> Average vehicle speed and throughput</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-primary mt-1">•</span>
                <span><strong>Penalty:</strong> Gridlock conditions and excessive delays</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-primary mt-1">•</span>
                <span>Weighted combination optimized for fairness</span>
              </li>
            </ul>
          </Card>

          <Card className="p-6 space-y-4">
            <div className="flex items-center gap-3">
              <div className="p-3 bg-primary/10 rounded-lg">
                <GitBranch className="h-6 w-6 text-primary" />
              </div>
              <h3 className="text-xl font-semibold">Transition Dynamics (P)</h3>
            </div>
            <ul className="space-y-2 text-muted-foreground">
              <li className="flex items-start gap-2">
                <span className="text-primary mt-1">•</span>
                <span>Environment: SUMO (Simulation of Urban MObility)</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-primary mt-1">•</span>
                <span>TraCI API for Python integration</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-primary mt-1">•</span>
                <span>Stochastic transitions based on traffic patterns</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-primary mt-1">•</span>
                <span>Next state depends on current state and actions</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-primary mt-1">•</span>
                <span>Realistic vehicle behavior and routing</span>
              </li>
            </ul>
          </Card>
        </div>

        <Card className="p-8 bg-primary/5 border-primary/20">
          <h3 className="text-xl font-semibold mb-4">Why This Formulation?</h3>
          <p className="text-muted-foreground leading-relaxed">
            Traditional traffic light systems use fixed timing or simple rule-based control. By formulating 
            traffic management as an MDP, we enable the agent to learn adaptive policies that account for 
            dynamic traffic patterns, time-of-day variations, and unexpected congestion. The reward function 
            balances multiple objectives (reduced wait time, increased throughput, fairness) while the 
            simulation environment provides a safe testing ground before real-world deployment.
          </p>
        </Card>
      </div>
    </section>
  );
};
