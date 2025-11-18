import { Card } from "@/components/ui/card";
import { Cpu, Database, TrendingUp, Search } from "lucide-react";

export const AlgorithmSection = () => {
  return (
    <section id="algorithm" className="py-20 px-4">
      <div className="container mx-auto max-w-6xl">
        <div className="text-center space-y-4 mb-16">
          <h2 className="text-3xl sm:text-4xl font-bold">Algorithm Selection: PPO</h2>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            Proximal Policy Optimization for stable and efficient traffic control learning
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-12">
          <Card className="p-6 space-y-4 lg:col-span-2">
            <h3 className="text-xl font-semibold">Why PPO?</h3>
            <div className="grid sm:grid-cols-2 gap-6">
              <div className="space-y-3">
                <h4 className="font-medium text-primary">✓ Advantages</h4>
                <ul className="space-y-2 text-sm text-muted-foreground">
                  <li>• Handles large and continuous action spaces effectively</li>
                  <li>• More stable than vanilla policy gradient methods</li>
                  <li>• Sample-efficient compared to DQN in complex environments</li>
                  <li>• Well-supported in modern RL libraries (Stable-Baselines3, Ray RLlib)</li>
                  <li>• Proven success in real-world control problems</li>
                </ul>
              </div>
              <div className="space-y-3">
                <h4 className="font-medium text-primary">✓ Suitability for Traffic Control</h4>
                <ul className="space-y-2 text-sm text-muted-foreground">
                  <li>• Traffic signals have continuous timing parameters</li>
                  <li>• Requires stable learning in noisy environments</li>
                  <li>• Benefits from on-policy learning with recent data</li>
                  <li>• Needs to balance exploration vs exploitation carefully</li>
                  <li>• Multi-agent coordination for multiple intersections</li>
                </ul>
              </div>
            </div>
          </Card>

          <Card className="p-6 space-y-4">
            <div className="flex items-center gap-3 mb-2">
              <div className="p-3 bg-primary/10 rounded-lg">
                <Database className="h-6 w-6 text-primary" />
              </div>
              <h3 className="text-xl font-semibold">Experience Replay</h3>
            </div>
            <p className="text-sm text-muted-foreground leading-relaxed">
              While PPO is primarily on-policy, we implement a replay buffer to store recent 
              experiences for multiple gradient updates. This improves sample efficiency while 
              maintaining the stability guarantees of PPO through clipped surrogate objectives.
            </p>
            <div className="bg-muted/50 p-4 rounded-lg font-mono text-xs">
              buffer.store(state, action, reward, next_state)
            </div>
          </Card>

          <Card className="p-6 space-y-4">
            <div className="flex items-center gap-3 mb-2">
              <div className="p-3 bg-primary/10 rounded-lg">
                <Cpu className="h-6 w-6 text-primary" />
              </div>
              <h3 className="text-xl font-semibold">Target Networks</h3>
            </div>
            <p className="text-sm text-muted-foreground leading-relaxed">
              PPO uses a separate value network that is updated more slowly than the policy network. 
              This stabilizes training by providing consistent value estimates for advantage calculation, 
              reducing oscillations during optimization.
            </p>
            <div className="bg-muted/50 p-4 rounded-lg font-mono text-xs">
              advantage = reward + γ * V(s') - V(s)
            </div>
          </Card>

          <Card className="p-6 space-y-4">
            <div className="flex items-center gap-3 mb-2">
              <div className="p-3 bg-primary/10 rounded-lg">
                <TrendingUp className="h-6 w-6 text-primary" />
              </div>
              <h3 className="text-xl font-semibold">Policy vs Value Functions</h3>
            </div>
            <p className="text-sm text-muted-foreground leading-relaxed">
              PPO is an actor-critic method: the <strong>policy network (actor)</strong> selects actions, 
              while the <strong>value network (critic)</strong> estimates state values. The critic guides 
              the actor by providing advantage estimates for policy gradient updates.
            </p>
            <div className="bg-muted/50 p-4 rounded-lg font-mono text-xs">
              π(a|s): Policy | V(s): Value Function
            </div>
          </Card>

          <Card className="p-6 space-y-4">
            <div className="flex items-center gap-3 mb-2">
              <div className="p-3 bg-primary/10 rounded-lg">
                <Search className="h-6 w-6 text-primary" />
              </div>
              <h3 className="text-xl font-semibold">Exploration-Exploitation</h3>
            </div>
            <p className="text-sm text-muted-foreground leading-relaxed">
              PPO maintains exploration through entropy regularization in the policy loss. This encourages 
              the agent to try diverse actions early in training, while gradually becoming more deterministic 
              as it learns optimal traffic control strategies.
            </p>
            <div className="bg-muted/50 p-4 rounded-lg font-mono text-xs">
              L = L_CLIP - c₁ * L_VF + c₂ * H(π)
            </div>
          </Card>
        </div>

        <Card className="p-8 bg-primary/5 border-primary/20">
          <h3 className="text-xl font-semibold mb-4">PPO Update Rule</h3>
          <p className="text-muted-foreground mb-4 leading-relaxed">
            PPO optimizes a clipped surrogate objective to prevent large policy updates that could destabilize training:
          </p>
          <div className="bg-background p-6 rounded-lg border font-mono text-sm overflow-x-auto">
            <div className="space-y-2">
              <div>L<sup>CLIP</sup>(θ) = Ê<sub>t</sub>[min(r<sub>t</sub>(θ)Â<sub>t</sub>, clip(r<sub>t</sub>(θ), 1-ε, 1+ε)Â<sub>t</sub>)]</div>
              <div className="text-muted-foreground text-xs mt-4">
                where r<sub>t</sub>(θ) = π<sub>θ</sub>(a<sub>t</sub>|s<sub>t</sub>) / π<sub>θ_old</sub>(a<sub>t</sub>|s<sub>t</sub>) and ε is the clipping parameter (typically 0.2)
              </div>
            </div>
          </div>
        </Card>
      </div>
    </section>
  );
};
