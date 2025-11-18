import { Brain, Target, Zap, GitBranch, TrendingUp, Award } from "lucide-react";

export default function ProposalPage() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-muted/20">
      <div className="mx-auto max-w-6xl px-6 py-16 lg:px-8">
        {/* Header */}
        <div className="mb-12 text-center">
          <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-blue-500/20 bg-blue-500/5 px-4 py-2">
            <Brain className="h-4 w-4 text-blue-600" />
            <span className="text-sm font-medium text-blue-600">Part A: Research & Proposal (40%)</span>
          </div>
          <h1 className="mb-4 text-4xl font-bold tracking-tight sm:text-5xl">
            Reinforcement Learning Project Proposal
          </h1>
          <p className="mx-auto max-w-2xl text-lg text-muted-foreground">
            Theoretical foundation and formal problem formulation for traffic congestion optimization
          </p>
        </div>

        {/* Section 1: Problem Formulation */}
        <section className="mb-16">
          <div className="mb-8 flex items-center gap-3">
            <Target className="h-8 w-8 text-blue-600" />
            <h2 className="text-3xl font-bold">1. Problem Formulation</h2>
          </div>

          <div className="space-y-6">
            {/* Real-world Problem */}
            <div className="rounded-xl border border-border bg-card p-6">
              <h3 className="mb-4 text-xl font-semibold">Real-world Problem</h3>
              <p className="mb-4 leading-relaxed text-muted-foreground">
                Urban traffic congestion is a pervasive problem affecting cities worldwide, causing:
              </p>
              <ul className="mb-4 space-y-2">
                {[
                  "Economic losses estimated at $166 billion annually in the US alone",
                  "Average commuter wastes 54 hours per year in traffic",
                  "Increased fuel consumption and CO₂ emissions contributing to climate change",
                  "Reduced quality of life and increased stress levels"
                ].map((item, i) => (
                  <li key={i} className="flex items-start gap-2">
                    <div className="mt-2 h-1.5 w-1.5 rounded-full bg-blue-600" />
                    <span className="text-muted-foreground">{item}</span>
                  </li>
                ))}
              </ul>
              <p className="leading-relaxed text-muted-foreground">
                Traditional traffic management systems use fixed-time signal controls that cannot adapt to real-time 
                traffic conditions. This project proposes an intelligent RL-based solution that learns optimal traffic 
                control policies to minimize congestion dynamically.
              </p>
            </div>

            {/* MDP Definition */}
            <div className="rounded-xl border border-border bg-card p-6">
              <h3 className="mb-4 text-xl font-semibold">Markov Decision Process (MDP) Definition</h3>
              <p className="mb-6 leading-relaxed text-muted-foreground">
                We formalize the traffic optimization problem as an MDP tuple: <strong>(S, A, P, R, γ)</strong>
              </p>

              <div className="space-y-6">
                {/* State Space */}
                <div className="rounded-lg bg-blue-500/5 p-4">
                  <h4 className="mb-3 font-semibold text-blue-600">State Space (S)</h4>
                  <p className="mb-3 text-sm text-muted-foreground">
                    The state at time <em>t</em> captures complete traffic network information:
                  </p>
                  <div className="space-y-2 text-sm">
                    <div className="rounded bg-background/50 p-3">
                      <strong>s<sub>t</sub> = [n<sub>1</sub>, n<sub>2</sub>, ..., n<sub>k</sub>, v<sub>1</sub>, v<sub>2</sub>, ..., v<sub>k</sub>, d<sub>1</sub>, d<sub>2</sub>, ..., d<sub>k</sub>, τ]</strong>
                    </div>
                    <ul className="ml-4 space-y-1 text-muted-foreground">
                      <li>• <strong>n<sub>i</sub></strong>: Number of vehicles in lane/intersection <em>i</em></li>
                      <li>• <strong>v<sub>i</sub></strong>: Average speed of vehicles in lane <em>i</em> (km/h)</li>
                      <li>• <strong>d<sub>i</sub></strong>: Density of vehicles per meter in lane <em>i</em></li>
                      <li>• <strong>τ</strong>: Time of day (encoded as continuous value)</li>
                      <li>• <strong>k</strong>: Number of lanes/road segments in the network</li>
                    </ul>
                    <p className="mt-3 text-muted-foreground">
                      State space is continuous and high-dimensional (~50-200 dimensions for typical urban network)
                    </p>
                  </div>
                </div>

                {/* Action Space */}
                <div className="rounded-lg bg-purple-500/5 p-4">
                  <h4 className="mb-3 font-semibold text-purple-600">Action Space (A)</h4>
                  <p className="mb-3 text-sm text-muted-foreground">
                    The agent controls traffic management parameters:
                  </p>
                  <div className="space-y-2 text-sm">
                    <div className="rounded bg-background/50 p-3">
                      <strong>a<sub>t</sub> ∈ {"{"}phase<sub>1</sub>, phase<sub>2</sub>, ..., phase<sub>m</sub>{"}"}</strong>
                    </div>
                    <ul className="ml-4 space-y-1 text-muted-foreground">
                      <li>• <strong>Traffic Light Phases</strong>: Switch between predefined signal phases (green/red combinations)</li>
                      <li>• <strong>Phase Duration</strong>: Time allocation for each phase (15-120 seconds)</li>
                      <li>• <strong>Lane Control</strong>: Enable/disable reversible lanes (optional)</li>
                      <li>• <strong>Speed Limits</strong>: Dynamic speed limit adjustments (optional)</li>
                    </ul>
                    <p className="mt-3 text-muted-foreground">
                      Action space is discrete with ~8-16 possible actions per intersection
                    </p>
                  </div>
                </div>

                {/* Reward Function */}
                <div className="rounded-lg bg-green-500/5 p-4">
                  <h4 className="mb-3 font-semibold text-green-600">Reward Function (R)</h4>
                  <p className="mb-3 text-sm text-muted-foreground">
                    The reward function balances multiple objectives:
                  </p>
                  <div className="space-y-2 text-sm">
                    <div className="rounded bg-background/50 p-3">
                      <strong>R(s, a) = -α·W - β·Q - γ·E + δ·T</strong>
                    </div>
                    <ul className="ml-4 space-y-2 text-muted-foreground">
                      <li>• <strong>W</strong>: Total waiting time (sum of all vehicle wait times)</li>
                      <li>• <strong>Q</strong>: Queue length (number of stopped vehicles)</li>
                      <li>• <strong>E</strong>: CO₂ emissions (based on speed and acceleration)</li>
                      <li>• <strong>T</strong>: Throughput (number of vehicles passing through)</li>
                      <li className="mt-2">• <strong>α, β, γ, δ</strong>: Weighting coefficients (e.g., α=1.0, β=0.5, γ=0.3, δ=0.8)</li>
                    </ul>
                    <p className="mt-3 text-muted-foreground">
                      Reward is negative (cost-based) to minimize congestion metrics
                    </p>
                  </div>
                </div>

                {/* Transition Dynamics */}
                <div className="rounded-lg bg-orange-500/5 p-4">
                  <h4 className="mb-3 font-semibold text-orange-600">Transition Dynamics (P)</h4>
                  <p className="mb-3 text-sm text-muted-foreground">
                    State transitions follow stochastic traffic flow dynamics:
                  </p>
                  <div className="space-y-2 text-sm">
                    <div className="rounded bg-background/50 p-3">
                      <strong>P(s<sub>t+1</sub> | s<sub>t</sub>, a<sub>t</sub>)</strong>
                    </div>
                    <ul className="ml-4 space-y-1 text-muted-foreground">
                      <li>• Vehicle arrivals follow Poisson distribution (λ varies by time of day)</li>
                      <li>• Driver behavior modeled with Intelligent Driver Model (IDM)</li>
                      <li>• Lane changes follow MOBIL model</li>
                      <li>• Traffic light transitions affect vehicle flow deterministically</li>
                      <li>• Stochasticity from random vehicle arrivals, destinations, and driver variations</li>
                    </ul>
                    <p className="mt-3 text-muted-foreground">
                      Model-free approach: Agent learns P implicitly through experience
                    </p>
                  </div>
                </div>

                {/* Discount Factor */}
                <div className="rounded-lg bg-pink-500/5 p-4">
                  <h4 className="mb-3 font-semibold text-pink-600">Discount Factor (γ)</h4>
                  <div className="space-y-2 text-sm">
                    <div className="rounded bg-background/50 p-3">
                      <strong>γ = 0.95</strong>
                    </div>
                    <p className="mt-3 text-muted-foreground">
                      High discount factor (0.95) ensures the agent considers long-term traffic flow optimization
                      rather than myopic short-term gains. This is crucial for preventing downstream congestion.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Section 2: Algorithm Selection */}
        <section className="mb-16">
          <div className="mb-8 flex items-center gap-3">
            <GitBranch className="h-8 w-8 text-purple-600" />
            <h2 className="text-3xl font-bold">2. RL Algorithm Selection</h2>
          </div>

          <div className="space-y-6">
            {/* Chosen Algorithm */}
            <div className="rounded-xl border-2 border-purple-500/30 bg-gradient-to-br from-purple-500/5 to-blue-500/5 p-6">
              <div className="mb-4 flex items-center gap-3">
                <Award className="h-6 w-6 text-purple-600" />
                <h3 className="text-xl font-semibold">Chosen Algorithm: Deep Q-Network (DQN)</h3>
              </div>
              <p className="mb-4 leading-relaxed text-muted-foreground">
                DQN is selected for its proven effectiveness in handling high-dimensional continuous state spaces 
                with discrete actions, making it ideal for traffic signal control.
              </p>
            </div>

            {/* Algorithm Justification */}
            <div className="rounded-xl border border-border bg-card p-6">
              <h3 className="mb-4 text-xl font-semibold">Justification</h3>
              <div className="space-y-4">
                <div className="rounded-lg bg-muted/50 p-4">
                  <h4 className="mb-2 font-semibold">Why DQN?</h4>
                  <ul className="space-y-2 text-sm text-muted-foreground">
                    <li className="flex items-start gap-2">
                      <span className="text-green-600">✓</span>
                      <span><strong>Continuous State Space</strong>: Neural networks can approximate Q-values for high-dimensional continuous states (vehicle counts, speeds, densities)</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-green-600">✓</span>
                      <span><strong>Discrete Action Space</strong>: Traffic light phases are naturally discrete (8-16 actions per intersection)</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-green-600">✓</span>
                      <span><strong>Model-Free</strong>: No need to model complex traffic dynamics P(s'|s,a) explicitly</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-green-600">✓</span>
                      <span><strong>Off-Policy Learning</strong>: Efficient sample utilization through experience replay</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-green-600">✓</span>
                      <span><strong>Proven Success</strong>: DQN has achieved superhuman performance in Atari games and traffic control benchmarks</span>
                    </li>
                  </ul>
                </div>

                <div className="rounded-lg bg-muted/50 p-4">
                  <h4 className="mb-2 font-semibold">Why Not Other Algorithms?</h4>
                  <ul className="space-y-2 text-sm text-muted-foreground">
                    <li className="flex items-start gap-2">
                      <span className="text-red-600">✗</span>
                      <span><strong>Q-Learning (Tabular)</strong>: Cannot handle continuous state space (would require discretization, losing precision)</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-red-600">✗</span>
                      <span><strong>Policy Gradient (PPO/A3C)</strong>: Slower convergence and higher variance for discrete actions compared to value-based methods</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-red-600">✗</span>
                      <span><strong>SAC/DDPG</strong>: Designed for continuous action spaces; our actions are discrete (signal phases)</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-orange-600">~</span>
                      <span><strong>Actor-Critic (A2C)</strong>: Viable alternative but DQN offers better sample efficiency for our use case</span>
                    </li>
                  </ul>
                </div>
              </div>
            </div>

            {/* Deep RL Concepts */}
            <div className="rounded-xl border border-border bg-card p-6">
              <h3 className="mb-4 text-xl font-semibold">Key Deep RL Concepts</h3>
              <div className="grid gap-4 md:grid-cols-2">
                <div className="rounded-lg border border-border p-4">
                  <h4 className="mb-2 font-semibold text-blue-600">Experience Replay</h4>
                  <p className="text-sm text-muted-foreground">
                    Store transitions (s, a, r, s') in replay buffer D. Sample random minibatches for training to:
                  </p>
                  <ul className="mt-2 space-y-1 text-sm text-muted-foreground">
                    <li>• Break temporal correlations</li>
                    <li>• Improve sample efficiency</li>
                    <li>• Enable off-policy learning</li>
                    <li>• Buffer size: 100,000 transitions</li>
                  </ul>
                </div>

                <div className="rounded-lg border border-border p-4">
                  <h4 className="mb-2 font-semibold text-purple-600">Target Network</h4>
                  <p className="text-sm text-muted-foreground">
                    Separate target network Q' with frozen parameters θ⁻ to stabilize learning:
                  </p>
                  <ul className="mt-2 space-y-1 text-sm text-muted-foreground">
                    <li>• Prevents moving target problem</li>
                    <li>• Updated every C=1000 steps</li>
                    <li>• Reduces oscillations in Q-values</li>
                    <li>• Improves convergence stability</li>
                  </ul>
                </div>

                <div className="rounded-lg border border-border p-4">
                  <h4 className="mb-2 font-semibold text-green-600">Value vs Policy Functions</h4>
                  <p className="text-sm text-muted-foreground">
                    DQN is value-based, learning Q(s,a):
                  </p>
                  <ul className="mt-2 space-y-1 text-sm text-muted-foreground">
                    <li>• Q-function: Expected return from (s,a)</li>
                    <li>• Policy derived: π(s) = argmax<sub>a</sub> Q(s,a)</li>
                    <li>• More sample-efficient for discrete actions</li>
                    <li>• Direct action selection via max operation</li>
                  </ul>
                </div>

                <div className="rounded-lg border border-border p-4">
                  <h4 className="mb-2 font-semibold text-orange-600">Exploration-Exploitation</h4>
                  <p className="text-sm text-muted-foreground">
                    ε-greedy strategy for balancing exploration:
                  </p>
                  <ul className="mt-2 space-y-1 text-sm text-muted-foreground">
                    <li>• ε starts at 1.0 (full exploration)</li>
                    <li>• Decays to 0.01 over 100k steps</li>
                    <li>• Random action with probability ε</li>
                    <li>• Greedy action otherwise: argmax<sub>a</sub> Q(s,a)</li>
                  </ul>
                </div>
              </div>
            </div>

            {/* DQN Architecture */}
            <div className="rounded-xl border border-border bg-card p-6">
              <h3 className="mb-4 text-xl font-semibold">Network Architecture</h3>
              <div className="rounded-lg bg-muted/30 p-4 font-mono text-sm">
                <div className="space-y-1">
                  <div>Input Layer: state_dim (e.g., 128 dimensions)</div>
                  <div className="ml-4">↓</div>
                  <div>Dense Layer 1: 256 units + ReLU + Batch Norm</div>
                  <div className="ml-4">↓</div>
                  <div>Dense Layer 2: 256 units + ReLU + Dropout(0.2)</div>
                  <div className="ml-4">↓</div>
                  <div>Dense Layer 3: 128 units + ReLU</div>
                  <div className="ml-4">↓</div>
                  <div>Output Layer: action_dim (e.g., 8-16 Q-values)</div>
                </div>
              </div>
              <p className="mt-3 text-sm text-muted-foreground">
                <strong>Loss Function:</strong> MSE between predicted Q(s,a) and target y = r + γ·max<sub>a'</sub> Q'(s',a')
              </p>
              <p className="mt-1 text-sm text-muted-foreground">
                <strong>Optimizer:</strong> Adam with learning rate 0.001
              </p>
            </div>
          </div>
        </section>

        {/* Section 3: Simulation & Training */}
        <section className="mb-16">
          <div className="mb-8 flex items-center gap-3">
            <Zap className="h-8 w-8 text-green-600" />
            <h2 className="text-3xl font-bold">3. Simulation & Training</h2>
          </div>

          <div className="space-y-6">
            {/* Environment Setup */}
            <div className="rounded-xl border border-border bg-card p-6">
              <h3 className="mb-4 text-xl font-semibold">Environment Setup</h3>
              <div className="space-y-4">
                <div>
                  <h4 className="mb-2 font-semibold text-blue-600">SUMO (Simulation of Urban MObility)</h4>
                  <p className="mb-2 text-sm text-muted-foreground">
                    Open-source microscopic traffic simulator chosen for:
                  </p>
                  <ul className="space-y-1 text-sm text-muted-foreground">
                    <li className="flex items-start gap-2">
                      <span>•</span>
                      <span>Realistic vehicle dynamics (car-following, lane-changing models)</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span>•</span>
                      <span>Support for large-scale road networks (import from OpenStreetMap)</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span>•</span>
                      <span>Traffic light control API (TraCI - Traffic Control Interface)</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span>•</span>
                      <span>Emission models for CO₂ calculations</span>
                    </li>
                  </ul>
                </div>

                <div className="rounded-lg bg-muted/30 p-4">
                  <h4 className="mb-2 font-semibold text-purple-600">Network Configuration</h4>
                  <ul className="space-y-1 text-sm text-muted-foreground">
                    <li>• <strong>Scenario:</strong> 4x4 grid network (16 intersections)</li>
                    <li>• <strong>Road Length:</strong> 300m between intersections</li>
                    <li>• <strong>Lanes:</strong> 2-3 lanes per direction</li>
                    <li>• <strong>Speed Limits:</strong> 50 km/h (urban)</li>
                    <li>• <strong>Traffic Demand:</strong> 1000-3000 vehicles/hour (peak hours)</li>
                    <li>• <strong>Simulation Step:</strong> 1 second</li>
                    <li>• <strong>Episode Length:</strong> 3600 seconds (1 hour)</li>
                  </ul>
                </div>

                <div className="rounded-lg bg-muted/30 p-4">
                  <h4 className="mb-2 font-semibold text-green-600">Gym Wrapper</h4>
                  <p className="text-sm text-muted-foreground">
                    Custom OpenAI Gym environment wrapping SUMO:
                  </p>
                  <ul className="mt-2 space-y-1 text-sm text-muted-foreground">
                    <li>• <strong>Observation Space:</strong> Box(low=0, high=inf, shape=(128,))</li>
                    <li>• <strong>Action Space:</strong> Discrete(8) [signal phases]</li>
                    <li>• <strong>Reset:</strong> Randomize initial traffic conditions</li>
                    <li>• <strong>Step:</strong> Execute action for 10 seconds, return next state & reward</li>
                  </ul>
                </div>
              </div>
            </div>

            {/* Hyperparameters */}
            <div className="rounded-xl border border-border bg-card p-6">
              <h3 className="mb-4 text-xl font-semibold">Hyperparameter Configuration</h3>
              <div className="grid gap-4 sm:grid-cols-2">
                <div className="rounded-lg bg-blue-500/5 p-4">
                  <h4 className="mb-3 font-semibold text-sm">Learning Parameters</h4>
                  <ul className="space-y-1 text-sm text-muted-foreground">
                    <li>• Learning Rate (α): <strong>0.001</strong></li>
                    <li>• Discount Factor (γ): <strong>0.95</strong></li>
                    <li>• Batch Size: <strong>64</strong></li>
                    <li>• Replay Buffer Size: <strong>100,000</strong></li>
                  </ul>
                </div>
                <div className="rounded-lg bg-purple-500/5 p-4">
                  <h4 className="mb-3 font-semibold text-sm">Exploration Parameters</h4>
                  <ul className="space-y-1 text-sm text-muted-foreground">
                    <li>• ε Start: <strong>1.0</strong></li>
                    <li>• ε End: <strong>0.01</strong></li>
                    <li>• ε Decay Steps: <strong>100,000</strong></li>
                    <li>• ε Decay: <strong>Linear</strong></li>
                  </ul>
                </div>
                <div className="rounded-lg bg-green-500/5 p-4">
                  <h4 className="mb-3 font-semibold text-sm">Network Parameters</h4>
                  <ul className="space-y-1 text-sm text-muted-foreground">
                    <li>• Hidden Layers: <strong>[256, 256, 128]</strong></li>
                    <li>• Activation: <strong>ReLU</strong></li>
                    <li>• Dropout Rate: <strong>0.2</strong></li>
                    <li>• Target Update Freq: <strong>1000 steps</strong></li>
                  </ul>
                </div>
                <div className="rounded-lg bg-orange-500/5 p-4">
                  <h4 className="mb-3 font-semibold text-sm">Training Parameters</h4>
                  <ul className="space-y-1 text-sm text-muted-foreground">
                    <li>• Total Episodes: <strong>1,000</strong></li>
                    <li>• Max Steps/Episode: <strong>3,600</strong></li>
                    <li>• Warmup Steps: <strong>10,000</strong></li>
                    <li>• Training Frequency: <strong>Every 4 steps</strong></li>
                  </ul>
                </div>
              </div>
            </div>

            {/* Evaluation Metrics */}
            <div className="rounded-xl border border-border bg-card p-6">
              <h3 className="mb-4 text-xl font-semibold">Evaluation Metrics</h3>
              <div className="space-y-4">
                <div className="rounded-lg bg-muted/30 p-4">
                  <h4 className="mb-2 font-semibold text-blue-600">Primary Metrics</h4>
                  <ul className="space-y-2 text-sm text-muted-foreground">
                    <li className="flex items-start gap-2">
                      <TrendingUp className="mt-0.5 h-4 w-4 flex-shrink-0 text-blue-600" />
                      <span><strong>Average Episode Reward:</strong> Cumulative reward per episode (higher is better)</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <TrendingUp className="mt-0.5 h-4 w-4 flex-shrink-0 text-blue-600" />
                      <span><strong>Average Waiting Time:</strong> Mean wait time per vehicle (lower is better, target: &lt;60s)</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <TrendingUp className="mt-0.5 h-4 w-4 flex-shrink-0 text-blue-600" />
                      <span><strong>Throughput:</strong> Number of vehicles completing trips per hour (higher is better)</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <TrendingUp className="mt-0.5 h-4 w-4 flex-shrink-0 text-blue-600" />
                      <span><strong>Queue Length:</strong> Average number of stopped vehicles (lower is better)</span>
                    </li>
                  </ul>
                </div>

                <div className="rounded-lg bg-muted/30 p-4">
                  <h4 className="mb-2 font-semibold text-purple-600">Secondary Metrics</h4>
                  <ul className="space-y-1 text-sm text-muted-foreground">
                    <li>• <strong>Convergence Speed:</strong> Episodes until reward plateaus (target: &lt;500)</li>
                    <li>• <strong>CO₂ Emissions:</strong> Total emissions per episode (kg)</li>
                    <li>• <strong>Travel Time:</strong> Average trip duration (seconds)</li>
                    <li>• <strong>Fuel Consumption:</strong> Average liters per vehicle</li>
                  </ul>
                </div>

                <div className="rounded-lg bg-green-500/5 p-4">
                  <h4 className="mb-2 font-semibold text-green-600">Baseline Comparison</h4>
                  <p className="mb-2 text-sm text-muted-foreground">
                    Compare DQN agent against:
                  </p>
                  <ul className="space-y-1 text-sm text-muted-foreground">
                    <li>• <strong>Fixed-Time Control:</strong> Traditional 60-second cycles</li>
                    <li>• <strong>Actuated Control:</strong> Vehicle-actuated signal timing</li>
                    <li>• <strong>Random Policy:</strong> Random action selection</li>
                    <li>• <strong>Max-Pressure:</strong> Classic traffic-responsive heuristic</li>
                  </ul>
                  <p className="mt-2 text-sm text-muted-foreground">
                    <strong>Expected Improvement:</strong> 20-30% reduction in average waiting time compared to fixed-time control
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Summary */}
        <section className="rounded-2xl border-2 border-primary/20 bg-gradient-to-br from-blue-500/5 to-purple-500/5 p-8">
          <h2 className="mb-4 text-2xl font-bold">Proposal Summary</h2>
          <p className="mb-4 leading-relaxed text-muted-foreground">
            This proposal presents a comprehensive RL-based solution for traffic congestion optimization using Deep Q-Networks. 
            The problem is rigorously formulated as an MDP with well-defined state/action spaces and a multi-objective reward function. 
            DQN is justified as the optimal algorithm choice for handling continuous states with discrete actions in a model-free setting.
          </p>
          <p className="leading-relaxed text-muted-foreground">
            The SUMO-based simulation environment provides a realistic testbed with proper evaluation metrics to validate the approach. 
            Expected outcomes include 20-30% reduction in waiting times, demonstrating the potential of deep RL for intelligent 
            traffic management systems.
          </p>
        </section>
      </div>
    </div>
  );
}
