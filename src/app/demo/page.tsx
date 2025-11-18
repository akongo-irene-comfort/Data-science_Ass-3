"use client";

import { useState, useEffect } from "react";
import { Play, Pause, RotateCcw, TrendingUp, Clock, Car, AlertCircle } from "lucide-react";

type TrafficPhase = "NS-Green" | "NS-Left" | "EW-Green" | "EW-Left" | "All-Red";

interface SimulationState {
  northCars: number;
  southCars: number;
  eastCars: number;
  westCars: number;
  currentPhase: TrafficPhase;
  avgWaitTime: number;
  throughput: number;
  episode: number;
  reward: number;
}

export default function DemoPage() {
  const [isRunning, setIsRunning] = useState(false);
  const [state, setState] = useState<SimulationState>({
    northCars: 12,
    southCars: 8,
    eastCars: 15,
    westCars: 10,
    currentPhase: "NS-Green",
    avgWaitTime: 45,
    throughput: 120,
    episode: 0,
    reward: -150,
  });
  const [metrics, setMetrics] = useState({
    totalWaitTime: 0,
    totalThroughput: 0,
    episodes: 0,
  });

  useEffect(() => {
    if (!isRunning) return;

    const interval = setInterval(() => {
      setState((prev) => {
        // Simulate traffic dynamics
        const newState = { ...prev };
        
        // Random arrivals
        if (Math.random() > 0.7) {
          const direction = Math.floor(Math.random() * 4);
          if (direction === 0) newState.northCars = Math.min(prev.northCars + 1, 30);
          if (direction === 1) newState.southCars = Math.min(prev.southCars + 1, 30);
          if (direction === 2) newState.eastCars = Math.min(prev.eastCars + 1, 30);
          if (direction === 3) newState.westCars = Math.min(prev.westCars + 1, 30);
        }

        // Process cars based on current phase
        if (newState.currentPhase === "NS-Green") {
          newState.northCars = Math.max(prev.northCars - 2, 0);
          newState.southCars = Math.max(prev.southCars - 2, 0);
          newState.throughput = prev.throughput + 4;
        } else if (newState.currentPhase === "EW-Green") {
          newState.eastCars = Math.max(prev.eastCars - 2, 0);
          newState.westCars = Math.max(prev.westCars - 2, 0);
          newState.throughput = prev.throughput + 4;
        }

        // DQN-based phase selection (simplified)
        const totalNS = newState.northCars + newState.southCars;
        const totalEW = newState.eastCars + newState.westCars;
        
        if (Math.random() > 0.85) {
          // Agent decides to switch
          if (totalEW > totalNS && newState.currentPhase !== "EW-Green") {
            newState.currentPhase = "EW-Green";
          } else if (totalNS > totalEW && newState.currentPhase !== "NS-Green") {
            newState.currentPhase = "NS-Green";
          }
        }

        // Calculate metrics
        const totalCars = newState.northCars + newState.southCars + newState.eastCars + newState.westCars;
        newState.avgWaitTime = totalCars * 2 + Math.random() * 10;
        newState.reward = -(totalCars * 1.5 + newState.avgWaitTime * 0.5);

        return newState;
      });

      setMetrics((prev) => ({
        totalWaitTime: prev.totalWaitTime + state.avgWaitTime,
        totalThroughput: prev.totalThroughput + state.throughput,
        episodes: prev.episodes + 1,
      }));
    }, 500);

    return () => clearInterval(interval);
  }, [isRunning, state.avgWaitTime, state.throughput]);

  const handleReset = () => {
    setIsRunning(false);
    setState({
      northCars: 12,
      southCars: 8,
      eastCars: 15,
      westCars: 10,
      currentPhase: "NS-Green",
      avgWaitTime: 45,
      throughput: 120,
      episode: 0,
      reward: -150,
    });
    setMetrics({
      totalWaitTime: 0,
      totalThroughput: 0,
      episodes: 0,
    });
  };

  const getPhaseColor = (phase: TrafficPhase) => {
    if (phase.includes("Green")) return "bg-green-500";
    if (phase === "All-Red") return "bg-red-500";
    return "bg-yellow-500";
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-muted/20">
      <div className="mx-auto max-w-7xl px-6 py-8 lg:px-8">
        {/* Header */}
        <div className="mb-6 text-center">
          <div className="mb-2 inline-flex items-center gap-2 rounded-full border border-green-500/20 bg-green-500/5 px-3 py-1.5">
            <Play className="h-3.5 w-3.5 text-green-600" />
            <span className="text-xs font-medium text-green-600">Interactive Simulation Demo</span>
          </div>
          <h1 className="mb-2 text-3xl font-bold tracking-tight sm:text-4xl">
            Traffic Optimization in Action
          </h1>
          <p className="mx-auto max-w-2xl text-sm text-muted-foreground">
            Watch the DQN agent dynamically control traffic lights to minimize congestion in real-time
          </p>
        </div>

        <div className="grid gap-4 lg:grid-cols-3">
          {/* Main Visualization */}
          <div className="lg:col-span-2">
            <div className="rounded-xl border border-border bg-card p-4">
              <div className="mb-3 flex items-center justify-between">
                <h2 className="text-lg font-semibold">Intersection Simulation</h2>
                <div className="flex gap-2">
                  <button
                    onClick={() => setIsRunning(!isRunning)}
                    className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm font-semibold transition-colors ${
                      isRunning
                        ? "bg-orange-500 text-white hover:bg-orange-600"
                        : "bg-green-500 text-white hover:bg-green-600"
                    }`}
                  >
                    {isRunning ? (
                      <>
                        <Pause className="h-3.5 w-3.5" />
                        Pause
                      </>
                    ) : (
                      <>
                        <Play className="h-3.5 w-3.5" />
                        Start
                      </>
                    )}
                  </button>
                  <button
                    onClick={handleReset}
                    className="flex items-center gap-1.5 rounded-lg border border-border bg-background px-3 py-1.5 text-sm font-semibold transition-colors hover:bg-accent"
                  >
                    <RotateCcw className="h-3.5 w-3.5" />
                    Reset
                  </button>
                </div>
              </div>

              {/* Traffic Intersection Visualization - More Compact */}
              <div className="relative aspect-square w-full rounded-lg bg-slate-900 p-4">
                {/* Roads */}
                <div className="absolute left-1/2 top-0 h-full w-20 -translate-x-1/2 bg-slate-700" />
                <div className="absolute left-0 top-1/2 h-20 w-full -translate-y-1/2 bg-slate-700" />
                
                {/* Center intersection */}
                <div className="absolute left-1/2 top-1/2 h-20 w-20 -translate-x-1/2 -translate-y-1/2 bg-slate-800" />

                {/* Traffic Lights - Smaller */}
                <div className={`absolute left-1/2 top-[12%] h-3 w-3 -translate-x-1/2 rounded-full ${
                  state.currentPhase.includes("NS") ? "bg-green-500 shadow-lg shadow-green-500/50" : "bg-red-500 shadow-lg shadow-red-500/50"
                }`} />
                <div className={`absolute left-1/2 bottom-[12%] h-3 w-3 -translate-x-1/2 rounded-full ${
                  state.currentPhase.includes("NS") ? "bg-green-500 shadow-lg shadow-green-500/50" : "bg-red-500 shadow-lg shadow-red-500/50"
                }`} />
                <div className={`absolute left-[12%] top-1/2 h-3 w-3 -translate-y-1/2 rounded-full ${
                  state.currentPhase.includes("EW") ? "bg-green-500 shadow-lg shadow-green-500/50" : "bg-red-500 shadow-lg shadow-red-500/50"
                }`} />
                <div className={`absolute right-[12%] top-1/2 h-3 w-3 -translate-y-1/2 rounded-full ${
                  state.currentPhase.includes("EW") ? "bg-green-500 shadow-lg shadow-green-500/50" : "bg-red-500 shadow-lg shadow-red-500/50"
                }`} />

                {/* Cars - North - More Compact */}
                <div className="absolute left-1/2 top-1 flex -translate-x-1/2 flex-col gap-0.5">
                  {Array.from({ length: Math.min(state.northCars, 10) }).map((_, i) => (
                    <div key={i} className="h-2.5 w-5 rounded bg-blue-500" />
                  ))}
                  {state.northCars > 10 && (
                    <div className="text-[10px] text-white">+{state.northCars - 10}</div>
                  )}
                </div>

                {/* Cars - South - More Compact */}
                <div className="absolute bottom-1 left-1/2 flex -translate-x-1/2 flex-col-reverse gap-0.5">
                  {Array.from({ length: Math.min(state.southCars, 10) }).map((_, i) => (
                    <div key={i} className="h-2.5 w-5 rounded bg-blue-500" />
                  ))}
                  {state.southCars > 10 && (
                    <div className="text-[10px] text-white">+{state.southCars - 10}</div>
                  )}
                </div>

                {/* Cars - East - More Compact */}
                <div className="absolute right-1 top-1/2 flex -translate-y-1/2 flex-row-reverse gap-0.5">
                  {Array.from({ length: Math.min(state.eastCars, 10) }).map((_, i) => (
                    <div key={i} className="h-5 w-2.5 rounded bg-blue-500" />
                  ))}
                  {state.eastCars > 10 && (
                    <div className="text-[10px] text-white">+{state.eastCars - 10}</div>
                  )}
                </div>

                {/* Cars - West - More Compact */}
                <div className="absolute left-1 top-1/2 flex -translate-y-1/2 gap-0.5">
                  {Array.from({ length: Math.min(state.westCars, 10) }).map((_, i) => (
                    <div key={i} className="h-5 w-2.5 rounded bg-blue-500" />
                  ))}
                  {state.westCars > 10 && (
                    <div className="text-[10px] text-white">+{state.westCars - 10}</div>
                  )}
                </div>

                {/* Current Phase Display */}
                <div className="absolute bottom-2 left-2 rounded-lg bg-black/50 px-2 py-1 text-xs text-white backdrop-blur">
                  {state.currentPhase}
                </div>
              </div>

              {/* Direction Stats - More Compact */}
              <div className="mt-3 grid grid-cols-4 gap-2">
                <div className="rounded-lg bg-blue-500/10 p-2 text-center">
                  <div className="mb-0.5 text-xl font-bold text-blue-600">{state.northCars}</div>
                  <div className="text-[10px] text-muted-foreground">North</div>
                </div>
                <div className="rounded-lg bg-blue-500/10 p-2 text-center">
                  <div className="mb-0.5 text-xl font-bold text-blue-600">{state.southCars}</div>
                  <div className="text-[10px] text-muted-foreground">South</div>
                </div>
                <div className="rounded-lg bg-blue-500/10 p-2 text-center">
                  <div className="mb-0.5 text-xl font-bold text-blue-600">{state.eastCars}</div>
                  <div className="text-[10px] text-muted-foreground">East</div>
                </div>
                <div className="rounded-lg bg-blue-500/10 p-2 text-center">
                  <div className="mb-0.5 text-xl font-bold text-blue-600">{state.westCars}</div>
                  <div className="text-[10px] text-muted-foreground">West</div>
                </div>
              </div>
            </div>
          </div>

          {/* Metrics Panel - More Compact */}
          <div className="space-y-3">
            {/* Real-time Metrics */}
            <div className="rounded-xl border border-border bg-card p-4">
              <h3 className="mb-3 text-sm font-semibold">Real-time Metrics</h3>
              <div className="space-y-2">
                <div className="rounded-lg bg-green-500/10 p-3">
                  <div className="mb-0.5 flex items-center gap-1.5 text-xs text-green-600">
                    <TrendingUp className="h-3 w-3" />
                    <span>Reward</span>
                  </div>
                  <div className="text-xl font-bold">{state.reward.toFixed(1)}</div>
                  <div className="mt-0.5 text-[10px] text-muted-foreground">Higher is better</div>
                </div>

                <div className="rounded-lg bg-orange-500/10 p-3">
                  <div className="mb-0.5 flex items-center gap-1.5 text-xs text-orange-600">
                    <Clock className="h-3 w-3" />
                    <span>Avg Wait Time</span>
                  </div>
                  <div className="text-xl font-bold">{state.avgWaitTime.toFixed(1)}s</div>
                  <div className="mt-0.5 text-[10px] text-muted-foreground">Target: &lt;60s</div>
                </div>

                <div className="rounded-lg bg-blue-500/10 p-3">
                  <div className="mb-0.5 flex items-center gap-1.5 text-xs text-blue-600">
                    <Car className="h-3 w-3" />
                    <span>Throughput</span>
                  </div>
                  <div className="text-xl font-bold">{state.throughput}</div>
                  <div className="mt-0.5 text-[10px] text-muted-foreground">Vehicles/hour</div>
                </div>

                <div className="rounded-lg bg-purple-500/10 p-3">
                  <div className="mb-0.5 flex items-center gap-1.5 text-xs text-purple-600">
                    <AlertCircle className="h-3 w-3" />
                    <span>Total Queue</span>
                  </div>
                  <div className="text-xl font-bold">
                    {state.northCars + state.southCars + state.eastCars + state.westCars}
                  </div>
                  <div className="mt-0.5 text-[10px] text-muted-foreground">Waiting vehicles</div>
                </div>
              </div>
            </div>

            {/* Agent Info - More Compact */}
            <div className="rounded-xl border border-border bg-card p-4">
              <h3 className="mb-3 text-sm font-semibold">Agent Information</h3>
              <div className="space-y-2 text-xs">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Algorithm:</span>
                  <span className="font-semibold">DQN</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">State Dim:</span>
                  <span className="font-semibold">128</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Action Dim:</span>
                  <span className="font-semibold">8</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Exploration (ε):</span>
                  <span className="font-semibold">0.05</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Training:</span>
                  <span className="font-semibold text-green-600">Complete</span>
                </div>
              </div>
            </div>

            {/* Performance Comparison - More Compact */}
            <div className="rounded-xl border border-border bg-card p-4">
              <h3 className="mb-3 text-sm font-semibold">vs Baseline</h3>
              <div className="space-y-2">
                <div>
                  <div className="mb-1 flex justify-between text-xs">
                    <span className="text-muted-foreground">Fixed-Time</span>
                    <span className="font-semibold text-red-600">85s</span>
                  </div>
                  <div className="h-1.5 w-full rounded-full bg-red-500/20">
                    <div className="h-full w-[85%] rounded-full bg-red-500" />
                  </div>
                </div>
                <div>
                  <div className="mb-1 flex justify-between text-xs">
                    <span className="text-muted-foreground">DQN Agent</span>
                    <span className="font-semibold text-green-600">{state.avgWaitTime.toFixed(0)}s</span>
                  </div>
                  <div className="h-1.5 w-full rounded-full bg-green-500/20">
                    <div
                      className="h-full rounded-full bg-green-500"
                      style={{ width: `${(state.avgWaitTime / 85) * 100}%` }}
                    />
                  </div>
                </div>
                <div className="mt-3 rounded-lg bg-green-500/10 p-2.5 text-center">
                  <div className="text-xl font-bold text-green-600">
                    {((1 - state.avgWaitTime / 85) * 100).toFixed(0)}%
                  </div>
                  <div className="text-[10px] text-muted-foreground">Improvement</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Explanation - More Compact */}
        <div className="mt-4 rounded-xl border border-border bg-card p-4">
          <h3 className="mb-3 text-base font-semibold">How It Works</h3>
          <div className="grid gap-3 md:grid-cols-3">
            <div className="rounded-lg bg-blue-500/5 p-3">
              <div className="mb-1.5 text-sm font-semibold text-blue-600">1. Observe</div>
              <p className="text-xs text-muted-foreground">
                The DQN agent observes the current traffic state: number of vehicles, speeds, densities, and time of day.
              </p>
            </div>
            <div className="rounded-lg bg-purple-500/5 p-3">
              <div className="mb-1.5 text-sm font-semibold text-purple-600">2. Decide</div>
              <p className="text-xs text-muted-foreground">
                Using its trained neural network, the agent selects the optimal traffic light phase to minimize congestion.
              </p>
            </div>
            <div className="rounded-lg bg-green-500/5 p-3">
              <div className="mb-1.5 text-sm font-semibold text-green-600">3. Learn</div>
              <p className="text-xs text-muted-foreground">
                The agent receives rewards based on reduced wait times and improved traffic flow, continuously optimizing its policy.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}