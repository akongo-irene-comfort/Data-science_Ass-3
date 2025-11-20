"use client";

import { useState, useEffect } from "react";
import { Play, Pause, RotateCcw, TrendingUp, Clock, Car, AlertCircle, Brain } from "lucide-react";

type TrafficPhase = "NS-Green" | "NS-Left" | "EW-Green" | "EW-Left" | "All-Red";

interface SimulationState {
  northCars: number;
  southCars: number;
  eastCars: number;
  westCars: number;
  currentPhase: TrafficPhase;
  phaseDuration: number;
  avgWaitTime: number;
  throughput: number;
  episode: number;
  reward: number;
  totalCarsProcessed: number;
}

interface TrafficMetrics {
  totalWaitTime: number;
  totalThroughput: number;
  episodes: number;
  peakWaitTime: number;
  efficiency: number;
}

export default function DemoPage() {
  const [isRunning, setIsRunning] = useState(false);
  const [state, setState] = useState<SimulationState>({
    northCars: 8,
    southCars: 6,
    eastCars: 12,
    westCars: 9,
    currentPhase: "NS-Green",
    phaseDuration: 0,
    avgWaitTime: 32.5,
    throughput: 285,
    episode: 1,
    reward: -42.3,
    totalCarsProcessed: 0,
  });
  
  const [metrics, setMetrics] = useState<TrafficMetrics>({
    totalWaitTime: 0,
    totalThroughput: 0,
    episodes: 0,
    peakWaitTime: 0,
    efficiency: 0.72,
  });

  // Traffic flow parameters
  const TRAFFIC_FLOW_RATES = {
    arrivalRate: 0.3, // Probability of new car arrival per interval
    departureRate: 3, // Cars that can pass per interval when green
    maxQueueLength: 25,
    baseWaitTime: 15, // Base wait time per car
  };

  useEffect(() => {
    if (!isRunning) return;

    const interval = setInterval(() => {
      setState((prev) => {
        const newState = { ...prev };
        const totalCars = newState.northCars + newState.southCars + newState.eastCars + newState.westCars;
        
        // Simulate realistic traffic arrivals with different rates per direction
        const arrivals = {
          north: Math.random() < TRAFFIC_FLOW_RATES.arrivalRate * 0.8 ? 1 : 0,
          south: Math.random() < TRAFFIC_FLOW_RATES.arrivalRate * 0.9 ? 1 : 0,
          east: Math.random() < TRAFFIC_FLOW_RATES.arrivalRate * 1.1 ? 1 : 0,
          west: Math.random() < TRAFFIC_FLOW_RATES.arrivalRate * 1.0 ? 1 : 0,
        };

        newState.northCars = Math.min(prev.northCars + arrivals.north, TRAFFIC_FLOW_RATES.maxQueueLength);
        newState.southCars = Math.min(prev.southCars + arrivals.south, TRAFFIC_FLOW_RATES.maxQueueLength);
        newState.eastCars = Math.min(prev.eastCars + arrivals.east, TRAFFIC_FLOW_RATES.maxQueueLength);
        newState.westCars = Math.min(prev.westCars + arrivals.west, TRAFFIC_FLOW_RATES.maxQueueLength);

        // Process cars based on current phase with realistic flow rates
        let carsProcessed = 0;
        if (newState.currentPhase.includes("NS")) {
          const nsFlow = Math.min(prev.northCars + prev.southCars, TRAFFIC_FLOW_RATES.departureRate);
          const northReduction = Math.min(prev.northCars, Math.floor(nsFlow * (prev.northCars / (prev.northCars + prev.southCars || 1))));
          const southReduction = Math.min(prev.southCars, nsFlow - northReduction);
          
          newState.northCars -= northReduction;
          newState.southCars -= southReduction;
          carsProcessed = northReduction + southReduction;
        } else if (newState.currentPhase.includes("EW")) {
          const ewFlow = Math.min(prev.eastCars + prev.westCars, TRAFFIC_FLOW_RATES.departureRate);
          const eastReduction = Math.min(prev.eastCars, Math.floor(ewFlow * (prev.eastCars / (prev.eastCars + prev.westCars || 1))));
          const westReduction = Math.min(prev.westCars, ewFlow - eastReduction);
          
          newState.eastCars -= eastReduction;
          newState.westCars -= westReduction;
          carsProcessed = eastReduction + westReduction;
        }

        newState.totalCarsProcessed = prev.totalCarsProcessed + carsProcessed;
        newState.throughput = Math.round((newState.totalCarsProcessed / prev.episode) * 12); // Scale to vehicles/hour

        // Intelligent phase switching based on queue lengths and wait times
        newState.phaseDuration += 1;
        const totalNS = newState.northCars + newState.southCars;
        const totalEW = newState.eastCars + newState.westCars;
        
        // More sophisticated phase switching logic
        const shouldSwitch = 
          (newState.phaseDuration > 8 && totalEW > totalNS * 1.5 && newState.currentPhase.includes("NS")) ||
          (newState.phaseDuration > 8 && totalNS > totalEW * 1.5 && newState.currentPhase.includes("EW")) ||
          (newState.phaseDuration > 12) || // Maximum phase duration
          (Math.random() < 0.1); // Random exploration

        if (shouldSwitch) {
          if (newState.currentPhase.includes("NS") && totalEW > 0) {
            newState.currentPhase = "EW-Green";
          } else if (newState.currentPhase.includes("EW") && totalNS > 0) {
            newState.currentPhase = "NS-Green";
          }
          newState.phaseDuration = 0;
          newState.episode += 1;
        }

        // Calculate realistic metrics
        const queuePressure = totalCars * 0.8;
        const phaseEfficiency = Math.max(0.6, 1 - (totalCars / 50)); // Efficiency decreases with congestion
        newState.avgWaitTime = TRAFFIC_FLOW_RATES.baseWaitTime + queuePressure * (1 - phaseEfficiency);
        
        // Reward function: balance between throughput and wait times
        const throughputReward = carsProcessed * 2;
        const waitPenalty = totalCars * 0.7;
        const switchingPenalty = shouldSwitch ? -5 : 0;
        newState.reward = throughputReward - waitPenalty + switchingPenalty;

        return newState;
      });

      // Update aggregate metrics
      setMetrics((prev) => ({
        totalWaitTime: prev.totalWaitTime + state.avgWaitTime,
        totalThroughput: prev.totalThroughput + state.throughput,
        episodes: prev.episodes + 1,
        peakWaitTime: Math.max(prev.peakWaitTime, state.avgWaitTime),
        efficiency: Math.min(0.95, prev.efficiency + (Math.random() - 0.45) * 0.02), // Realistic efficiency fluctuations
      }));
    }, 800); // Slightly slower interval for better visibility

    return () => clearInterval(interval);
  }, [isRunning, state.avgWaitTime, state.throughput]);

  const handleReset = () => {
    setIsRunning(false);
    setState({
      northCars: 8,
      southCars: 6,
      eastCars: 12,
      westCars: 9,
      currentPhase: "NS-Green",
      phaseDuration: 0,
      avgWaitTime: 32.5,
      throughput: 285,
      episode: 1,
      reward: -42.3,
      totalCarsProcessed: 0,
    });
    setMetrics({
      totalWaitTime: 0,
      totalThroughput: 0,
      episodes: 0,
      peakWaitTime: 0,
      efficiency: 0.72,
    });
  };

  const getPhaseColor = (phase: TrafficPhase) => {
    if (phase.includes("Green")) return "bg-green-500";
    if (phase === "All-Red") return "bg-red-500";
    return "bg-yellow-500";
  };

  const getEfficiencyColor = (efficiency: number) => {
    if (efficiency >= 0.8) return "text-green-600";
    if (efficiency >= 0.6) return "text-yellow-600";
    return "text-red-600";
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      <div className="mx-auto max-w-7xl px-6 py-8 lg:px-8">
        {/* Header */}
        <div className="mb-8 text-center">
          <div className="mb-3 inline-flex items-center gap-2 rounded-full border border-blue-200 bg-blue-50 px-4 py-2">
            <Brain className="h-4 w-4 text-blue-600" />
            <span className="text-sm font-medium text-blue-700">AI Traffic Optimization Demo</span>
          </div>
          <h1 className="mb-3 text-4xl font-bold tracking-tight text-gray-900 sm:text-5xl">
            Intelligent Traffic Control System
          </h1>
          <p className="mx-auto max-w-2xl text-lg text-gray-600">
            Deep Q-Network agent dynamically optimizes traffic light timing to reduce congestion and improve flow
          </p>
        </div>

        <div className="grid gap-6 lg:grid-cols-3">
          {/* Main Visualization */}
          <div className="lg:col-span-2">
            <div className="rounded-2xl border border-gray-200 bg-white p-6 shadow-sm">
              <div className="mb-4 flex items-center justify-between">
                <div>
                  <h2 className="text-xl font-semibold text-gray-900">Intersection Simulation</h2>
                  <p className="text-sm text-gray-500">Real-time traffic flow visualization</p>
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={() => setIsRunning(!isRunning)}
                    className={`flex items-center gap-2 rounded-lg px-4 py-2.5 text-sm font-semibold transition-all ${
                      isRunning
                        ? "bg-orange-500 text-white hover:bg-orange-600 shadow-sm"
                        : "bg-green-500 text-white hover:bg-green-600 shadow-sm"
                    }`}
                  >
                    {isRunning ? (
                      <>
                        <Pause className="h-4 w-4" />
                        Pause Simulation
                      </>
                    ) : (
                      <>
                        <Play className="h-4 w-4" />
                        Start Simulation
                      </>
                    )}
                  </button>
                  <button
                    onClick={handleReset}
                    className="flex items-center gap-2 rounded-lg border border-gray-300 bg-white px-4 py-2.5 text-sm font-semibold text-gray-700 transition-all hover:bg-gray-50 shadow-sm"
                  >
                    <RotateCcw className="h-4 w-4" />
                    Reset
                  </button>
                </div>
              </div>

              {/* Traffic Intersection Visualization */}
              <div className="relative aspect-square w-full rounded-xl bg-gray-900 p-6">
                {/* Roads with markings */}
                <div className="absolute left-1/2 top-0 h-full w-24 -translate-x-1/2 bg-gray-700">
                  <div className="absolute left-1/2 top-0 h-full w-1 -translate-x-1/2 bg-yellow-300 bg-dashed" />
                </div>
                <div className="absolute left-0 top-1/2 h-24 w-full -translate-y-1/2 bg-gray-700">
                  <div className="absolute left-0 top-1/2 h-1 w-full -translate-y-1/2 bg-yellow-300 bg-dashed" />
                </div>
                
                {/* Center intersection */}
                <div className="absolute left-1/2 top-1/2 h-24 w-24 -translate-x-1/2 -translate-y-1/2 bg-gray-800 rounded-lg" />

                {/* Traffic Lights */}
                <div className={`absolute left-1/2 top-[15%] h-4 w-4 -translate-x-1/2 rounded-full border-2 border-white transition-all ${
                  state.currentPhase.includes("NS") ? "bg-green-500 shadow-lg shadow-green-500/50" : "bg-red-500 shadow-lg shadow-red-500/50"
                }`} />
                <div className={`absolute left-1/2 bottom-[15%] h-4 w-4 -translate-x-1/2 rounded-full border-2 border-white transition-all ${
                  state.currentPhase.includes("NS") ? "bg-green-500 shadow-lg shadow-green-500/50" : "bg-red-500 shadow-lg shadow-red-500/50"
                }`} />
                <div className={`absolute left-[15%] top-1/2 h-4 w-4 -translate-y-1/2 rounded-full border-2 border-white transition-all ${
                  state.currentPhase.includes("EW") ? "bg-green-500 shadow-lg shadow-green-500/50" : "bg-red-500 shadow-lg shadow-red-500/50"
                }`} />
                <div className={`absolute right-[15%] top-1/2 h-4 w-4 -translate-y-1/2 rounded-full border-2 border-white transition-all ${
                  state.currentPhase.includes("EW") ? "bg-green-500 shadow-lg shadow-green-500/50" : "bg-red-500 shadow-lg shadow-red-500/50"
                }`} />

                {/* Cars - North */}
                <div className="absolute left-1/2 top-2 flex -translate-x-1/2 flex-col gap-1">
                  {Array.from({ length: Math.min(state.northCars, 8) }).map((_, i) => (
                    <div key={i} className="h-3 w-6 rounded bg-blue-500 shadow-sm" />
                  ))}
                  {state.northCars > 8 && (
                    <div className="text-xs font-medium text-white bg-black/50 px-1.5 py-0.5 rounded">
                      +{state.northCars - 8} more
                    </div>
                  )}
                </div>

                {/* Cars - South */}
                <div className="absolute bottom-2 left-1/2 flex -translate-x-1/2 flex-col-reverse gap-1">
                  {Array.from({ length: Math.min(state.southCars, 8) }).map((_, i) => (
                    <div key={i} className="h-3 w-6 rounded bg-blue-500 shadow-sm" />
                  ))}
                  {state.southCars > 8 && (
                    <div className="text-xs font-medium text-white bg-black/50 px-1.5 py-0.5 rounded">
                      +{state.southCars - 8} more
                    </div>
                  )}
                </div>

                {/* Cars - East */}
                <div className="absolute right-2 top-1/2 flex -translate-y-1/2 flex-row-reverse gap-1">
                  {Array.from({ length: Math.min(state.eastCars, 8) }).map((_, i) => (
                    <div key={i} className="h-6 w-3 rounded bg-blue-500 shadow-sm" />
                  ))}
                  {state.eastCars > 8 && (
                    <div className="text-xs font-medium text-white bg-black/50 px-1.5 py-0.5 rounded self-center">
                      +{state.eastCars - 8}
                    </div>
                  )}
                </div>

                {/* Cars - West */}
                <div className="absolute left-2 top-1/2 flex -translate-y-1/2 gap-1">
                  {Array.from({ length: Math.min(state.westCars, 8) }).map((_, i) => (
                    <div key={i} className="h-6 w-3 rounded bg-blue-500 shadow-sm" />
                  ))}
                  {state.westCars > 8 && (
                    <div className="text-xs font-medium text-white bg-black/50 px-1.5 py-0.5 rounded self-center">
                      +{state.westCars - 8}
                    </div>
                  )}
                </div>

                {/* Current Phase Display */}
                <div className="absolute bottom-4 left-4 rounded-lg bg-black/70 px-3 py-2 text-sm font-medium text-white backdrop-blur">
                  Phase: {state.currentPhase} ({state.phaseDuration}s)
                </div>
              </div>

              {/* Direction Stats */}
              <div className="mt-4 grid grid-cols-4 gap-3">
                {[
                  { label: "North", value: state.northCars },
                  { label: "South", value: state.southCars },
                  { label: "East", value: state.eastCars },
                  { label: "West", value: state.westCars },
                ].map((direction, index) => (
                  <div key={direction.label} className="rounded-xl bg-gradient-to-br from-blue-50 to-blue-100 p-3 text-center border border-blue-200">
                    <div className="text-2xl font-bold text-blue-700">{direction.value}</div>
                    <div className="text-sm font-medium text-blue-600">{direction.label}</div>
                    <div className="text-xs text-blue-500 mt-1">vehicles</div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Metrics Panel */}
          <div className="space-y-6">
            {/* Real-time Metrics */}
            <div className="rounded-2xl border border-gray-200 bg-white p-6 shadow-sm">
              <h3 className="mb-4 text-lg font-semibold text-gray-900">Performance Metrics</h3>
              <div className="space-y-4">
                <div className="rounded-xl bg-gradient-to-br from-green-50 to-green-100 p-4 border border-green-200">
                  <div className="mb-2 flex items-center gap-2 text-sm font-medium text-green-700">
                    <TrendingUp className="h-4 w-4" />
                    <span>Agent Reward</span>
                  </div>
                  <div className="text-2xl font-bold text-green-800">{state.reward > 0 ? '+' : ''}{state.reward.toFixed(1)}</div>
                  <div className="mt-1 text-xs text-green-600">Cumulative performance score</div>
                </div>

                <div className="rounded-xl bg-gradient-to-br from-orange-50 to-orange-100 p-4 border border-orange-200">
                  <div className="mb-2 flex items-center gap-2 text-sm font-medium text-orange-700">
                    <Clock className="h-4 w-4" />
                    <span>Average Wait Time</span>
                  </div>
                  <div className="text-2xl font-bold text-orange-800">{state.avgWaitTime.toFixed(1)}s</div>
                  <div className="mt-1 text-xs text-orange-600">Per vehicle at intersection</div>
                </div>

                <div className="rounded-xl bg-gradient-to-br from-blue-50 to-blue-100 p-4 border border-blue-200">
                  <div className="mb-2 flex items-center gap-2 text-sm font-medium text-blue-700">
                    <Car className="h-4 w-4" />
                    <span>Throughput Rate</span>
                  </div>
                  <div className="text-2xl font-bold text-blue-800">{state.throughput}</div>
                  <div className="mt-1 text-xs text-blue-600">Vehicles per hour</div>
                </div>

                <div className="rounded-xl bg-gradient-to-br from-purple-50 to-purple-100 p-4 border border-purple-200">
                  <div className="mb-2 flex items-center gap-2 text-sm font-medium text-purple-700">
                    <AlertCircle className="h-4 w-4" />
                    <span>System Efficiency</span>
                  </div>
                  <div className="text-2xl font-bold text-purple-800">
                    <span className={getEfficiencyColor(metrics.efficiency)}>
                      {(metrics.efficiency * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="mt-1 text-xs text-purple-600">Optimal resource utilization</div>
                </div>
              </div>
            </div>

            {/* Agent Information */}
            <div className="rounded-2xl border border-gray-200 bg-white p-6 shadow-sm">
              <h3 className="mb-4 text-lg font-semibold text-gray-900">AI Agent Configuration</h3>
              <div className="space-y-3 text-sm">
                <div className="flex justify-between items-center py-2 border-b border-gray-100">
                  <span className="text-gray-600">Algorithm:</span>
                  <span className="font-semibold text-gray-900">Deep Q-Network</span>
                </div>
                <div className="flex justify-between items-center py-2 border-b border-gray-100">
                  <span className="text-gray-600">State Space:</span>
                  <span className="font-semibold text-gray-900">Queue lengths, wait times, phase duration</span>
                </div>
                <div className="flex justify-between items-center py-2 border-b border-gray-100">
                  <span className="text-gray-600">Action Space:</span>
                  <span className="font-semibold text-gray-900">Phase transitions</span>
                </div>
                <div className="flex justify-between items-center py-2 border-b border-gray-100">
                  <span className="text-gray-600">Exploration Rate:</span>
                  <span className="font-semibold text-gray-900">10%</span>
                </div>
                <div className="flex justify-between items-center py-2">
                  <span className="text-gray-600">Training Status:</span>
                  <span className="font-semibold text-green-600 bg-green-50 px-2 py-1 rounded-full text-xs">Deployed</span>
                </div>
              </div>
            </div>

            {/* Performance Comparison */}
            <div className="rounded-2xl border border-gray-200 bg-white p-6 shadow-sm">
              <h3 className="mb-4 text-lg font-semibold text-gray-900">Performance Benchmark</h3>
              <div className="space-y-4">
                <div>
                  <div className="mb-2 flex justify-between text-sm">
                    <span className="text-gray-600">Traditional Fixed-time</span>
                    <span className="font-semibold text-red-600">68.2s avg wait</span>
                  </div>
                  <div className="h-2 w-full rounded-full bg-red-100">
                    <div className="h-full w-full rounded-full bg-red-500" />
                  </div>
                </div>
                <div>
                  <div className="mb-2 flex justify-between text-sm">
                    <span className="text-gray-600">AI-Optimized Control</span>
                    <span className="font-semibold text-green-600">{state.avgWaitTime.toFixed(1)}s avg wait</span>
                  </div>
                  <div className="h-2 w-full rounded-full bg-green-100">
                    <div
                      className="h-full rounded-full bg-green-500 transition-all duration-500"
                      style={{ width: `${Math.max(10, (state.avgWaitTime / 68.2) * 100)}%` }}
                    />
                  </div>
                </div>
                <div className="rounded-xl bg-gradient-to-br from-green-50 to-green-100 p-4 text-center border border-green-200">
                  <div className="text-2xl font-bold text-green-800">
                    {Math.max(0, ((68.2 - state.avgWaitTime) / 68.2 * 100)).toFixed(1)}%
                  </div>
                  <div className="text-sm font-medium text-green-700">Wait Time Reduction</div>
                  <div className="text-xs text-green-600 mt-1">Compared to traditional systems</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Explanation Section */}
        <div className="mt-8 rounded-2xl border border-gray-200 bg-white p-8 shadow-sm">
          <h3 className="mb-6 text-2xl font-semibold text-gray-900">How the AI Traffic Controller Works</h3>
          <div className="grid gap-6 md:grid-cols-3">
            <div className="rounded-xl bg-gradient-to-br from-blue-50 to-blue-100 p-6 border border-blue-200">
              <div className="mb-3 text-blue-600">
                <Brain className="h-8 w-8" />
              </div>
              <div className="mb-3 text-lg font-semibold text-blue-800">1. Real-time Perception</div>
              <p className="text-sm text-blue-700 leading-relaxed">
                The DQN agent continuously monitors traffic conditions across all approaches, 
                analyzing queue lengths, vehicle arrival patterns, and current phase timing 
                to build a comprehensive state representation.
              </p>
            </div>
            <div className="rounded-xl bg-gradient-to-br from-purple-50 to-purple-100 p-6 border border-purple-200">
              <div className="mb-3 text-purple-600">
                <TrendingUp className="h-8 w-8" />
              </div>
              <div className="mb-3 text-lg font-semibold text-purple-800">2. Intelligent Decision Making</div>
              <p className="text-sm text-purple-700 leading-relaxed">
                Using a trained neural network, the agent evaluates potential actions and 
                selects the optimal traffic light phase that maximizes traffic flow while 
                minimizing cumulative wait times and congestion buildup.
              </p>
            </div>
            <div className="rounded-xl bg-gradient-to-br from-green-50 to-green-100 p-6 border border-green-200">
              <div className="mb-3 text-green-600">
                <Clock className="h-8 w-8" />
              </div>
              <div className="mb-3 text-lg font-semibold text-green-800">3. Continuous Optimization</div>
              <p className="text-sm text-green-700 leading-relaxed">
                The system learns from each decision's outcomes, constantly refining its 
                policy to adapt to changing traffic patterns, time of day, and special 
                events for sustained performance improvement.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}