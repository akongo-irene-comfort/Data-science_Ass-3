"use client";

import { useState, useEffect } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Play, Pause, RotateCcw, Zap } from "lucide-react";

interface Intersection {
  id: string;
  x: number;
  y: number;
  phase: number;
  queueLengths: number[];
}

export const DemoSection = () => {
  const [isRunning, setIsRunning] = useState(false);
  const [episode, setEpisode] = useState(0);
  const [totalReward, setTotalReward] = useState(0);
  const [avgWaitTime, setAvgWaitTime] = useState(45.2);
  const [throughput, setThroughput] = useState(320);
  const [intersections, setIntersections] = useState<Intersection[]>([
    { id: "I1", x: 1, y: 1, phase: 0, queueLengths: [12, 8, 5, 10] },
    { id: "I2", x: 2, y: 1, phase: 1, queueLengths: [7, 15, 9, 6] },
    { id: "I3", x: 1, y: 2, phase: 2, queueLengths: [10, 11, 8, 7] },
    { id: "I4", x: 2, y: 2, phase: 3, queueLengths: [6, 9, 12, 8] },
  ]);

  useEffect(() => {
    if (!isRunning) return;

    const interval = setInterval(() => {
      setIntersections((prev) =>
        prev.map((int) => ({
          ...int,
          phase: (int.phase + 1) % 4,
          queueLengths: int.queueLengths.map(() =>
            Math.max(0, Math.floor(Math.random() * 15))
          ),
        }))
      );

      setTotalReward((prev) => prev + Math.random() * 5 - 1);
      setAvgWaitTime((prev) => Math.max(10, prev - Math.random() * 2));
      setThroughput((prev) => Math.min(500, prev + Math.random() * 10));
    }, 1000);

    return () => clearInterval(interval);
  }, [isRunning]);

  const handleReset = () => {
    setIsRunning(false);
    setEpisode((prev) => prev + 1);
    setTotalReward(0);
    setAvgWaitTime(45.2);
    setThroughput(320);
    setIntersections([
      { id: "I1", x: 1, y: 1, phase: 0, queueLengths: [12, 8, 5, 10] },
      { id: "I2", x: 2, y: 1, phase: 1, queueLengths: [7, 15, 9, 6] },
      { id: "I3", x: 1, y: 2, phase: 2, queueLengths: [10, 11, 8, 7] },
      { id: "I4", x: 2, y: 2, phase: 3, queueLengths: [6, 9, 12, 8] },
    ]);
  };

  const getPhaseColor = (phase: number) => {
    const colors = ["#ef4444", "#f59e0b", "#10b981", "#3b82f6"];
    return colors[phase % 4];
  };

  return (
    <section id="demo" className="py-20 px-4">
      <div className="container mx-auto max-w-6xl">
        <div className="text-center space-y-4 mb-16">
          <h2 className="text-3xl sm:text-4xl font-bold">Interactive Demo</h2>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            Live simulation of PPO agent controlling traffic lights in a 2x2 grid network
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <Card className="p-6 lg:col-span-2 space-y-6">
            <div className="flex items-center justify-between">
              <h3 className="text-xl font-semibold">Traffic Network</h3>
              <div className="flex items-center gap-2">
                <Button
                  size="sm"
                  variant={isRunning ? "destructive" : "default"}
                  onClick={() => setIsRunning(!isRunning)}
                  className="gap-2"
                >
                  {isRunning ? (
                    <>
                      <Pause className="h-4 w-4" />
                      Pause
                    </>
                  ) : (
                    <>
                      <Play className="h-4 w-4" />
                      Start
                    </>
                  )}
                </Button>
                <Button size="sm" variant="outline" onClick={handleReset} className="gap-2">
                  <RotateCcw className="h-4 w-4" />
                  Reset
                </Button>
              </div>
            </div>

            <div className="aspect-square bg-muted/30 rounded-lg p-8 relative">
              {/* Grid lines */}
              <div className="absolute inset-8 grid grid-cols-2 grid-rows-2 gap-8">
                {intersections.map((int) => (
                  <div
                    key={int.id}
                    className="relative flex items-center justify-center"
                  >
                    {/* Intersection */}
                    <div
                      className="w-16 h-16 rounded-full shadow-lg transition-all duration-300 flex items-center justify-center"
                      style={{
                        backgroundColor: getPhaseColor(int.phase),
                      }}
                    >
                      <span className="text-white font-bold text-sm">{int.id}</span>
                    </div>

                    {/* Queue indicators */}
                    <div className="absolute -top-8 left-1/2 -translate-x-1/2 text-xs font-mono bg-background px-2 py-1 rounded shadow">
                      {int.queueLengths[0]}
                    </div>
                    <div className="absolute -bottom-8 left-1/2 -translate-x-1/2 text-xs font-mono bg-background px-2 py-1 rounded shadow">
                      {int.queueLengths[2]}
                    </div>
                    <div className="absolute -left-8 top-1/2 -translate-y-1/2 text-xs font-mono bg-background px-2 py-1 rounded shadow">
                      {int.queueLengths[3]}
                    </div>
                    <div className="absolute -right-8 top-1/2 -translate-y-1/2 text-xs font-mono bg-background px-2 py-1 rounded shadow">
                      {int.queueLengths[1]}
                    </div>

                    {/* Road connections */}
                    {int.x === 1 && (
                      <div className="absolute -right-8 top-1/2 w-16 h-1 bg-muted -translate-y-1/2" />
                    )}
                    {int.y === 1 && (
                      <div className="absolute -bottom-8 left-1/2 w-1 h-16 bg-muted -translate-x-1/2" />
                    )}
                  </div>
                ))}
              </div>
            </div>

            <div className="flex items-center gap-4 text-sm text-muted-foreground">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-[#ef4444]" />
                <span>Phase 0</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-[#f59e0b]" />
                <span>Phase 1</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-[#10b981]" />
                <span>Phase 2</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-[#3b82f6]" />
                <span>Phase 3</span>
              </div>
            </div>
          </Card>

          <div className="space-y-6">
            <Card className="p-6 space-y-4">
              <div className="flex items-center gap-3">
                <div className="p-3 bg-primary/10 rounded-lg">
                  <Zap className="h-5 w-5 text-primary" />
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Episode</p>
                  <p className="text-2xl font-bold">{episode}</p>
                </div>
              </div>
            </Card>

            <Card className="p-6 space-y-4">
              <h4 className="font-semibold">Real-time Metrics</h4>
              <div className="space-y-3">
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-muted-foreground">Total Reward</span>
                    <span className="font-mono">{totalReward.toFixed(1)}</span>
                  </div>
                  <div className="h-2 bg-muted rounded-full overflow-hidden">
                    <div
                      className="h-full bg-primary transition-all duration-300"
                      style={{
                        width: `${Math.min(100, (totalReward / 100) * 100)}%`,
                      }}
                    />
                  </div>
                </div>

                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-muted-foreground">Avg Wait Time</span>
                    <span className="font-mono">{avgWaitTime.toFixed(1)}s</span>
                  </div>
                  <div className="h-2 bg-muted rounded-full overflow-hidden">
                    <div
                      className="h-full bg-chart-2 transition-all duration-300"
                      style={{
                        width: `${Math.max(0, 100 - (avgWaitTime / 60) * 100)}%`,
                      }}
                    />
                  </div>
                </div>

                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-muted-foreground">Throughput</span>
                    <span className="font-mono">{throughput} veh/h</span>
                  </div>
                  <div className="h-2 bg-muted rounded-full overflow-hidden">
                    <div
                      className="h-full bg-chart-4 transition-all duration-300"
                      style={{
                        width: `${(throughput / 500) * 100}%`,
                      }}
                    />
                  </div>
                </div>
              </div>
            </Card>

            <Card className="p-6 space-y-2">
              <h4 className="font-semibold text-sm">Agent Status</h4>
              <div className="flex items-center gap-2">
                <div
                  className={`w-2 h-2 rounded-full ${
                    isRunning ? "bg-green-500 animate-pulse" : "bg-muted"
                  }`}
                />
                <span className="text-sm text-muted-foreground">
                  {isRunning ? "Active - Learning" : "Paused"}
                </span>
              </div>
              <p className="text-xs text-muted-foreground">
                Algorithm: PPO | Network: 2x2 Grid
              </p>
            </Card>
          </div>
        </div>

        <Card className="mt-8 p-6 bg-muted/30">
          <p className="text-sm text-muted-foreground text-center">
            <strong>Note:</strong> This is a simplified visualization. The actual implementation uses SUMO 
            for realistic vehicle dynamics, traffic patterns, and multi-agent coordination across larger networks.
          </p>
        </Card>
      </div>
    </section>
  );
};
