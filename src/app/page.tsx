import Link from "next/link";
import { ArrowRight, Brain, Car, Network, Zap, Target, Code, Cloud } from "lucide-react";

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-muted/20">
      {/* Hero Section */}
      <section className="relative overflow-hidden px-6 py-20 sm:py-32 lg:px-8">
        <div className="mx-auto max-w-5xl text-center">
          <div className="mb-8 inline-flex items-center gap-2 rounded-full border border-primary/20 bg-primary/5 px-4 py-2">
            <Brain className="h-4 w-4 text-primary" />
            <span className="text-sm font-medium">Reinforcement Learning Project</span>
          </div>
          
          <h1 className="mb-6 text-4xl font-bold tracking-tight sm:text-6xl bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 bg-clip-text text-transparent">
            Traffic Congestion Optimization Using Deep RL
          </h1>
          
          <p className="mx-auto mb-10 max-w-2xl text-lg leading-8 text-muted-foreground">
            A comprehensive reinforcement learning solution for intelligent traffic management.
            Leveraging Deep Q-Networks to reduce congestion, minimize wait times, and optimize traffic flow in urban environments.
          </p>
          
          <div className="flex flex-wrap items-center justify-center gap-4">
            <Link
              href="/proposal"
              className="flex items-center gap-2 rounded-lg bg-primary px-6 py-3 font-semibold text-primary-foreground transition-colors hover:bg-primary/90"
            >
              View Proposal
              <ArrowRight className="h-4 w-4" />
            </Link>
            <Link
              href="/implementation"
              className="flex items-center gap-2 rounded-lg border border-border bg-background px-6 py-3 font-semibold transition-colors hover:bg-accent"
            >
              Implementation
            </Link>
            <Link
              href="/demo"
              className="flex items-center gap-2 rounded-lg border border-border bg-background px-6 py-3 font-semibold transition-colors hover:bg-accent"
            >
              Live Demo
            </Link>
          </div>
        </div>
      </section>

      {/* Problem Statement */}
      <section className="px-6 py-16 lg:px-8">
        <div className="mx-auto max-w-5xl">
          <div className="rounded-2xl border border-border bg-card p-8 shadow-sm">
            <div className="mb-6 flex items-center gap-3">
              <Car className="h-8 w-8 text-destructive" />
              <h2 className="text-3xl font-bold">The Problem</h2>
            </div>
            <p className="mb-4 text-lg leading-relaxed text-muted-foreground">
              Traffic congestion is a critical urban challenge affecting millions daily. Traditional traffic management
              systems lack the intelligence to dynamically adapt to real-time conditions, resulting in:
            </p>
            <ul className="grid gap-3 sm:grid-cols-2">
              {[
                "Increased commute times and fuel consumption",
                "Higher CO2 emissions and pollution",
                "Reduced productivity and economic impact",
                "Driver frustration and road safety concerns"
              ].map((item, i) => (
                <li key={i} className="flex items-start gap-2">
                  <div className="mt-1 h-1.5 w-1.5 rounded-full bg-primary" />
                  <span className="text-muted-foreground">{item}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </section>

      {/* RL Solution Overview */}
      <section className="px-6 py-16 lg:px-8">
        <div className="mx-auto max-w-5xl">
          <div className="mb-12 text-center">
            <h2 className="mb-4 text-3xl font-bold">Our Solution: Deep Reinforcement Learning</h2>
            <p className="text-lg text-muted-foreground">
              An intelligent agent that learns optimal traffic control policies through trial and error
            </p>
          </div>

          <div className="grid gap-6 md:grid-cols-2">
            <div className="rounded-xl border border-border bg-card p-6 transition-shadow hover:shadow-lg">
              <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-lg bg-blue-500/10">
                <Target className="h-6 w-6 text-blue-600" />
              </div>
              <h3 className="mb-2 text-xl font-semibold">MDP Formulation</h3>
              <p className="text-muted-foreground">
                <strong>State:</strong> Vehicle count, average speed, lane density, time of day<br />
                <strong>Action:</strong> Traffic light timing, lane control, speed limits<br />
                <strong>Reward:</strong> Reduced wait time, improved flow, congestion reduction
              </p>
            </div>

            <div className="rounded-xl border border-border bg-card p-6 transition-shadow hover:shadow-lg">
              <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-lg bg-purple-500/10">
                <Network className="h-6 w-6 text-purple-600" />
              </div>
              <h3 className="mb-2 text-xl font-semibold">Algorithm: DQN</h3>
              <p className="text-muted-foreground">
                Deep Q-Network with experience replay and target networks. Handles continuous state spaces
                with discrete actions, perfect for traffic light control and dynamic routing.
              </p>
            </div>

            <div className="rounded-xl border border-border bg-card p-6 transition-shadow hover:shadow-lg">
              <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-lg bg-green-500/10">
                <Zap className="h-6 w-6 text-green-600" />
              </div>
              <h3 className="mb-2 text-xl font-semibold">Simulation Environment</h3>
              <p className="text-muted-foreground">
                Custom SUMO-based traffic simulator modeling realistic urban road networks with
                vehicle dynamics, traffic patterns, and real-world constraints.
              </p>
            </div>

            <div className="rounded-xl border border-border bg-card p-6 transition-shadow hover:shadow-lg">
              <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-lg bg-orange-500/10">
                <Cloud className="h-6 w-6 text-orange-600" />
              </div>
              <h3 className="mb-2 text-xl font-semibold">MLOps & Deployment</h3>
              <p className="text-muted-foreground">
                Production-ready with FastAPI serving, Docker containerization, CI/CD pipelines,
                and cloud deployment on AWS SageMaker with real-time monitoring.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Project Structure */}
      <section className="px-6 py-16 lg:px-8">
        <div className="mx-auto max-w-5xl">
          <h2 className="mb-8 text-center text-3xl font-bold">Project Structure</h2>
          
          <div className="grid gap-6 md:grid-cols-2">
            {/* Part A */}
            <Link href="/proposal" className="group rounded-2xl border-2 border-blue-500/20 bg-gradient-to-br from-blue-500/5 to-purple-500/5 p-8 transition-all hover:border-blue-500/40 hover:shadow-xl">
              <div className="mb-4 flex items-center justify-between">
                <span className="text-sm font-semibold text-blue-600">40% Weight</span>
                <Brain className="h-8 w-8 text-blue-600 transition-transform group-hover:scale-110" />
              </div>
              <h3 className="mb-3 text-2xl font-bold">Part A: Research & Proposal</h3>
              <p className="mb-4 text-muted-foreground">
                Theoretical foundation and problem formulation
              </p>
              <ul className="space-y-2 text-sm">
                <li className="flex items-center gap-2">
                  <div className="h-1.5 w-1.5 rounded-full bg-blue-600" />
                  <span>MDP Definition & Problem Formulation</span>
                </li>
                <li className="flex items-center gap-2">
                  <div className="h-1.5 w-1.5 rounded-full bg-blue-600" />
                  <span>Algorithm Selection & Justification</span>
                </li>
                <li className="flex items-center gap-2">
                  <div className="h-1.5 w-1.5 rounded-full bg-blue-600" />
                  <span>Simulation Setup & Evaluation Plan</span>
                </li>
              </ul>
              <div className="mt-6 flex items-center gap-2 text-sm font-semibold text-blue-600">
                View Details
                <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-1" />
              </div>
            </Link>

            {/* Part B */}
            <Link href="/implementation" className="group rounded-2xl border-2 border-purple-500/20 bg-gradient-to-br from-purple-500/5 to-pink-500/5 p-8 transition-all hover:border-purple-500/40 hover:shadow-xl">
              <div className="mb-4 flex items-center justify-between">
                <span className="text-sm font-semibold text-purple-600">60% Weight</span>
                <Code className="h-8 w-8 text-purple-600 transition-transform group-hover:scale-110" />
              </div>
              <h3 className="mb-3 text-2xl font-bold">Part B: Implementation & Deployment</h3>
              <p className="mb-4 text-muted-foreground">
                Practical engineering and MLOps
              </p>
              <ul className="space-y-2 text-sm">
                <li className="flex items-center gap-2">
                  <div className="h-1.5 w-1.5 rounded-full bg-purple-600" />
                  <span>RL Agent Training & Model Saving</span>
                </li>
                <li className="flex items-center gap-2">
                  <div className="h-1.5 w-1.5 rounded-full bg-purple-600" />
                  <span>Model Serving API & Containerization</span>
                </li>
                <li className="flex items-center gap-2">
                  <div className="h-1.5 w-1.5 rounded-full bg-purple-600" />
                  <span>CI/CD Pipeline & Cloud Deployment</span>
                </li>
              </ul>
              <div className="mt-6 flex items-center gap-2 text-sm font-semibold text-purple-600">
                View Details
                <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-1" />
              </div>
            </Link>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="px-6 py-16 lg:px-8">
        <div className="mx-auto max-w-5xl">
          <div className="rounded-2xl border border-border bg-gradient-to-r from-blue-600/10 via-purple-600/10 to-pink-600/10 p-12 text-center">
            <h2 className="mb-4 text-3xl font-bold">Ready to Explore?</h2>
            <p className="mx-auto mb-8 max-w-2xl text-lg text-muted-foreground">
              Dive into the technical details, explore the implementation, or try the interactive simulation demo
            </p>
            <div className="flex flex-wrap justify-center gap-4">
              <Link
                href="/demo"
                className="flex items-center gap-2 rounded-lg bg-primary px-6 py-3 font-semibold text-primary-foreground transition-colors hover:bg-primary/90"
              >
                Try Interactive Demo
                <ArrowRight className="h-4 w-4" />
              </Link>
              <Link
                href="/docs"
                className="flex items-center gap-2 rounded-lg border border-border bg-background px-6 py-3 font-semibold transition-colors hover:bg-accent"
              >
                View Documentation
              </Link>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
