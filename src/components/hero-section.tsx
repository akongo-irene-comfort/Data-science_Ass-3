"use client";

import { ArrowRight, Github, FileText } from "lucide-react";
import { Button } from "@/components/ui/button";

export const HeroSection = () => {
  return (
    <section id="overview" className="pt-32 pb-20 px-4">
      <div className="container mx-auto max-w-6xl">
        <div className="text-center space-y-6">
          <div className="inline-block px-4 py-2 bg-primary/10 rounded-full text-sm font-medium text-primary mb-4">
            Reinforcement Learning Project 2025
          </div>
          
          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold tracking-tight">
            Traffic Congestion Management
            <span className="block text-primary mt-2">Using Deep Reinforcement Learning</span>
          </h1>
          
          <p className="text-lg sm:text-xl text-muted-foreground max-w-3xl mx-auto leading-relaxed">
            An adaptive traffic light control system that learns optimal strategies to reduce congestion, 
            minimize waiting times, and maximize traffic flow using Proximal Policy Optimization (PPO).
          </p>

          <div className="flex flex-col sm:flex-row items-center justify-center gap-4 pt-8">
            <Button size="lg" className="gap-2 w-full sm:w-auto">
              <FileText className="h-5 w-5" />
              View Proposal
              <ArrowRight className="h-4 w-4" />
            </Button>
            <Button size="lg" variant="outline" className="gap-2 w-full sm:w-auto">
              <Github className="h-5 w-5" />
              GitHub Repository
            </Button>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-8 pt-16">
            <div className="space-y-2">
              <div className="text-4xl font-bold text-primary">40%</div>
              <div className="text-sm font-medium text-muted-foreground">Proposal & Theory</div>
            </div>
            <div className="space-y-2">
              <div className="text-4xl font-bold text-primary">60%</div>
              <div className="text-sm font-medium text-muted-foreground">Implementation & Deployment</div>
            </div>
            <div className="space-y-2">
              <div className="text-4xl font-bold text-primary">PPO</div>
              <div className="text-sm font-medium text-muted-foreground">Algorithm Choice</div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};
