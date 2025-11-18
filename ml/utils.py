"""
Utility functions for model training and evaluation
"""

import os
import json
import torch
from pathlib import Path


def save_checkpoint(filepath, model, optimizer, episode, global_step, reward, epsilon):
    """
    Save training checkpoint

    Args:
        filepath: Path to save checkpoint
        model: Model to save
        optimizer: Optimizer state
        episode: Current episode number
        global_step: Global training step
        reward: Episode reward
        epsilon: Current epsilon value
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "episode": episode,
        "global_step": global_step,
        "reward": reward,
        "epsilon": epsilon,
    }

    torch.save(checkpoint, filepath)


def load_checkpoint(filepath, model, optimizer=None, device="cpu"):
    """
    Load training checkpoint

    Args:
        filepath: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to restore state
        device: Device to map model to

    Returns:
        Dictionary with training state
    """
    checkpoint = torch.load(filepath, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return {
        "episode": checkpoint.get("episode", 0),
        "global_step": checkpoint.get("global_step", 0),
        "reward": checkpoint.get("reward", 0),
        "epsilon": checkpoint.get("epsilon", 0.01),
    }


def save_metrics(filepath, metrics):
    """Save training metrics to JSON"""
    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=2)


def load_metrics(filepath):
    """Load metrics from JSON"""
    with open(filepath, "r") as f:
        return json.load(f)


def create_output_dirs(base_dir):
    """Create output directory structure"""
    dirs = ["models", "logs", "metrics", "checkpoints"]

    for d in dirs:
        path = Path(base_dir) / d
        path.mkdir(parents=True, exist_ok=True)

    return base_dir
