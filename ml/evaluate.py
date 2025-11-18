"""
Model Evaluation Script
Evaluate trained DQN agent on test episodes
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path

from environment import TrafficEnv
from train import DQNNetwork


def evaluate_model(
    model_path,
    episodes=10,
    sumo_cfg="sumo/grid.sumocfg",
    output_file="metrics/evaluation.json",
):
    """
    Evaluate trained model

    Args:
        model_path: Path to trained model
        episodes: Number of evaluation episodes
        sumo_cfg: SUMO configuration file
        output_file: Path to save evaluation metrics

    Returns:
        Dictionary of evaluation metrics
    """

    # Initialize environment
    env = TrafficEnv(sumo_cfg=sumo_cfg, gui=False)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model = DQNNetwork(state_dim, action_dim).to(device)

    if model_path.endswith(".pth"):
        model.load_state_dict(torch.load(model_path, map_location=device))
    elif model_path.endswith(".pt"):
        model = torch.jit.load(model_path, map_location=device)

    model.eval()

    print(f"Evaluating model for {episodes} episodes...")

    # Evaluation metrics
    episode_rewards = []
    episode_lengths = []
    avg_waiting_times = []
    avg_queue_lengths = []
    throughputs = []

    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        steps = 0

        while True:
            # Select action (greedy, no exploration)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = model(state_tensor)
                action = q_values.argmax().item()

            # Take step
            next_state, reward, done, truncated, info = env.step(action)

            episode_reward += reward
            steps += 1
            state = next_state

            if done or truncated:
                break

        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        avg_waiting_times.append(info["avg_waiting_time"])
        avg_queue_lengths.append(info["avg_queue_length"])
        throughputs.append(info["throughput"])

        print(
            f"Episode {episode + 1}/{episodes}: "
            f"Reward={episode_reward:.2f}, "
            f"Length={steps}, "
            f"Waiting Time={info['avg_waiting_time']:.2f}s"
        )

    env.close()

    # Compute statistics
    metrics = {
        "episodes": episodes,
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
        "mean_waiting_time": float(np.mean(avg_waiting_times)),
        "mean_queue_length": float(np.mean(avg_queue_lengths)),
        "mean_throughput": float(np.mean(throughputs)),
        "episode_rewards": [float(r) for r in episode_rewards],
        "model_path": model_path,
    }

    # Save metrics
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n=== Evaluation Results ===")
    print(f"Mean Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
    print(f"Mean Episode Length: {metrics['mean_length']:.1f}")
    print(f"Mean Waiting Time: {metrics['mean_waiting_time']:.2f}s")
    print(f"Mean Queue Length: {metrics['mean_queue_length']:.2f}")
    print(f"Mean Throughput: {metrics['mean_throughput']:.0f} vehicles")
    print(f"Metrics saved to: {output_path}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained DQN model")

    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to trained model"
    )
    parser.add_argument(
        "--episodes", type=int, default=10, help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--sumo-cfg",
        type=str,
        default="sumo/grid.sumocfg",
        help="SUMO configuration file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="metrics/evaluation.json",
        help="Output file for metrics",
    )

    args = parser.parse_args()

    evaluate_model(
        model_path=args.model_path,
        episodes=args.episodes,
        sumo_cfg=args.sumo_cfg,
        output_file=args.output,
    )
