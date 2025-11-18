"""
DQN Training Script for Traffic Signal Control
Implements Deep Q-Network with experience replay and target network
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import wandb

from environment import TrafficEnv
from dqn_agent import DQNAgent, ReplayBuffer
from utils import save_checkpoint, load_checkpoint


class DQNNetwork(nn.Module):
    """Deep Q-Network architecture"""

    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256, 128]):
        super(DQNNetwork, self).__init__()

        layers = []
        input_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Dropout(0.2),
                ]
            )
            input_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(input_dim, action_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, state):
        return self.network(state)


def train_dqn(args):
    """Main training loop for DQN agent"""

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Initialize environment
    env = TrafficEnv(
        sumo_cfg=args.sumo_cfg, gui=False, max_steps=args.max_steps_per_episode
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print(f"Environment initialized: state_dim={state_dim}, action_dim={action_dim}")

    # Initialize networks
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = DQNNetwork(state_dim, action_dim).to(device)
    target_net = DQNNetwork(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # Initialize optimizer and replay buffer
    optimizer = optim.Adam(policy_net.parameters(), lr=args.learning_rate)
    replay_buffer = ReplayBuffer(capacity=args.buffer_size)

    # Initialize logging
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.use_wandb:
        wandb.init(
            project="rl-traffic-control",
            config=vars(args),
            name=f"dqn_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )

    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))

    # Training metrics
    episode_rewards = []
    episode_lengths = []
    losses = []

    # Epsilon decay schedule
    epsilon = args.epsilon_start
    epsilon_decay = (args.epsilon_start - args.epsilon_end) / args.epsilon_decay_steps

    global_step = 0
    best_reward = -float("inf")

    print(f"Starting training for {args.episodes} episodes...")
    print(f"Device: {device}")

    for episode in range(args.episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_loss = []

        for step in range(args.max_steps_per_episode):
            # Select action using epsilon-greedy
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    q_values = policy_net(state_tensor)
                    action = q_values.argmax().item()

            # Take action in environment
            next_state, reward, done, truncated, info = env.step(action)

            # Store transition in replay buffer
            replay_buffer.push(state, action, reward, next_state, done)

            episode_reward += reward
            global_step += 1

            # Update epsilon
            epsilon = max(args.epsilon_end, epsilon - epsilon_decay)

            # Train the network
            if (
                len(replay_buffer) >= args.batch_size
                and global_step >= args.warmup_steps
            ):
                if global_step % args.train_frequency == 0:
                    loss = train_step(
                        policy_net,
                        target_net,
                        replay_buffer,
                        optimizer,
                        args.batch_size,
                        args.gamma,
                        device,
                    )
                    episode_loss.append(loss)
                    losses.append(loss)

                    # Log to tensorboard
                    writer.add_scalar("Training/Loss", loss, global_step)
                    writer.add_scalar("Training/Epsilon", epsilon, global_step)

            # Update target network
            if global_step % args.target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())
                print(f"[Step {global_step}] Target network updated")

            state = next_state

            if done or truncated:
                break

        # Episode summary
        episode_rewards.append(episode_reward)
        episode_lengths.append(step + 1)
        avg_loss = np.mean(episode_loss) if episode_loss else 0

        # Logging
        if episode % args.log_interval == 0:
            avg_reward_100 = np.mean(episode_rewards[-100:])
            avg_length_100 = np.mean(episode_lengths[-100:])

            print(f"Episode {episode}/{args.episodes}")
            print(f"  Reward: {episode_reward:.2f} | Avg(100): {avg_reward_100:.2f}")
            print(f"  Length: {step + 1} | Avg(100): {avg_length_100:.1f}")
            print(f"  Loss: {avg_loss:.4f} | Epsilon: {epsilon:.4f}")
            print(f"  Buffer: {len(replay_buffer)} | Global Step: {global_step}")
            print(f"  Metrics: {info}")

            # Tensorboard logging
            writer.add_scalar("Episode/Reward", episode_reward, episode)
            writer.add_scalar("Episode/Length", step + 1, episode)
            writer.add_scalar("Episode/AvgReward100", avg_reward_100, episode)
            writer.add_scalar(
                "Episode/WaitingTime", info.get("avg_waiting_time", 0), episode
            )
            writer.add_scalar(
                "Episode/QueueLength", info.get("avg_queue_length", 0), episode
            )
            writer.add_scalar("Episode/Throughput", info.get("throughput", 0), episode)

            # Weights & Biases logging
            if args.use_wandb:
                wandb.log(
                    {
                        "episode": episode,
                        "reward": episode_reward,
                        "avg_reward_100": avg_reward_100,
                        "episode_length": step + 1,
                        "loss": avg_loss,
                        "epsilon": epsilon,
                        "global_step": global_step,
                        "waiting_time": info.get("avg_waiting_time", 0),
                        "queue_length": info.get("avg_queue_length", 0),
                        "throughput": info.get("throughput", 0),
                    }
                )

        # Save checkpoint
        if episode % args.save_interval == 0 or episode_reward > best_reward:
            if episode_reward > best_reward:
                best_reward = episode_reward
                checkpoint_path = output_dir / "dqn_best.pth"
            else:
                checkpoint_path = output_dir / f"dqn_episode_{episode}.pth"

            save_checkpoint(
                checkpoint_path,
                policy_net,
                optimizer,
                episode,
                global_step,
                episode_reward,
                epsilon,
            )
            print(f"  Checkpoint saved: {checkpoint_path}")

    # Save final model
    print("\nTraining completed!")
    print(f"Best reward: {best_reward:.2f}")

    # Save PyTorch model
    torch.save(policy_net.state_dict(), output_dir / "dqn_final.pth")

    # Save TorchScript model for production
    example_input = torch.randn(1, state_dim).to(device)
    traced_model = torch.jit.trace(policy_net, example_input)
    traced_model.save(str(output_dir / "dqn_final_traced.pt"))

    # Save training metrics
    metrics = {
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "best_reward": best_reward,
        "final_epsilon": epsilon,
        "total_steps": global_step,
        "config": vars(args),
    }

    with open(output_dir / "training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    env.close()
    writer.close()

    if args.use_wandb:
        wandb.finish()

    return policy_net, metrics


def train_step(
    policy_net, target_net, replay_buffer, optimizer, batch_size, gamma, device
):
    """Single training step"""

    # Sample batch
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

    # Convert to tensors
    states = torch.FloatTensor(states).to(device)
    actions = torch.LongTensor(actions).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    next_states = torch.FloatTensor(next_states).to(device)
    dones = torch.FloatTensor(dones).to(device)

    # Compute current Q values
    current_q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # Compute target Q values
    with torch.no_grad():
        next_q_values = target_net(next_states).max(1)[0]
        target_q_values = rewards + gamma * next_q_values * (1 - dones)

    # Compute loss
    loss = nn.MSELoss()(current_q_values, target_q_values)

    # Optimize
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
    optimizer.step()

    return loss.item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN for traffic signal control")

    # Environment
    parser.add_argument("--sumo-cfg", type=str, default="sumo/grid.sumocfg")
    parser.add_argument("--max-steps-per-episode", type=int, default=3600)

    # Training
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--buffer-size", type=int, default=100000)
    parser.add_argument("--warmup-steps", type=int, default=10000)
    parser.add_argument("--train-frequency", type=int, default=4)
    parser.add_argument("--target-update", type=int, default=1000)

    # Exploration
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.01)
    parser.add_argument("--epsilon-decay-steps", type=int, default=100000)

    # Logging
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--output-dir", type=str, default="models")

    # Misc
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    train_dqn(args)
