"""
train.py
========
Training script for the DQN agent on the Healthcare Routing environment.

Run:
    python train.py                   # default training
    python train.py --episodes 500    # longer training
    python train.py --render          # enable console rendering
"""

import argparse
import sys
import os
import json
import numpy as np

# Allow running from any directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "openenv_env"))
sys.path.insert(0, os.path.dirname(__file__))

from healthcare_env import HealthcareRoutingEnv
from dqn_agent import DQNAgent


def train(
    num_episodes:   int   = 300,
    steps_per_ep:   int   = 100,
    hidden_dim:     int   = 256,
    lr:             float = 1e-3,
    gamma:          float = 0.99,
    render:         bool  = False,
    save_path:      str   = "models/dqn_healthcare.pth",
    log_interval:   int   = 20,
):
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

    env = HealthcareRoutingEnv(render_mode="human" if render else None)
    obs, _ = env.reset()

    agent = DQNAgent(
        obs_size    = env.observation_space.shape[0],
        action_size = env.action_space.n,
        hidden_dim  = hidden_dim,
        lr          = lr,
        gamma       = gamma,
    )

    print(f"\n{'='*60}")
    print(f"  Healthcare Routing DQN Training")
    print(f"  Episodes:   {num_episodes}")
    print(f"  Steps/ep:   {steps_per_ep}")
    print(f"  Obs size:   {env.observation_space.shape[0]}")
    print(f"  Actions:    {env.action_space.n}")
    print(f"  Device:     {agent.device}")
    print(f"{'='*60}\n")

    all_ep_rewards = []

    for episode in range(1, num_episodes + 1):
        obs, _ = env.reset()
        ep_reward = 0.0
        outcomes  = {"success": 0, "no_bed": 0, "ambulance_busy": 0}

        for step in range(steps_per_ep):
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated
            agent.store_transition(obs, action, reward, next_obs, done)
            agent.update()

            ep_reward += reward
            obs = next_obs
            outcomes[info.get("outcome", "success")] = outcomes.get(info.get("outcome", "success"), 0) + 1

            if render:
                env.render()

            if done:
                break

        all_ep_rewards.append(ep_reward)

        if episode % log_interval == 0:
            recent_mean = np.mean(all_ep_rewards[-log_interval:])
            stats = agent.get_stats()
            print(
                f"Episode {episode:4d}/{num_episodes}  |  "
                f"Reward: {ep_reward:8.1f}  |  "
                f"Mean({log_interval}): {recent_mean:8.1f}  |  "
                f"ε={stats['epsilon']:.3f}  |  "
                f"Loss={stats['mean_loss_100']:.4f}  |  "
                f"✅{outcomes['success']} ❌{outcomes['no_bed']} 🚑{outcomes['ambulance_busy']}"
            )

    # Save trained model
    agent.save(save_path)

    # Save training log
    log_path = save_path.replace(".pth", "_log.json")
    with open(log_path, "w") as f:
        json.dump({
            "episode_rewards": all_ep_rewards,
            "final_stats":     agent.get_stats(),
            "config": {
                "num_episodes": num_episodes,
                "steps_per_ep": steps_per_ep,
                "hidden_dim":   hidden_dim,
                "lr":           lr,
                "gamma":        gamma,
            }
        }, f, indent=2)

    print(f"\n[Train] Done! Final mean reward (last 50): {np.mean(all_ep_rewards[-50:]):.2f}")
    print(f"[Train] Log saved  → {log_path}")
    return agent, all_ep_rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN on HealthcareRoutingEnv")
    parser.add_argument("--episodes",  type=int,   default=300)
    parser.add_argument("--steps",     type=int,   default=100)
    parser.add_argument("--hidden",    type=int,   default=256)
    parser.add_argument("--lr",        type=float, default=1e-3)
    parser.add_argument("--render",    action="store_true")
    parser.add_argument("--save",      type=str,   default="models/dqn_healthcare.pth")
    args = parser.parse_args()

    train(
        num_episodes = args.episodes,
        steps_per_ep = args.steps,
        hidden_dim   = args.hidden,
        lr           = args.lr,
        render       = args.render,
        save_path    = args.save,
    )
