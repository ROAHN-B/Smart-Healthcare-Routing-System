"""
dqn_agent.py
============
Deep Q-Network (DQN) Agent using PyTorch
for the Healthcare Routing RL Environment.

Architecture:
  - 3-layer MLP with ReLU activations
  - Experience Replay buffer
  - Target network for stable training
  - Epsilon-greedy exploration
  - Trained with Adam optimizer & MSE loss

This is the PRIMARY ML model for the hackathon submission.
Uses Meta's PyTorch as required by the competition.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import json
import os
from collections import deque
from typing import Tuple, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Neural Network Architecture
# ─────────────────────────────────────────────────────────────────────────────

class DQNNetwork(nn.Module):
    """
    Multi-layer perceptron Q-network.

    Input:  observation vector (state)
    Output: Q-value for every possible action
    """

    def __init__(self, obs_size: int, action_size: int, hidden_dim: int = 256):
        super(DQNNetwork, self).__init__()

        self.network = nn.Sequential(
            # Layer 1
            nn.Linear(obs_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),

            # Layer 2
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),

            # Layer 3
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),

            # Output: one Q-value per action
            nn.Linear(hidden_dim // 2, action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ─────────────────────────────────────────────────────────────────────────────
# Experience Replay Buffer
# ─────────────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    """
    Circular buffer that stores (state, action, reward, next_state, done)
    tuples and provides random mini-batch sampling for DQN training.
    """

    def __init__(self, capacity: int = 10_000):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ─────────────────────────────────────────────────────────────────────────────
# DQN Agent
# ─────────────────────────────────────────────────────────────────────────────

class DQNAgent:
    """
    Full DQN agent with:
      - Online network (updated every step)
      - Target network (synced every `target_update` steps)
      - Epsilon-greedy exploration schedule
      - Experience replay
    """

    def __init__(
        self,
        obs_size:          int,
        action_size:       int,
        hidden_dim:        int   = 256,
        lr:                float = 1e-3,
        gamma:             float = 0.99,
        epsilon_start:     float = 1.0,
        epsilon_end:       float = 0.05,
        epsilon_decay:     float = 0.995,
        buffer_capacity:   int   = 10_000,
        batch_size:        int   = 64,
        target_update:     int   = 100,
        device:            str   = "auto",
    ):
        self.obs_size     = obs_size
        self.action_size  = action_size
        self.gamma        = gamma
        self.epsilon      = epsilon_start
        self.epsilon_end  = epsilon_end
        self.epsilon_decay= epsilon_decay
        self.batch_size   = batch_size
        self.target_update= target_update
        self.steps_done   = 0

        # Device selection
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Networks
        self.online_net = DQNNetwork(obs_size, action_size, hidden_dim).to(self.device)
        self.target_net = DQNNetwork(obs_size, action_size, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()  # target net is never directly trained

        # Optimizer & loss
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss_fn   = nn.MSELoss()

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)

        # Logging
        self.training_losses: List[float] = []
        self.episode_rewards: List[float] = []

    # ── Action selection ────────────────────────────────────────────────────

    def select_action(self, state: np.ndarray) -> int:
        """
        Epsilon-greedy action selection.
        With probability epsilon → random action (exploration)
        Otherwise             → greedy action from Q-network (exploitation)
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.online_net(state_t)
            return int(q_values.argmax(dim=1).item())

    def greedy_action(self, state: np.ndarray) -> int:
        """Pure greedy action (no exploration) – used at inference time."""
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.online_net(state_t)
            return int(q_values.argmax(dim=1).item())

    # ── Learning ────────────────────────────────────────────────────────────

    def store_transition(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self) -> Optional[float]:
        """
        Sample a mini-batch and perform one gradient update.
        Returns the loss value, or None if buffer is too small.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states_t      = torch.FloatTensor(states).to(self.device)
        actions_t     = torch.LongTensor(actions).to(self.device)
        rewards_t     = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t       = torch.FloatTensor(dones).to(self.device)

        # Current Q values: Q(s, a)
        current_q = self.online_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target Q values: r + γ * max_a' Q_target(s', a')  (0 if done)
        with torch.no_grad():
            max_next_q = self.target_net(next_states_t).max(dim=1)[0]
            target_q   = rewards_t + self.gamma * max_next_q * (1 - dones_t)

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stable training
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Sync target network periodically
        self.steps_done += 1
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        loss_val = loss.item()
        self.training_losses.append(loss_val)
        return loss_val

    # ── Persistence ─────────────────────────────────────────────────────────

    def save(self, path: str = "dqn_healthcare.pth"):
        """Save model weights and training metadata."""
        torch.save({
            "online_state_dict":  self.online_net.state_dict(),
            "target_state_dict":  self.target_net.state_dict(),
            "optimizer_state":    self.optimizer.state_dict(),
            "epsilon":            self.epsilon,
            "steps_done":         self.steps_done,
            "obs_size":           self.obs_size,
            "action_size":        self.action_size,
        }, path)
        print(f"[DQN] Model saved → {path}")

    def load(self, path: str = "dqn_healthcare.pth"):
        """Load saved model weights."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.online_net.load_state_dict(checkpoint["online_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.epsilon   = checkpoint["epsilon"]
        self.steps_done= checkpoint["steps_done"]
        print(f"[DQN] Model loaded ← {path}")

    def get_stats(self) -> dict:
        """Return a summary of training statistics."""
        recent = self.training_losses[-100:] if self.training_losses else [0]
        return {
            "steps_done":     self.steps_done,
            "epsilon":        round(self.epsilon, 4),
            "buffer_size":    len(self.replay_buffer),
            "mean_loss_100":  round(float(np.mean(recent)), 4),
            "device":         str(self.device),
        }
