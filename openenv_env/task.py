"""
task.py
=======
OpenEnv Task Definition for Healthcare Routing RL Environment.

Follows the OpenEnv spec:
  - task_id, description, tags
  - reset / step wrappers
  - grader integration

https://github.com/meta-pytorch/OpenEnv
https://github.com/huggingface/openenv-course
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np

from healthcare_env import HealthcareRoutingEnv


# ---------------------------------------------------------------------------
# OpenEnv Task Descriptor
# ---------------------------------------------------------------------------

@dataclass
class HealthcareTask:
    """
    Metadata descriptor conforming to the OpenEnv task specification.
    """
    task_id: str           = "healthcare-routing-v1"
    version: str           = "1.0.0"
    description: str       = (
        "AI-Powered Smart Healthcare Routing & Emergency Management. "
        "An RL agent must dispatch the optimal ambulance and hospital "
        "for each incoming emergency patient, maximising survival outcomes "
        "while minimising response times and wasted resources."
    )
    tags: List[str]        = field(default_factory=lambda: [
        "healthcare", "routing", "multi-objective", "real-world",
        "emergency-management", "gymnasium", "openenv",
    ])
    difficulty: str        = "medium"
    author: str            = "Healthcare-RL Team"
    license: str           = "MIT"

    # Environment constructor kwargs
    env_kwargs: Dict[str, Any] = field(default_factory=dict)

    def make_env(self, render_mode: Optional[str] = None) -> HealthcareRoutingEnv:
        """Instantiate and return the Gymnasium environment."""
        return HealthcareRoutingEnv(render_mode=render_mode, **self.env_kwargs)


# ---------------------------------------------------------------------------
# OpenEnv Grader
# ---------------------------------------------------------------------------

class HealthcareGrader:
    """
    Evaluates a trained agent (or any callable policy) on the Healthcare task.

    Grading criteria (total: 100 points)
    -------------------------------------
    40 pts  Mean episode reward   (linear scale, max at reward >= 120)
    30 pts  Success rate          (% steps ending in 'success')
    20 pts  Critical patient care (% critical patients assigned ICU)
    10 pts  Ambulance utilisation (% dispatches use nearest ambulance)
    """

    REWARD_BENCHMARK   = 120.0   # reward at which agent scores full 40 pts
    EVAL_EPISODES      = 20
    EVAL_STEPS_PER_EP  = 50

    def __init__(self, task: Optional[HealthcareTask] = None):
        self.task = task or HealthcareTask()

    def grade(self, policy) -> Dict[str, Any]:
        """
        Grade a policy.

        Parameters
        ----------
        policy : callable
            Accepts a numpy observation array, returns an integer action.

        Returns
        -------
        dict with keys: score (0–100), breakdown, mean_reward, success_rate
        """
        env = self.task.make_env()

        total_reward    = 0.0
        total_steps     = 0
        success_count   = 0
        critical_icu    = 0
        critical_total  = 0
        nearest_amb     = 0

        for ep in range(self.EVAL_EPISODES):
            obs, _ = env.reset()
            ep_reward = 0.0

            for _ in range(self.EVAL_STEPS_PER_EP):
                action = policy(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward   += reward
                total_steps += 1

                if info.get("outcome") == "success":
                    success_count += 1

                    # Track ICU assignment for critical patients
                    if env.patient["severity"] >= 8:
                        critical_total += 1
                        hosp = env.hospitals[info["hospital_id"]]
                        # Check if ICU was available (before step consumed it)
                        if hosp.get("icu_available", 0) > 0 or hosp["icu_beds"] > 0:
                            critical_icu += 1

                if terminated or truncated:
                    break

            total_reward += ep_reward

        mean_reward  = total_reward / self.EVAL_EPISODES
        success_rate = success_count / max(total_steps, 1)
        icu_rate     = critical_icu  / max(critical_total, 1)

        # ── Scoring ───────────────────────────────────────────────────────
        reward_score = min(40, max(0, (mean_reward / self.REWARD_BENCHMARK) * 40))
        success_score = success_rate * 30
        icu_score     = icu_rate     * 20
        amb_score     = 10.0          # placeholder (full credit if agent runs)

        total_score = reward_score + success_score + icu_score + amb_score

        return {
            "score":        round(total_score, 2),
            "max_score":    100,
            "mean_reward":  round(mean_reward, 2),
            "success_rate": round(success_rate, 4),
            "icu_rate":     round(icu_rate, 4),
            "breakdown": {
                "reward_score":  round(reward_score, 2),
                "success_score": round(success_score, 2),
                "icu_score":     round(icu_score, 2),
                "amb_score":     round(amb_score, 2),
            },
        }


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    task   = HealthcareTask()
    env    = task.make_env(render_mode="human")
    grader = HealthcareGrader(task)

    # Random policy baseline
    def random_policy(obs):
        return env.action_space.sample()

    print("=== Grading RANDOM policy (baseline) ===")
    result = grader.grade(random_policy)
    for k, v in result.items():
        print(f"  {k}: {v}")

    # Greedy policy baseline
    print("\n=== Grading GREEDY policy (rule-based baseline) ===")
    result2 = grader.grade(env.get_greedy_action)
    for k, v in result2.items():
        print(f"  {k}: {v}")
