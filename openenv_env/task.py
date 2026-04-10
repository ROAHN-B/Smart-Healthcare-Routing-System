"""
task.py
=======
OpenEnv Task Definition for Healthcare Routing RL Environment.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np

from openenv_env.healthcare_env import HealthcareRoutingEnv

# ---------------------------------------------------------------------------
# 1. Define 3 Distinct Tasks (Required for Phase 2 Validation)
# ---------------------------------------------------------------------------

@dataclass
class HealthcareTaskEasy:
    task_id: str           = "healthcare-routing-easy"
    version: str           = "1.0.0"
    description: str       = "Easy routing scenario with 50 steps."
    tags: List[str]        = field(default_factory=lambda: ["healthcare", "easy"])
    difficulty: str        = "easy"
    env_kwargs: Dict[str, Any] = field(default_factory=lambda: {"max_steps": 50, "config": {"max_patients": 50, "traffic_mult": 1.0}})

    def make_env(self, render_mode: Optional[str] = None) -> HealthcareRoutingEnv:
        return HealthcareRoutingEnv(render_mode=render_mode, **self.env_kwargs)

@dataclass
class HealthcareTaskMedium:
    task_id: str           = "healthcare-routing-medium"
    version: str           = "1.0.0"
    description: str       = "Medium routing scenario with 100 steps."
    tags: List[str]        = field(default_factory=lambda: ["healthcare", "medium"])
    difficulty: str        = "medium"
    env_kwargs: Dict[str, Any] = field(default_factory=lambda: {"max_steps": 100, "config": {"max_patients": 100, "traffic_mult": 1.5}})

    def make_env(self, render_mode: Optional[str] = None) -> HealthcareRoutingEnv:
        return HealthcareRoutingEnv(render_mode=render_mode, **self.env_kwargs)

@dataclass
class HealthcareTaskHard:
    task_id: str           = "healthcare-routing-hard"
    version: str           = "1.0.0"
    description: str       = "Hard routing scenario with 200 steps."
    tags: List[str]        = field(default_factory=lambda: ["healthcare", "hard"])
    difficulty: str        = "hard"
    env_kwargs: Dict[str, Any] = field(default_factory=lambda: {"max_steps": 200, "config": {"max_patients": 200, "traffic_mult": 2.5}})

    def make_env(self, render_mode: Optional[str] = None) -> HealthcareRoutingEnv:
        return HealthcareRoutingEnv(render_mode=render_mode, **self.env_kwargs)

# ---------------------------------------------------------------------------
# 2. Base Grader Logic
# ---------------------------------------------------------------------------

class BaseHealthcareGrader:
    REWARD_BENCHMARK   = 120.0   
    EVAL_EPISODES      = 10
    EVAL_STEPS_PER_EP  = 50

    def __init__(self, task):
        self.task = task

    def grade(self, policy) -> Dict[str, Any]:
        env = self.task.make_env()

        total_reward    = 0.0
        total_steps     = 0
        success_count   = 0
        critical_icu    = 0
        critical_total  = 0

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
                    if env.patient["severity"] >= 8:
                        critical_total += 1
                        hosp = env.hospitals[info["hospital_id"]]
                        if hosp.get("icu_available", 0) > 0 or hosp["icu_beds"] > 0:
                            critical_icu += 1

                if terminated or truncated:
                    break

            total_reward += ep_reward

        mean_reward  = total_reward / self.EVAL_EPISODES
        success_rate = success_count / max(total_steps, 1)
        icu_rate     = critical_icu  / max(critical_total, 1)

        # Base calculations
        reward_score = min(40, max(0, (mean_reward / self.REWARD_BENCHMARK) * 40))
        success_score = success_rate * 30
        icu_score     = icu_rate     * 20
        amb_score     = 10.0          

        total_score_100 = reward_score + success_score + icu_score + amb_score

        # --- MANDATORY PHASE 2 FIX: Clamp score strictly to (0.01, 0.99) ---
        final_score = min(max(total_score_100 / 100.0, 0.01), 0.99)

        return {
            "score":        round(final_score, 4),
            "max_score":    1.0,
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
# 3. Expose 3 Graders to OpenEnv
# ---------------------------------------------------------------------------

class HealthcareGraderEasy(BaseHealthcareGrader):
    def __init__(self, task=None):
        super().__init__(task or HealthcareTaskEasy())

class HealthcareGraderMedium(BaseHealthcareGrader):
    def __init__(self, task=None):
        super().__init__(task or HealthcareTaskMedium())

class HealthcareGraderHard(BaseHealthcareGrader):
    def __init__(self, task=None):
        super().__init__(task or HealthcareTaskHard())
