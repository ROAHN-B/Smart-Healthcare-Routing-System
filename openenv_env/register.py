"""
register.py
===========
Register the HealthcareRoutingEnv with Gymnasium so it can be created
with gym.make("healthcare-routing-v1").

Also provides the OpenEnv-spec environment_config dict for
submission to the HuggingFace Hub.

Usage:
    python register.py          # verify registration works
    import register             # in any script to auto-register
"""

import gymnasium as gym
from gymnasium.envs.registration import register

# ── Gymnasium registration ───────────────────────────────────────────────────
register(
    id              = "healthcare-routing-v1",
    entry_point     = "openenv_env.healthcare_env:HealthcareRoutingEnv",
    max_episode_steps = 200,
    reward_threshold  = 100.0,
    kwargs          = {},
)

# ── OpenEnv spec dict (for Hub submission) ───────────────────────────────────
ENVIRONMENT_CONFIG = {
    "env_id":           "healthcare-routing-v1",
    "version":          "1.0.0",
    "description":      (
        "AI-Powered Smart Healthcare Routing & Emergency Management. "
        "An RL agent dispatches the optimal ambulance and hospital for each "
        "incoming emergency patient, maximising survival outcomes while "
        "minimising response times and wasted resources."
    ),
    "tags":             ["healthcare", "routing", "multi-objective", "real-world", "emergency"],
    "observation_type": "Box",
    "action_type":      "Discrete",
    "reward_range":     (-200, 220),
    "authors":          ["Healthcare-RL Team"],
    "license":          "MIT",
    "gym_make_kwargs":  {},
    "entry_point":      "openenv_env.healthcare_env:HealthcareRoutingEnv",
}


def make_env(**kwargs):
    """Convenience factory used by OpenEnv infrastructure."""
    from openenv_env.healthcare_env import HealthcareRoutingEnv
    return HealthcareRoutingEnv(**kwargs)


if __name__ == "__main__":
    print("Registering healthcare-routing-v1 with Gymnasium …")
    env = gym.make("healthcare-routing-v1")
    obs, info = env.reset()
    print(f"  ✅ Environment created successfully")
    print(f"  Observation shape : {obs.shape}")
    print(f"  Action space      : {env.action_space}")
    print(f"  Config            : {ENVIRONMENT_CONFIG['env_id']} v{ENVIRONMENT_CONFIG['version']}")
    env.close()
    print("Registration OK 🎉")
