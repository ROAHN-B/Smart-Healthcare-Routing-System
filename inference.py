import asyncio
import os
import textwrap
import sys
import numpy as np
from typing import List, Optional
import torch
from openai import OpenAI

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT_DIR, "openenv_env"))
sys.path.insert(0, os.path.join(ROOT_DIR, "rl"))

from healthcare_env import HealthcareRoutingEnv
from dqn_agent import DQNAgent

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
HF_TOKEN = os.getenv("HF_TOKEN", "dummy-token-if-empty")

BENCHMARK = "healthcare-routing"
MAX_STEPS = 10

# The 3 tasks we just created in task.py
TASKS = ["healthcare-routing-easy", "healthcare-routing-medium", "healthcare-routing-hard"]

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

async def run_scenario(client: OpenAI, agent: DQNAgent, task_name: str) -> float:
    env = HealthcareRoutingEnv()
    rewards: List[float] = []
    steps_taken = 0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs, info = env.reset()

        for step in range(1, MAX_STEPS + 1):
            try:
                prompt = f"System state array: {obs.tolist()}. Give a 1-word strategy (e.g., 'Reroute' or 'Wait')."
                response = await asyncio.to_thread(
                    client.chat.completions.create,
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=5,
                    temperature=0.0,
                )
                raw_response = response.choices[0].message.content.strip()
                llm_strategy = raw_response.replace("`", "").strip()
            except Exception:
                llm_strategy = "Fallback"

            try:
                state_tensor = torch.FloatTensor(obs).unsqueeze(0)
                net = getattr(agent, "model", getattr(agent, "policy_net", getattr(agent, "q_net", getattr(agent, "network", None))))
                if net is not None:
                    with torch.no_grad():
                        q_values = net(state_tensor).cpu().numpy()[0]
                    NUM_HOSPITALS = 5
                    NUM_AMBULANCES = 4
                    for action_idx in range(NUM_HOSPITALS * NUM_AMBULANCES):
                        h_id = action_idx // NUM_AMBULANCES
                        a_id = action_idx % NUM_AMBULANCES
                        hospital_beds_norm = obs[4 + (h_id * 4)]
                        ambulance_available = obs[24 + (a_id * 2) + 1]
                        if hospital_beds_norm <= 0.0 or ambulance_available == 0.0:
                            q_values[action_idx] = -999999.0
                    action_int = int(np.argmax(q_values))
                else:
                    action_int = env.get_greedy_action()
            except Exception:
                action_int = env.get_greedy_action()

            obs, reward, terminated, truncated, info = env.step(action_int)
            done = terminated or truncated

            rewards.append(float(reward))
            steps_taken = step

            log_step(step=step, action=str(action_int), reward=float(reward), done=done, error=None)

            if done:
                break

        total_reward = sum(rewards)
        raw_score = total_reward / 800.0
        # Phase 2 score clamping to strictly (0, 1)
        score = min(max(raw_score, 0.01), 0.99)
        success = any(r > 50 for r in rewards)

    except Exception as e:
        log_step(step=steps_taken + 1, action="-1", reward=0.0, done=True, error=str(e))
        score = 0.01
        success = False

    finally:
        env.close()
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env = HealthcareRoutingEnv()
    obs_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(obs_size, action_size)

    model_path = os.path.join(ROOT_DIR, "rl", "models", "dqn_healthcare.pth")
    if os.path.exists(model_path):
        try:
            agent.load(model_path)
        except Exception as e:
            print(f"[Warning] Could not load PyTorch weights. Error: {e}")

    # Loop through the 3 required tasks
    for task_name in TASKS:
        await run_scenario(client, agent, task_name)

if __name__ == "__main__":
    asyncio.run(main())