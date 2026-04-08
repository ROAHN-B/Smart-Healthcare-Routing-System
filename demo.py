"""
demo.py
=======
Standalone demo script — runs the RL environment and shows assignment
decisions in the terminal. No backend server needed.

Run:
    python demo.py
    python demo.py --episodes 5 --steps 20
"""

import sys
import os
import argparse
import time

# Make sure local modules are on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "openenv_env"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "rl"))

from healthcare_env import HealthcareRoutingEnv, haversine_distance, compute_eta

RESET  = "\033[0m"
RED    = "\033[91m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
BLUE   = "\033[94m"


def severity_color(sev):
    if sev >= 8:   return RED
    if sev >= 5:   return YELLOW
    return GREEN


def severity_label(sev):
    if sev >= 8:   return "CRITICAL"
    if sev >= 5:   return "MODERATE"
    return "MILD"


def print_header():
    print(f"\n{BOLD}{CYAN}{'='*65}{RESET}")
    print(f"{BOLD}{CYAN}  🚑  SmartER — AI Healthcare Routing Demo{RESET}")
    print(f"{BOLD}{CYAN}       Meta PyTorch OpenEnv Hackathon 2026{RESET}")
    print(f"{BOLD}{CYAN}{'='*65}{RESET}\n")


def print_step(step, patient, hospital, ambulance, reward, outcome, eta):
    sc = severity_color(patient["severity"])
    sl = severity_label(patient["severity"])

    print(f"{BOLD}[Step {step:3d}]{RESET}  Patient severity: {sc}{patient['severity']:.1f} — {sl}{RESET}")
    print(f"         Location : ({patient['lat']:.4f}, {patient['lon']:.4f})  |  Traffic: {patient['traffic']:.2f}x")

    if outcome == "success":
        print(f"         {GREEN}✅ ASSIGNED{RESET}  →  🏥 {hospital['name']}")
        print(f"                    beds={hospital['beds_available']}  ICU={hospital['icu_available']}  wait={hospital['current_wait']}min")
        print(f"                    🚑 {ambulance.get('name', ambulance['id'])}  |  ETA: {CYAN}{eta:.1f} min{RESET}")
    elif outcome == "no_bed":
        print(f"         {RED}❌ NO BED   →  {hospital['name']}{RESET}")
    else:
        print(f"         {YELLOW}⚠️  AMB BUSY  →  {ambulance.get('name', ambulance['id'])}{RESET}")

    color = GREEN if reward > 0 else RED
    print(f"         Reward   : {color}{reward:+.1f}{RESET}")
    print()


def run_demo(episodes=3, steps_per_ep=15, use_greedy=True, delay=0.3):
    print_header()
    env = HealthcareRoutingEnv(render_mode=None)

    total_rewards = []

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        ep_reward = 0.0
        outcomes  = {"success": 0, "no_bed": 0, "ambulance_busy": 0}

        print(f"{BOLD}{BLUE}── Episode {ep}/{episodes} {'─'*45}{RESET}")
        print(f"   Hospitals   : {len(env.hospitals)}   |   Ambulances: {len(env.ambulances)}")
        print(f"   Total beds  : {sum(h['total_beds'] for h in env.hospitals)}")
        print()

        for step in range(1, steps_per_ep + 1):
            # Choose action
            if use_greedy:
                action = env.get_greedy_action()
            else:
                action = env.action_space.sample()

            next_obs, reward, terminated, truncated, info = env.step(action)

            # Decode for display
            hosp_idx, amb_idx = env.decode_action(action)
            hospital  = env.hospitals[hosp_idx]
            ambulance = env.ambulances[amb_idx]
            patient   = env.patient  # next patient already set — use prev
            # Use the patient from before the step (stored in info implicitly)
            # Recreate approximate ETA for display
            dist = haversine_distance(
                ambulance["lat"], ambulance["lon"],
                info.get("patient_lat", env.patient["lat"]),
                info.get("patient_lon", env.patient["lon"]),
            )
            eta = compute_eta(dist, 1.2, 60)

            print_step(
                step    = step,
                patient = {
                    "severity": info.get("patient_severity", env.patient["severity"]),
                    "lat":      env.patient["lat"],
                    "lon":      env.patient["lon"],
                    "traffic":  env.patient["traffic"],
                },
                hospital  = hospital,
                ambulance = ambulance,
                reward    = reward,
                outcome   = info.get("outcome", "success"),
                eta       = eta,
            )

            ep_reward += reward
            outcomes[info.get("outcome", "success")] = outcomes.get(info.get("outcome", "success"), 0) + 1

            if delay > 0:
                time.sleep(delay)

            if terminated or truncated:
                break

        total_rewards.append(ep_reward)
        print(f"{BOLD}  Episode {ep} Summary:{RESET}")
        print(f"    Total Reward : {GREEN if ep_reward > 0 else RED}{ep_reward:.1f}{RESET}")
        print(f"    Successes    : {GREEN}{outcomes['success']}{RESET}   "
              f"No-Bed: {RED}{outcomes['no_bed']}{RESET}   "
              f"Amb Busy: {YELLOW}{outcomes.get('ambulance_busy', 0)}{RESET}")
        print()

    print(f"{BOLD}{CYAN}{'='*65}{RESET}")
    print(f"{BOLD}  Overall Mean Reward: {sum(total_rewards)/len(total_rewards):.2f}{RESET}")
    print(f"  Policy: {'Greedy (rule-based)' if use_greedy else 'Random'}")
    print(f"\n  To train the DQN model, run:{RESET}")
    print(f"    {CYAN}cd rl && python train.py --episodes 300{RESET}")
    print(f"\n  To start the full system, run:{RESET}")
    print(f"    {CYAN}cd backend/api && uvicorn main:app --reload{RESET}")
    print(f"    {CYAN}cd frontend     && npm install && npm run dev{RESET}")
    print(f"{BOLD}{CYAN}{'='*65}{RESET}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Healthcare Routing Demo")
    parser.add_argument("--episodes", type=int,   default=3)
    parser.add_argument("--steps",    type=int,   default=15)
    parser.add_argument("--random",   action="store_true", help="Use random policy instead of greedy")
    parser.add_argument("--fast",     action="store_true", help="No delay between steps")
    args = parser.parse_args()

    run_demo(
        episodes      = args.episodes,
        steps_per_ep  = args.steps,
        use_greedy    = not args.random,
        delay         = 0.0 if args.fast else 0.3,
    )
