"""
healthcare_env.py
=================
AI-Powered Smart Healthcare Routing & Emergency Management
OpenEnv / Gymnasium-compatible RL Environment

Compatible with:
  - Meta's OpenEnv spec (https://github.com/meta-pytorch/OpenEnv)
  - HuggingFace openenv-course
  - Standard Gymnasium API

Author: Healthcare-RL Team
Hackathon: Meta PyTorch OpenEnv Hackathon x SST 2026
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, Dict, Tuple, Any
import math
import random


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two GPS points (in km).
    Uses the Haversine formula.
    """
    R = 6371.0  # Earth radius in kilometres
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def compute_eta(distance_km: float, traffic_factor: float = 1.0, speed_kmh: float = 60.0) -> float:
    """
    Estimate travel time in minutes.
    traffic_factor > 1 means heavier traffic (slower travel).
    """
    return (distance_km / speed_kmh) * 60.0 * traffic_factor


# ---------------------------------------------------------------------------
# Simulation Data (realistic Indian city coordinates - Bengaluru area)
# ---------------------------------------------------------------------------

DEFAULT_HOSPITALS = [
    {"id": 0, "name": "City General Hospital",   "lat": 12.9716, "lon": 77.5946, "total_beds": 100, "icu_beds": 20, "wait_time": 10},
    {"id": 1, "name": "Apex Medical Centre",     "lat": 12.9352, "lon": 77.6244, "total_beds": 80,  "icu_beds": 15, "wait_time": 15},
    {"id": 2, "name": "St. Mary's Hospital",     "lat": 13.0012, "lon": 77.5800, "total_beds": 60,  "icu_beds": 10, "wait_time": 5},
    {"id": 3, "name": "LifeCare Institute",      "lat": 12.9582, "lon": 77.6478, "total_beds": 120, "icu_beds": 30, "wait_time": 20},
    {"id": 4, "name": "Metro Emergency Hospital","lat": 12.9830, "lon": 77.6080, "total_beds": 90,  "icu_beds": 25, "wait_time": 8},
]

DEFAULT_AMBULANCES = [
    {"id": 0, "lat": 12.9600, "lon": 77.5900, "status": "available"},
    {"id": 1, "lat": 12.9800, "lon": 77.6100, "status": "available"},
    {"id": 2, "lat": 12.9450, "lon": 77.6300, "status": "available"},
    {"id": 3, "lat": 13.0000, "lon": 77.5700, "status": "available"},
]

NUM_HOSPITALS  = len(DEFAULT_HOSPITALS)
NUM_AMBULANCES = len(DEFAULT_AMBULANCES)


# ---------------------------------------------------------------------------
# Core RL Environment
# ---------------------------------------------------------------------------

class HealthcareRoutingEnv(gym.Env):
    """
    HealthcareRoutingEnv
    --------------------
    A Gymnasium-compatible (OpenEnv-spec) reinforcement learning environment
    for intelligent ambulance dispatch and hospital routing in emergency healthcare.

    PROBLEM STATEMENT
    -----------------
    When a patient calls for emergency help, the system must decide:
      1. Which ambulance to dispatch (closest + available)?
      2. Which hospital should receive the patient (best match for severity,
         beds available, distance, ICU availability)?

    The agent learns to maximise patient outcomes while minimising response
    times and avoiding poor resource allocation.

    STATE SPACE (observation)
    -------------------------
    A flat numpy array containing:
      - patient_severity      : float [0, 1]  (normalised 1–10)
      - patient_lat           : float [0, 1]  (normalised)
      - patient_lon           : float [0, 1]  (normalised)
      - traffic_condition     : float [0, 1]  (1 = worst traffic)
      - For each hospital (NUM_HOSPITALS):
        - beds_available_norm : float [0, 1]
        - icu_beds_norm       : float [0, 1]
        - distance_norm       : float [0, 1]
        - wait_time_norm      : float [0, 1]
      - For each ambulance (NUM_AMBULANCES):
        - distance_to_patient : float [0, 1]
        - is_available        : float {0, 1}

    ACTION SPACE
    ------------
    Discrete: NUM_HOSPITALS × NUM_AMBULANCES
    (i.e. choose one (hospital, ambulance) pair from all combinations)

    REWARD FUNCTION
    ---------------
    +100  patient successfully admitted
    +70   critical patient (severity >= 8) gets ICU bed
    +50   ambulance arrives fast (ETA < 10 min)
    -100  no bed available at chosen hospital
    -50   ambulance already busy / unavailable
    -distance_penalty   proportional to ambulance travel distance
    -wait_penalty       proportional to hospital wait time
    -severity_penalty   if critical patient sent to hospital with no ICU
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    # Geographic bounding box for normalisation (Bengaluru region)
    LAT_MIN, LAT_MAX = 12.85, 13.10
    LON_MIN, LON_MAX = 77.45, 77.75

    def __init__(
        self,
        hospitals: Optional[list] = None,
        ambulances: Optional[list] = None,
        render_mode: Optional[str] = None,
        max_steps: int = 200,
    ):
        super().__init__()

        self.hospitals_template  = hospitals  or DEFAULT_HOSPITALS
        self.ambulances_template = ambulances or DEFAULT_AMBULANCES
        self.render_mode = render_mode
        self.max_steps   = max_steps

        self.num_hospitals  = len(self.hospitals_template)
        self.num_ambulances = len(self.ambulances_template)

        # ── Action space ──────────────────────────────────────────────────
        # Flat index: action = hospital_id * num_ambulances + ambulance_id
        self.action_space = spaces.Discrete(self.num_hospitals * self.num_ambulances)

        # ── Observation space ─────────────────────────────────────────────
        obs_size = (
            4                          # patient severity, lat, lon, traffic
            + self.num_hospitals * 4   # beds, icu, distance, wait per hospital
            + self.num_ambulances * 2  # distance, availability per ambulance
        )
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )

        # Internal state (populated on reset)
        self.hospitals  = []
        self.ambulances = []
        self.patient    = {}
        self.step_count = 0
        self.episode_rewards = []

    # ------------------------------------------------------------------
    # OpenEnv / Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Reset the environment and return the initial observation."""
        super().reset(seed=seed)

        # Deep-copy hospital data so beds change per episode
        self.hospitals = [
            {
                **h,
                "beds_available": int(h["total_beds"] * random.uniform(0.2, 1.0)),
                "icu_available":  int(h["icu_beds"]   * random.uniform(0.1, 1.0)),
                "current_wait":   int(h["wait_time"]  * random.uniform(0.5, 2.0)),
            }
            for h in self.hospitals_template
        ]

        # Reset ambulances (all available, slight position jitter)
        self.ambulances = [
            {
                **a,
                "status": "available",
                "lat": a["lat"] + random.uniform(-0.01, 0.01),
                "lon": a["lon"] + random.uniform(-0.01, 0.01),
            }
            for a in self.ambulances_template
        ]

        # Generate a new emergency patient
        self.patient = self._generate_patient()
        self.step_count = 0

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one decision step.

        Parameters
        ----------
        action : int
            Flat index encoding (hospital_id, ambulance_id)

        Returns
        -------
        observation, reward, terminated, truncated, info
        """
        self.step_count += 1

        # Decode action
        hospital_id  = action // self.num_ambulances
        ambulance_id = action  % self.num_ambulances

        hospital  = self.hospitals[hospital_id]
        ambulance = self.ambulances[ambulance_id]

        reward, outcome = self._compute_reward(hospital, ambulance)

        # Update world state after the assignment
        if outcome != "no_bed" and outcome != "ambulance_busy":
            self._update_state(hospital_id, ambulance_id)

        # Generate the next patient for the next step
        self.patient = self._generate_patient()

        obs        = self._get_observation()
        terminated = False                       # episode runs for max_steps
        truncated  = self.step_count >= self.max_steps
        info       = self._get_info()
        info["outcome"]      = outcome
        info["hospital_id"]  = hospital_id
        info["ambulance_id"] = ambulance_id
        info["reward"]       = reward

        self.episode_rewards.append(reward)
        return obs, reward, terminated, truncated, info

    def render(self):
        """Simple human-readable console render."""
        if self.render_mode == "human":
            p = self.patient
            print(
                f"\n[Step {self.step_count:3d}] "
                f"Patient severity={p['severity']:.1f} "
                f"@ ({p['lat']:.4f}, {p['lon']:.4f}) | "
                f"Traffic={p['traffic']:.2f}"
            )
            for h in self.hospitals:
                print(
                    f"  🏥 {h['name']:30s} "
                    f"beds={h['beds_available']:3d}/{h['total_beds']:3d}  "
                    f"ICU={h['icu_available']:2d}/{h['icu_beds']:2d}  "
                    f"wait={h['current_wait']:2d}min"
                )
            for a in self.ambulances:
                print(
                    f"  🚑 Ambulance-{a['id']} "
                    f"status={a['status']:10s} "
                    f"@ ({a['lat']:.4f}, {a['lon']:.4f})"
                )

    def close(self):
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_patient(self) -> Dict:
        """Randomly generate a new emergency patient within the bounding box."""
        return {
            "severity": round(random.uniform(1, 10), 1),
            "lat":     random.uniform(self.LAT_MIN, self.LAT_MAX),
            "lon":     random.uniform(self.LON_MIN, self.LON_MAX),
            "traffic": round(random.uniform(0.8, 2.5), 2),  # traffic factor
        }

    def _get_observation(self) -> np.ndarray:
        """Build the flat observation vector."""
        p = self.patient

        # Normalise patient fields
        sev_norm     = (p["severity"] - 1) / 9.0
        lat_norm     = (p["lat"] - self.LAT_MIN) / (self.LAT_MAX - self.LAT_MIN)
        lon_norm     = (p["lon"] - self.LON_MIN) / (self.LON_MAX - self.LON_MIN)
        traffic_norm = min((p["traffic"] - 0.8) / 1.7, 1.0)

        obs = [sev_norm, lat_norm, lon_norm, traffic_norm]

        # Hospital features
        max_beds = max(h["total_beds"] for h in self.hospitals) or 1
        max_icu  = max(h["icu_beds"]   for h in self.hospitals) or 1
        max_wait = 120.0  # normalise wait time up to 120 min

        for h in self.hospitals:
            dist = haversine_distance(p["lat"], p["lon"], h["lat"], h["lon"])
            obs += [
                h["beds_available"] / max_beds,
                h["icu_available"]  / max_icu,
                min(dist / 50.0, 1.0),             # normalise distance up to 50 km
                min(h["current_wait"] / max_wait, 1.0),
            ]

        # Ambulance features
        for a in self.ambulances:
            dist = haversine_distance(p["lat"], p["lon"], a["lat"], a["lon"])
            obs += [
                min(dist / 50.0, 1.0),
                1.0 if a["status"] == "available" else 0.0,
            ]

        return np.array(obs, dtype=np.float32)

    def _compute_reward(self, hospital: Dict, ambulance: Dict) -> Tuple[float, str]:
        """
        Calculate the reward for assigning a patient to this
        (hospital, ambulance) pair.
        """
        p      = self.patient
        reward = 0.0

        # ── Penalty: ambulance not available ─────────────────────────────
        if ambulance["status"] != "available":
            return -50.0, "ambulance_busy"

        # ── Penalty: no bed available ─────────────────────────────────────
        if hospital["beds_available"] <= 0:
            return -100.0, "no_bed"

        # ── Base reward: patient admitted ─────────────────────────────────
        reward += 100.0

        # ── Bonus: critical patient gets ICU ─────────────────────────────
        if p["severity"] >= 8:
            if hospital["icu_available"] > 0:
                reward += 70.0
            else:
                reward -= 40.0  # severity_penalty: no ICU for critical

        # ── Bonus: fast ambulance arrival ─────────────────────────────────
        amb_to_patient = haversine_distance(
            ambulance["lat"], ambulance["lon"], p["lat"], p["lon"]
        )
        eta = compute_eta(amb_to_patient, p["traffic"])
        if eta < 10.0:
            reward += 50.0
        elif eta < 20.0:
            reward += 20.0

        # ── Distance penalty (ambulance to patient) ───────────────────────
        reward -= min(amb_to_patient * 2.0, 40.0)  # up to -40

        # ── Wait time penalty ─────────────────────────────────────────────
        reward -= min(hospital["current_wait"] * 0.5, 30.0)  # up to -30

        # ── Severity × delay penalty ──────────────────────────────────────
        # If patient is critical but ETA is high → extra penalty
        if p["severity"] >= 7 and eta > 15:
            reward -= (p["severity"] - 7) * 5.0

        return round(reward, 2), "success"

    def _update_state(self, hospital_id: int, ambulance_id: int):
        """Consume resources after a successful assignment."""
        h = self.hospitals[hospital_id]
        a = self.ambulances[ambulance_id]

        # Decrement beds
        h["beds_available"] = max(0, h["beds_available"] - 1)
        if self.patient["severity"] >= 8 and h["icu_available"] > 0:
            h["icu_available"] -= 1

        # Slightly increase wait time due to load
        h["current_wait"] = min(h["current_wait"] + random.randint(0, 3), 120)

        # Mark ambulance busy (would return to available after a trip in full sim)
        a["status"] = "busy"

        # Free one random busy ambulance so the episode doesn't dead-lock
        busy = [x for x in self.ambulances if x["status"] == "busy"]
        if len(busy) == self.num_ambulances and busy:
            random.choice(busy)["status"] = "available"

    def _get_info(self) -> Dict[str, Any]:
        """Return auxiliary info for logging / debugging."""
        return {
            "step":              self.step_count,
            "patient_severity":  self.patient.get("severity", 0),
            "available_beds":    sum(h["beds_available"] for h in self.hospitals),
            "available_ambs":    sum(1 for a in self.ambulances if a["status"] == "available"),
            "episode_mean_reward": (
                np.mean(self.episode_rewards) if self.episode_rewards else 0.0
            ),
        }

    # ------------------------------------------------------------------
    # Convenience: decode action ↔ (hospital, ambulance)
    # ------------------------------------------------------------------

    def decode_action(self, action: int) -> Tuple[int, int]:
        """Return (hospital_id, ambulance_id) from flat action index."""
        return action // self.num_ambulances, action % self.num_ambulances

    def encode_action(self, hospital_id: int, ambulance_id: int) -> int:
        """Return flat action index from (hospital_id, ambulance_id)."""
        return hospital_id * self.num_ambulances + ambulance_id

    def get_greedy_action(self) -> int:
        """
        Rule-based greedy baseline (useful for comparison with RL).
        Picks nearest available ambulance + hospital with most beds.
        """
        p = self.patient
        # Best ambulance: nearest available
        best_amb = min(
            (a for a in self.ambulances if a["status"] == "available"),
            key=lambda a: haversine_distance(p["lat"], p["lon"], a["lat"], a["lon"]),
            default=self.ambulances[0],
        )
        # Best hospital: max beds, weighted by distance
        best_hosp = max(
            self.hospitals,
            key=lambda h: (
                h["beds_available"] * 10
                - haversine_distance(p["lat"], p["lon"], h["lat"], h["lon"])
            ),
        )
        return self.encode_action(best_hosp["id"], best_amb["id"])
