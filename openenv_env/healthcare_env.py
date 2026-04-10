"""
healthcare_env.py
=================
AI-Powered Smart Healthcare Routing & Emergency Management
OpenEnv / Gymnasium-compatible RL Environment
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, Dict, Tuple, Any
import math
import random

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0 
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def compute_eta(distance_km: float, traffic_factor: float = 1.0, speed_kmh: float = 60.0) -> float:
    return (distance_km / speed_kmh) * 60.0 * traffic_factor

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

class HealthcareRoutingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    LAT_MIN, LAT_MAX = 12.85, 13.10
    LON_MIN, LON_MAX = 77.45, 77.75

    def __init__(
        self,
        config: Optional[dict] = None,
        hospitals: Optional[list] = None,
        ambulances: Optional[list] = None,
        render_mode: Optional[str] = None,
        max_steps: int = 200,
    ):
        super().__init__()
        self.config = config or {"max_patients": 20, "traffic_mult": 1.0}
        
        self.hospitals_template  = hospitals  or DEFAULT_HOSPITALS
        self.ambulances_template = ambulances or DEFAULT_AMBULANCES
        self.render_mode = render_mode
        self.max_steps   = max_steps

        self.num_hospitals  = len(self.hospitals_template)
        self.num_ambulances = len(self.ambulances_template)

        self.action_space = spaces.Discrete(self.num_hospitals * self.num_ambulances)

        obs_size = (4 + self.num_hospitals * 4 + self.num_ambulances * 2)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32)

        self.hospitals  = []
        self.ambulances = []
        self.patient    = {}
        self.step_count = 0
        self.episode_rewards = []

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        self.hospitals = [
            {
                **h,
                "beds_available": int(h["total_beds"] * random.uniform(0.2, 1.0)),
                "icu_available":  int(h["icu_beds"]   * random.uniform(0.1, 1.0)),
                "current_wait":   int(h["wait_time"]  * random.uniform(0.5, 2.0)),
            }
            for h in self.hospitals_template
        ]

        self.ambulances = [
            {
                **a,
                "status": "available",
                "lat": a["lat"] + random.uniform(-0.01, 0.01),
                "lon": a["lon"] + random.uniform(-0.01, 0.01),
            }
            for a in self.ambulances_template
        ]

        self.patient = self._generate_patient()
        self.step_count = 0
        
        self.metrics = {
            "patients_admitted": 0,
            "invalid_actions": 0,
            "critical_saved": 0
        }

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self.step_count += 1

        hospital_id  = action // self.num_ambulances
        ambulance_id = action  % self.num_ambulances

        hospital  = self.hospitals[hospital_id]
        ambulance = self.ambulances[ambulance_id]

        reward, outcome = self._compute_reward(hospital, ambulance)

        if outcome in ["ambulance_busy", "no_bed"]:
            self.metrics["invalid_actions"] += 1
        elif outcome == "success":
            self.metrics["patients_admitted"] += 1
            if self.patient["severity"] >= 8 and hospital["icu_available"] > 0:
                self.metrics["critical_saved"] += 1

        if outcome != "no_bed" and outcome != "ambulance_busy":
            self._update_state(hospital_id, ambulance_id)

        self.patient = self._generate_patient()

        obs        = self._get_observation()
        terminated = False
        truncated  = self.step_count >= self.max_steps
        info       = self._get_info()
        
        info["outcome"]      = outcome
        info["hospital_id"]  = hospital_id
        info["ambulance_id"] = ambulance_id
        info["reward"]       = reward

        self.episode_rewards.append(reward)
        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            print(f"\n[Step {self.step_count:3d}] Patient severity={self.patient['severity']:.1f}")

    def close(self):
        pass

    def _generate_patient(self) -> Dict:
        return {
            "severity": round(random.uniform(1, 10), 1),
            "lat":     random.uniform(self.LAT_MIN, self.LAT_MAX),
            "lon":     random.uniform(self.LON_MIN, self.LON_MAX),
            "traffic": round(random.uniform(0.8, 2.5) * self.config["traffic_mult"], 2),
        }

    def _get_observation(self) -> np.ndarray:
        p = self.patient
        sev_norm     = (p["severity"] - 1) / 9.0
        lat_norm     = (p["lat"] - self.LAT_MIN) / (self.LAT_MAX - self.LAT_MIN)
        lon_norm     = (p["lon"] - self.LON_MIN) / (self.LON_MAX - self.LON_MIN)
        traffic_norm = min((p["traffic"] - 0.8) / 1.7, 1.0)

        obs = [sev_norm, lat_norm, lon_norm, traffic_norm]

        max_beds = max(h["total_beds"] for h in self.hospitals) or 1
        max_icu  = max(h["icu_beds"]   for h in self.hospitals) or 1
        max_wait = 120.0 

        for h in self.hospitals:
            dist = haversine_distance(p["lat"], p["lon"], h["lat"], h["lon"])
            obs += [
                h["beds_available"] / max_beds,
                h["icu_available"]  / max_icu,
                min(dist / 50.0, 1.0),             
                min(h["current_wait"] / max_wait, 1.0),
            ]

        for a in self.ambulances:
            dist = haversine_distance(p["lat"], p["lon"], a["lat"], a["lon"])
            obs += [
                min(dist / 50.0, 1.0),
                1.0 if a["status"] == "available" else 0.0,
            ]

        return np.array(obs, dtype=np.float32)

    def _compute_reward(self, hospital: Dict, ambulance: Dict) -> Tuple[float, str]:
        p      = self.patient
        reward = 0.0

        if ambulance["status"] != "available": return -50.0, "ambulance_busy"
        if hospital["beds_available"] <= 0: return -100.0, "no_bed"

        reward += 100.0

        if p["severity"] >= 8:
            if hospital["icu_available"] > 0: reward += 70.0
            else: reward -= 40.0 

        amb_to_patient = haversine_distance(ambulance["lat"], ambulance["lon"], p["lat"], p["lon"])
        eta = compute_eta(amb_to_patient, p["traffic"])
        if eta < 10.0: reward += 50.0
        elif eta < 20.0: reward += 20.0

        reward -= min(amb_to_patient * 2.0, 40.0) 
        reward -= min(hospital["current_wait"] * 0.5, 30.0) 

        if p["severity"] >= 7 and eta > 15:
            reward -= (p["severity"] - 7) * 5.0

        return round(reward, 2), "success"

    def _update_state(self, hospital_id: int, ambulance_id: int):
        h = self.hospitals[hospital_id]
        a = self.ambulances[ambulance_id]
        h["beds_available"] = max(0, h["beds_available"] - 1)
        if self.patient["severity"] >= 8 and h["icu_available"] > 0:
            h["icu_available"] -= 1
        h["current_wait"] = min(h["current_wait"] + random.randint(0, 3), 120)
        a["status"] = "busy"
        busy = [x for x in self.ambulances if x["status"] == "busy"]
        if len(busy) == self.num_ambulances and busy:
            random.choice(busy)["status"] = "available"

    def _get_info(self) -> Dict[str, Any]:
        info_dict = {
            "step":              self.step_count,
            "patient_severity":  self.patient.get("severity", 0),
            "available_beds":    sum(h["beds_available"] for h in self.hospitals),
            "available_ambs":    sum(1 for a in self.ambulances if a["status"] == "available"),
            "episode_mean_reward": (np.mean(self.episode_rewards) if self.episode_rewards else 0.0),
        }
        if hasattr(self, 'metrics'):
            info_dict.update(self.metrics)
        return info_dict

    def decode_action(self, action: int) -> Tuple[int, int]:
        return action // self.num_ambulances, action % self.num_ambulances

    def encode_action(self, hospital_id: int, ambulance_id: int) -> int:
        return hospital_id * self.num_ambulances + ambulance_id
