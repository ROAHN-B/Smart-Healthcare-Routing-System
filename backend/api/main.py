"""
main.py  –  FastAPI Backend (Strict Grid Snapping & MySQL Integrated)
=====================================================================
"""

import asyncio
import json
import math
import os
import random
import sys
import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

import mysql.connector

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BACKEND_DIR, "..", "openenv_env"))
sys.path.insert(0, os.path.join(BACKEND_DIR, "..", "rl"))

from healthcare_env import HealthcareRoutingEnv, haversine_distance, compute_eta

# ── RL GLOBAL BOUNDS ──
LAT_MIN, LAT_MAX = 17.60, 17.75
LON_MIN, LON_MAX = 75.85, 76.00

# ── GRID SNAPPING LOGIC (Forces ambulances onto roads) ──
def snap_val(val, min_v, max_v, steps=20):
    """Snaps any floating coordinate exactly to the 20x20 road grid."""
    step = (max_v - min_v) / steps
    idx = round((val - min_v) / step)
    return min_v + idx * step

def snap_lat(lat): return snap_val(lat, LAT_MIN, LAT_MAX, 20)
def snap_lon(lon): return snap_val(lon, LON_MIN, LON_MAX, 20)


# ── IN-MEMORY DATABASE (Snapped to Intersections) ──
HOSPITALS: List[Dict] = [
    {"id": "h0", "name": "Ashwini Hospital",         "lat": snap_lat(17.7300), "lon": snap_lon(75.9700), "total_beds": 100, "beds_available": 72, "icu_beds": 20, "icu_available": 14, "wait_time": 10, "address": "North-East Sector"},
    {"id": "h1", "name": "Markandey Hospital",       "lat": snap_lat(17.7200), "lon": snap_lon(75.8800), "total_beds": 80,  "beds_available": 55, "icu_beds": 15, "icu_available": 9,  "wait_time": 15, "address": "North-West Sector"},
    {"id": "h2", "name": "Yashodhara Hospital",      "lat": snap_lat(17.6800), "lon": snap_lon(75.9300), "total_beds": 60,  "beds_available": 48, "icu_beds": 10, "icu_available": 7,  "wait_time": 5,  "address": "Central Sector"},
    {"id": "h3", "name": "Monark Hospital",          "lat": snap_lat(17.6200), "lon": snap_lon(75.8700), "total_beds": 120, "beds_available": 91, "icu_beds": 30, "icu_available": 22, "wait_time": 20, "address": "South-West Sector"},
    {"id": "h4", "name": "Civil Hospital",           "lat": snap_lat(17.6300), "lon": snap_lon(75.9800), "total_beds": 90,  "beds_available": 63, "icu_beds": 25, "icu_available": 18, "wait_time": 8,  "address": "South-East Sector"},
]

AMBULANCES: List[Dict] = [
    {"id": "a0", "name": "AMB-001", "lat": snap_lat(17.7000), "lon": snap_lon(75.9000), "status": "available", "assigned_patient": None, "target_hosp_id": None, "speed_kmh": 60},
    {"id": "a1", "name": "AMB-002", "lat": snap_lat(17.6500), "lon": snap_lon(75.9500), "status": "available", "assigned_patient": None, "target_hosp_id": None, "speed_kmh": 65},
    {"id": "a2", "name": "AMB-003", "lat": snap_lat(17.7200), "lon": snap_lon(75.9600), "status": "available", "assigned_patient": None, "target_hosp_id": None, "speed_kmh": 55},
    {"id": "a3", "name": "AMB-004", "lat": snap_lat(17.6400), "lon": snap_lon(75.8900), "status": "available", "assigned_patient": None, "target_hosp_id": None, "speed_kmh": 70},
]

PATIENTS: Dict[str, Dict] = {}
ASSIGNMENTS: List[Dict]   = []


# ── RL MODEL LOADER ──
rl_env: Optional[HealthcareRoutingEnv] = None
try:
    import torch
    from dqn_agent import DQNAgent
    _model_path = os.path.abspath(os.path.join(BACKEND_DIR, "..", "..", "rl", "models", "dqn_healthcare.pth"))
    dqn_agent: Optional[DQNAgent] = None
    if os.path.exists(_model_path):
        _tmp_env = HealthcareRoutingEnv()
        _tmp_obs, _ = _tmp_env.reset()
        dqn_agent = DQNAgent(obs_size=_tmp_env.observation_space.shape[0], action_size=_tmp_env.action_space.n)
        dqn_agent.load(_model_path)
        print("[Backend] DQN model loaded ✓")
    else:
        print(f"[Backend] Model not found at {_model_path} – using greedy fallback")
except Exception as e:
    dqn_agent = None
    print(f"[Backend] DQN unavailable ({e}) – using greedy fallback")

# ── MYSQL HELPER FUNCTIONS ──
def save_dispatch_to_db(patient_data: dict, assignment_data: dict):
    try:
        db = mysql.connector.connect(
            host=os.getenv("DB_HOST", "localhost"),
            user=os.getenv("DB_USER", "root"),
            password=os.getenv("DB_PASS", "your_actual_password"), # UPDATE THIS
            database=os.getenv("DB_NAME", "smart_healthcare_db"),
            port=3306
        )
        cursor = db.cursor()

        sql_patient = """
            INSERT INTO recent_dispatches (patient_id, severity, status, latitude, longitude)
            VALUES (%s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE status=VALUES(status)
        """
        cursor.execute(sql_patient, (
            patient_data["id"], patient_data["severity_label"], patient_data["status"],
            patient_data["lat"], patient_data["lon"]
        ))

        sql_routing = """
            INSERT INTO simulation_routing (patient_id, ambulance_id, hospital_id, distance_km, routing_score)
            VALUES (%s, %s, %s, %s, %s)
        """
        routing_score = 0.95 if assignment_data.get("model_used") == "DQN" else 0.50
        cursor.execute(sql_routing, (
            patient_data["id"], assignment_data["ambulance"]["id"], assignment_data["hospital"]["id"],
            assignment_data["dist_to_hospital"], routing_score
        ))

        db.commit()
    except Exception as e:
        print(f"[DB Error] Could not save to MySQL: {e}")
    finally:
        if 'db' in locals() and db.is_connected(): cursor.close(); db.close()

# ── FASTAPI SETUP ──
simulation_running = False
simulation_task    = None
ws_connections: List[WebSocket] = []

class PatientIn(BaseModel):
    name:           str           = Field(default="Unknown Patient")
    severity:       float         = Field(..., ge=1, le=10)
    lat:            float         = Field(...)
    lon:            float         = Field(...)
    emergency_type: str           = Field(default="General")
    notes:          Optional[str] = None

def save_manual_patient_to_db(patient_in: PatientIn, generated_id: str):
    if patient_in.name.startswith("Sim-"): return
    try:
        db = mysql.connector.connect(
            host=os.getenv("DB_HOST", "localhost"), user=os.getenv("DB_USER", "root"),
            password=os.getenv("DB_PASS", "Rohan@54321"), # UPDATE THIS
            database=os.getenv("DB_NAME", "smart_healthcare_db"), port=3306
        )
        cursor = db.cursor()
        sql = """
            INSERT INTO manual_patients (patient_id, patient_name, severity, latitude, longitude, emergency_type, notes)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(sql, (generated_id, patient_in.name, patient_in.severity, patient_in.lat, patient_in.lon, patient_in.emergency_type, patient_in.notes))
        db.commit()
    except Exception as e: print(f"\n❌ FATAL DATABASE ERROR: {e}\n")
    finally:
        if 'db' in locals() and db.is_connected(): cursor.close(); db.close()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Backend] Healthcare Routing API started")
    yield
    print("[Backend] Shutting down")

app = FastAPI(title="SmartER API", lifespan=lifespan)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

def rl_assign(patient: Dict) -> Dict:
    env = HealthcareRoutingEnv(hospitals=HOSPITALS, ambulances=AMBULANCES)
    env.LAT_MIN, env.LAT_MAX = LAT_MIN, LAT_MAX
    env.LON_MIN, env.LON_MAX = LON_MIN, LON_MAX
    env.hospitals  = [{**h, "current_wait": h["wait_time"]} for h in HOSPITALS]
    env.ambulances = [{**a} for a in AMBULANCES]
    env.patient = {"severity": patient["severity"], "lat": patient["lat"], "lon": patient["lon"], "traffic": random.uniform(0.9, 1.8)}

    if dqn_agent:
        obs = env._get_observation()
        action = dqn_agent.greedy_action(obs)
        hosp_idx, amb_idx = env.decode_action(action)
    else:
        amb_idx = 0
        min_d = float('inf')
        for i, a in enumerate(AMBULANCES):
            if a["status"] == "available":
                d = haversine_distance(patient["lat"], patient["lon"], a["lat"], a["lon"])
                if d < min_d: min_d = d; amb_idx = i
        hosp_idx = 0
        max_score = -float('inf')
        for i, h in enumerate(HOSPITALS):
            score = h["beds_available"] * 10 - haversine_distance(patient["lat"], patient["lon"], h["lat"], h["lon"])
            if score > max_score: max_score = score; hosp_idx = i

    hospital  = HOSPITALS[min(hosp_idx, len(HOSPITALS) - 1)]
    ambulance = AMBULANCES[min(amb_idx, len(AMBULANCES) - 1)]

    dist_amb_pt = haversine_distance(ambulance["lat"], ambulance["lon"], patient["lat"], patient["lon"])
    dist_pt_hosp = haversine_distance(patient["lat"], patient["lon"], hospital["lat"], hospital["lon"])
    total_eta = round(compute_eta(dist_amb_pt + dist_pt_hosp, 1.2, 60), 1)

    return {
        "hospital": hospital, "ambulance": ambulance, "eta_minutes": total_eta,
        "dist_to_hospital": round(dist_pt_hosp, 2),
        "reasoning": [f"✅ {hospital['beds_available']} beds free", f"📍 Ambulance {ambulance['name']} dispatched"],
        "model_used": "DQN" if dqn_agent else "Greedy",
    }

@app.post("/add_patient")
def add_patient(patient_in: PatientIn):
    patient_id = f"PT-{str(uuid.uuid4())[:6].upper()}" 
    
    # Snap incoming coordinates to the grid so ambulances never leave the road!
    patient_in.lat = snap_lat(patient_in.lat)
    patient_in.lon = snap_lon(patient_in.lon)
    
    save_manual_patient_to_db(patient_in, patient_id)

    patient = {
        **patient_in.dict(), "id": patient_id, "status": "pending", "timestamp": time.time(),
        "severity_label": "critical" if patient_in.severity >= 8 else "moderate" if patient_in.severity >= 5 else "mild"
    }
    PATIENTS[patient_id] = patient
    assignment = rl_assign(patient)

    for h in HOSPITALS:
        if h["id"] == assignment["hospital"]["id"]:
            h["beds_available"] = max(0, h["beds_available"] - 1)
            break
    for a in AMBULANCES:
        if a["id"] == assignment["ambulance"]["id"]:
            a["status"] = "en_route_to_patient"
            a["assigned_patient"] = patient_id
            a["target_hosp_id"] = assignment["hospital"]["id"]
            break

    patient["status"] = "assigned"
    ASSIGNMENTS.append({"patient_id": patient_id, "hospital_id": h["id"], "ambulance_id": a["id"]})
    save_dispatch_to_db(patient, assignment)
    return {"patient": patient, "assignment": assignment}

@app.get("/get_live_tracking")
def get_live_tracking(): return {"hospitals": HOSPITALS, "ambulances": AMBULANCES, "patients": list(PATIENTS.values())}

@app.get("/stats")
def get_stats():
    used_beds = sum(h["total_beds"] - h["beds_available"] for h in HOSPITALS)
    total_beds = sum(h["total_beds"] for h in HOSPITALS)
    return {
        "total_patients": len(PATIENTS),
        "bed_occupancy_pct": round(used_beds / total_beds * 100, 1) if total_beds else 0,
        "available_ambs": sum(1 for a in AMBULANCES if a["status"] == "available"),
        "total_assignments": len(ASSIGNMENTS),
        "hospitals": [{"name": h["name"], "occupancy_pct": round((h["total_beds"] - h["beds_available"]) / h["total_beds"] * 100, 1), "icu_occupancy": max(0, int(((h["total_beds"] - h["beds_available"]) / h["total_beds"]) * h["icu_beds"]))} for h in HOSPITALS]
    }

async def simulation_loop():
    global simulation_running
    while simulation_running:
        for a in AMBULANCES:
            if a["status"] in ["en_route_to_patient", "en_route_to_hospital"]:
                pid = a["assigned_patient"]
                target = PATIENTS.get(pid) if a["status"] == "en_route_to_patient" else next((h for h in HOSPITALS if h["id"] == a["target_hosp_id"]), None)
                if not target: continue
                
                dlat = target["lat"] - a["lat"]
                dlon = target["lon"] - a["lon"]
                
                # STRICT GRID ALIGNMENT: Prevents diagonal cutting through buildings
                step = 0.002 
                if abs(dlon) > step:
                    a["lon"] += math.copysign(step, dlon)
                elif abs(dlat) > step:
                    a["lon"] = target["lon"] # Snap perfectly to the vertical road
                    a["lat"] += math.copysign(step, dlat)
                else:
                    a["lat"] = target["lat"]
                    a["lon"] = target["lon"]
                    if a["status"] == "en_route_to_patient":
                        a["status"] = "en_route_to_hospital"
                        PATIENTS[pid]["status"] = "picked_up"
                    else:
                        a["status"] = "available"
                        a["assigned_patient"] = None
                        PATIENTS[pid]["status"] = "admitted"

        free_ambs = [a for a in AMBULANCES if a["status"] == "available"]
        if free_ambs and random.random() < 0.075: 
            # Snap generated patients perfectly to the grid roads!
            fake = PatientIn(name=f"Sim-{random.randint(100,999)}", severity=random.uniform(1,10), lat=snap_lat(random.uniform(LAT_MIN, LAT_MAX)), lon=snap_lon(random.uniform(LON_MIN, LON_MAX)))
            try: add_patient(fake)
            except: pass

        for h in HOSPITALS:
            if random.random() < 0.005 and h["beds_available"] < h["total_beds"]: h["beds_available"] += 1

        payload = json.dumps({"ambulances": AMBULANCES, "hospitals": HOSPITALS, "patients": list(PATIENTS.values())})
        
        dead_sockets = []
        for ws in ws_connections:
            try: await ws.send_text(payload)
            except: dead_sockets.append(ws)
        for dead in dead_sockets: ws_connections.remove(dead)
        await asyncio.sleep(0.1)

@app.get("/simulation/start")
async def start_sim():
    global simulation_running, simulation_task
    if not simulation_running:
        simulation_running = True
        simulation_task = asyncio.create_task(simulation_loop())
    return {"message": "Simulation started"}

@app.get("/simulation/stop")
async def stop_sim():
    global simulation_running
    simulation_running = False
    return {"message": "Simulation stopped"}

@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    ws_connections.append(websocket)
    try:
        while True: await websocket.receive_text() 
    except WebSocketDisconnect:
        if websocket in ws_connections: ws_connections.remove(websocket)
    except Exception:
        if websocket in ws_connections: ws_connections.remove(websocket)

@app.post("/reset")
def env_reset():
    global PATIENTS, ASSIGNMENTS
    PATIENTS.clear(); ASSIGNMENTS.clear()
    for h in HOSPITALS: h["beds_available"] = h["total_beds"] // 2
    for a in AMBULANCES: a["status"] = "available"
    return {"message": "Environment reset", "status": "success"}

@app.post("/step")
def env_step(action: dict): return {"observation": get_live_tracking(), "reward": 0.0, "done": False}

@app.get("/state")
def env_state(): return get_live_tracking()

static_path = os.path.join(os.getcwd(), "static")
if os.path.exists(static_path):
    app.mount("/assets", StaticFiles(directory=os.path.join(static_path, "assets")), name="assets")
    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        local_file = os.path.join(static_path, full_path)
        if full_path != "" and os.path.exists(local_file): return FileResponse(local_file)
        return FileResponse(os.path.join(static_path, "index.html"))