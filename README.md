---
title: Smart Routing System
emoji: 🚑
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---
# 🚑 SmartER — AI-Powered Healthcare Routing & Emergency Management

> **Hackathon:** Meta PyTorch OpenEnv Hackathon x SST 2026
> **Framework:** Meta's OpenEnv + PyTorch DQN
> **Tech Stack:** Python · FastAPI · React · Leaflet · Gymnasium

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Folder Structure](#folder-structure)
3. [Quick Start (3 minutes)](#quick-start)
4. [RL Environment Details](#rl-environment)
5. [Backend API](#backend-api)
6. [Frontend Pages](#frontend-pages)
7. [Training the DQN Model](#training)
8. [Simulation & Live Demo](#simulation)
9. [Deployment](#deployment)
10. [OpenEnv Submission](#openenv-submission)

---

## 🧠 Project Overview

SmartER uses **Reinforcement Learning** to solve real-world emergency healthcare routing:

| Problem | RL Solution |
|---------|-------------|
| Which hospital has capacity? | State includes live bed counts |
| Which ambulance is nearest? | State includes all GPS positions |
| Critical patient needs ICU? | Reward function weights ICU access |
| Minimize response time? | Distance + ETA encoded as penalties |

**Core RL Setup (OpenEnv-compatible):**
- **Environment:** `HealthcareRoutingEnv` (Gymnasium API)
- **State:** Patient severity + location + hospital beds + ambulance positions
- **Actions:** Discrete — choose (hospital, ambulance) pair
- **Reward:** +100 admit, +70 ICU for critical, +50 fast response, penalties for bad choices
- **Agent:** Deep Q-Network (DQN) with PyTorch

---

## 📁 Folder Structure

```
healthcare_rl/
├── openenv_env/            ← 🏆 Core OpenEnv submission
│   ├── healthcare_env.py   ← Gymnasium RL environment
│   ├── task.py             ← OpenEnv Task + Grader
│   ├── register.py         ← gym.make() registration
│   └── requirements.txt
│
├── rl/                     ← PyTorch DQN agent
│   ├── dqn_agent.py        ← DQN network + replay buffer
│   ├── train.py            ← Training loop
│   └── models/             ← Saved model weights (created on train)
│
├── backend/                ← FastAPI REST + WebSocket server
│   ├── api/
│   │   └── main.py         ← All endpoints + simulation engine
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/               ← React dashboard
│   ├── src/
│   │   ├── App.jsx
│   │   ├── App.css
│   │   └── pages/
│   │       ├── Dashboard.jsx    ← KPIs + charts
│   │       ├── MapView.jsx      ← Live Leaflet map
│   │       ├── PatientForm.jsx  ← Emergency intake
│   │       └── AdminPanel.jsx   ← Hospital/ambulance management
│   ├── index.html          ← Leaflet CDN loaded here
│   ├── package.json
│   ├── vite.config.js
│   └── Dockerfile
│
├── data/
│   └── sample_data.json    ← Seed data (hospitals, ambulances, patients)
│
├── demo.py                 ← Terminal demo (no server needed)
├── docker-compose.yml
└── README.md
```

---

## ⚡ Quick Start

### Option A — Terminal Demo (no setup needed)

```bash
# 1. Install Python dependencies
pip install gymnasium numpy torch

# 2. Run the demo
python demo.py

# Options:
python demo.py --episodes 5 --steps 20   # longer demo
python demo.py --random                  # compare with random policy
python demo.py --fast                    # no delay between steps
```

### Option B — Full Stack (Backend + Frontend)

**Step 1: Backend**
```bash
cd backend
pip install -r requirements.txt

# Copy RL modules to backend path
export PYTHONPATH=../openenv_env:../rl:$PYTHONPATH

cd api
uvicorn main:app --reload --port 8000
```
> API is now live at http://localhost:8000
> Interactive docs: http://localhost:8000/docs

**Step 2: Frontend**
```bash
cd frontend
cp .env.example .env          # sets VITE_API_URL=http://localhost:8000
npm install
npm run dev
```
> UI is now live at http://localhost:5173

### Option C — Docker (one command)

```bash
docker-compose up --build
```
> Frontend: http://localhost:3000
> Backend:  http://localhost:8000

---

## 🤖 RL Environment

### State Space
The environment builds a flat numpy array with:
- Patient: severity (normalised), latitude, longitude, traffic factor
- Per hospital (×5): bed availability, ICU availability, distance, wait time
- Per ambulance (×4): distance to patient, availability flag

**Total observation size: 4 + 5×4 + 4×2 = 32 floats**

### Action Space
`Discrete(20)` — one action per (hospital, ambulance) combination.
```python
action = hospital_id * num_ambulances + ambulance_id
```

### Reward Function
```
+100  patient admitted
+70   critical patient (severity ≥ 8) assigned ICU bed
+50   ambulance ETA < 10 minutes
+20   ambulance ETA < 20 minutes
-100  no bed available at hospital
-50   ambulance is busy
- distance_penalty (up to -40)
- wait_time_penalty (up to -30)
- severity_delay_penalty (critical patient + high ETA)
```

### Usage
```python
from openenv_env.healthcare_env import HealthcareRoutingEnv

env = HealthcareRoutingEnv(render_mode="human")
obs, info = env.reset()

for _ in range(100):
    action = env.get_greedy_action()      # or your RL policy
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
```

---

## 🚀 Training the DQN Model

```bash
cd rl

# Quick training (300 episodes)
python train.py

# Production training
python train.py --episodes 1000 --steps 150 --lr 0.0005

# With console rendering
python train.py --episodes 100 --render
```

The trained model is saved to `rl/models/dqn_healthcare.pth`.

**Once trained**, the backend API will automatically load it and use DQN instead of the greedy fallback.

---

## 🌐 Backend API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET  | `/` | Health check |
| POST | `/add_patient` | Register patient + get RL assignment |
| GET  | `/get_hospitals` | Live hospital bed counts |
| GET  | `/get_ambulances` | Live ambulance GPS positions |
| POST | `/assign_resources` | RL assignment without persisting |
| PUT  | `/update_ambulance_location` | Update GPS position |
| PUT  | `/update_hospital_beds` | Update bed availability |
| GET  | `/get_live_tracking` | Full snapshot (map view) |
| GET  | `/stats` | Dashboard statistics |
| GET  | `/simulation/start` | Start auto simulation |
| GET  | `/simulation/stop` | Stop simulation |
| WS   | `/ws/live` | Real-time WebSocket push |

**Example — Add a patient:**
```bash
curl -X POST http://localhost:8000/add_patient \
  -H "Content-Type: application/json" \
  -d '{"name":"Ravi Kumar","severity":9.2,"lat":12.972,"lon":77.601}'
```

---

## 🗺️ Frontend Pages

| Page | Description |
|------|-------------|
| **Dashboard** | KPI cards, bed occupancy bar chart, ambulance status pie, recent patients, simulation toggle |
| **Live Map** | Leaflet map with moving 🚑 ambulances, 🏥 hospitals (color-coded by occupancy), 🤕 patients. WebSocket updates every 2 seconds |
| **New Emergency** | Patient intake form with severity slider → instant RL dispatch result with reasoning |
| **Admin Panel** | Edit hospital beds/ICU/wait times inline, view ambulance statuses, full assignment history |

---

## 🎮 Simulation & Live Demo

1. Open the Dashboard tab
2. Click **▶️ Start Simulation**
3. Switch to **Live Map** — watch ambulances move in real-time
4. New simulated patients appear every ~6 seconds
5. Each patient is automatically assigned by the RL agent

For judges: the simulation creates realistic emergency scenarios with varying severity levels and traffic conditions to showcase the RL decision-making.

# 🌟 Core Features

### 🧠 AI-Powered Routing (PyTorch DQN)
The system's dispatch engine is powered by a custom-trained **Deep Q-Network (DQN)** built with PyTorch. 
- **Smart Dispatch:** Instead of using basic distance-based greedy algorithms, the AI evaluates patient severity, hospital bed/ICU availability, and real-time distance to make optimal routing decisions.
- **Seamless Integration:** The FastAPI backend automatically detects and loads the trained model weights (`rl/models/dqn_healthcare.pth`) at startup.

### 🏙️ 3D Cyberpunk Radar (Three.js)
The live tracking map features a fully interactive, real-time 3D cityscape of Solapur built with **Three.js**.
- **Dynamic Visuals:** Procedurally generated hospital buildings, pulsing patient spheres, and hovering ambulances with directional aiming.
- **Two-Phase Routing:** Visualizes the complete emergency lifecycle in real-time—blue dotted paths when en-route to the patient, and green paths during hospital transport.
- **Interactive:** Full drag, rotate, and zoom capabilities to monitor the dispatch grid.

---

## ☁️ Deployment

### Frontend → Vercel
```bash
cd frontend
npm run build
# Deploy dist/ folder to Vercel
vercel deploy
```

### Backend → Render
1. Create new Web Service on Render
2. Set build command: `pip install -r backend/requirements.txt`
3. Set start command: `uvicorn backend.api.main:app --host 0.0.0.0 --port $PORT`
4. Set env var: `PYTHONPATH=./openenv_env:./rl`

---

## 🏆 OpenEnv Submission

The `openenv_env/` folder contains the complete OpenEnv-spec submission:

```python
# Verify submission
cd openenv_env
python register.py         # test gym.make() works
python task.py             # run grader on random + greedy policies
```

**Grading criteria (100 pts total):**
- 40 pts — Mean episode reward
- 30 pts — Success rate (% assignments that succeed)
- 20 pts — ICU assignment rate for critical patients
- 10 pts — Ambulance utilisation

**Expected scores:**
- Random policy:  ~25–35 pts
- Greedy policy:  ~55–65 pts
- Trained DQN:    ~75–90 pts

---

## 🧪 Running Tests

```bash
# Test RL environment
python -c "
from openenv_env.healthcare_env import HealthcareRoutingEnv
env = HealthcareRoutingEnv()
obs, _ = env.reset()
print('Obs shape:', obs.shape)
obs, r, term, trunc, info = env.step(0)
print('Reward:', r, '| Info:', info)
print('✅ Environment OK')
"

# Test grader
cd openenv_env && python task.py

# Test backend (requires uvicorn running)
curl http://localhost:8000/stats
```

---

## 👥 Team & Acknowledgements

Built for **Meta PyTorch OpenEnv Hackathon x SST 2026**.

Team Members:
1. Rohan Belsare
2. Rajat Sahuji
3. Sanchita Patil

Tech Stack:
- [Meta OpenEnv](https://github.com/meta-pytorch/OpenEnv) — environment spec
- [HuggingFace openenv-course](https://github.com/huggingface/openenv-course) — reference implementation
- [PyTorch](https://pytorch.org) — DQN training
- [Gymnasium](https://gymnasium.farama.org) — RL API
- [FastAPI](https://fastapi.tiangolo.com) — backend
- [React](https://react.dev) + [Leaflet](https://leafletjs.com) — frontend