import { useState, useCallback } from "react";
import Dashboard from "./pages/Dashboard";
import RadarView3D from "./pages/RadarView3D"; 
import PatientForm from "./pages/PatientForm";
import AdminPanel from "./pages/AdminPanel";
import "./App.css";

const API = import.meta.env.PROD ? window.location.origin : "http://localhost:8000";

const TABS = [
  { id: "dashboard", label: "Overview" },
  { id: "map",       label: "Live simulation" },
  { id: "patient",   label: "Dispatch" },
  { id: "admin",     label: "System"   },
];

export default function App() {
  const [activeTab, setActiveTab] = useState("dashboard");
  const [simRunning, setSimRunning] = useState(false);

  // Toggle the simulation engine on the backend
  const toggleSim = useCallback(async () => {
    try { 
      await fetch(`${API}/simulation/${simRunning ? "stop" : "start"}`); 
    } catch (e) {
      console.warn("Backend unreached, toggling local sim state.", e);
    }
    setSimRunning(s => !s);
  }, [simRunning]);

  return (
    <div className="os-root dot-bg">
      {/* ── TOP MENU BAR ── */}
      <div className="menubar">
        <div className="menubar-left">
          <span className="app-name">Smart Routing System</span>
        </div>
        <div className="menubar-right">
          <button 
            onClick={toggleSim} 
            style={{
              background: simRunning ? 'var(--accent)' : 'transparent', 
              color: simRunning ? '#ffffff' : 'var(--text)', 
              border: `2px solid ${simRunning ? 'var(--accent)' : 'var(--text)'}`,
              fontFamily: 'var(--mono-font)', 
              textTransform: 'uppercase', 
              padding: '8px 20px', 
              borderRadius: '40px', 
              cursor: 'pointer',
              fontWeight: 'bold',
              transition: 'all 0.3s ease'
            }}>
            {simRunning ? "[ STOP ENGINE ]" : "[ IGNITE ENGINE ]"}
          </button>
          <div className="status-pill">
            <span className="pulse-dot" style={{background: simRunning ? "var(--accent)" : "var(--text)"}} />
            {simRunning ? "SYS.ACTIVE" : "SYS.IDLE"}
          </div>
        </div>
      </div>

      {/* ── MAIN WORKSPACE ── */}
      <div className="window-frame">
        {/* SIDEBAR NAVIGATION */}
        <div className="sidebar">
          <nav className="sidebar-nav">
            {TABS.map(tab => (
              <button 
                key={tab.id} 
                className={`sidebar-item ${activeTab === tab.id ? "active" : ""}`} 
                onClick={() => setActiveTab(tab.id)}
              >
                {tab.label}
              </button>
            ))}
          </nav>
        </div>

        {/* CONTENT AREA */}
        <div className="content-area">
          <div className="content-header">
            <div className="ch-breadcrumb">
              PATH <span className="ch-page">/{TABS.find(t => t.id === activeTab)?.label}</span>
            </div>
          </div>
          <div className="content-body">
            
            <div style={{ display: activeTab === "dashboard" ? "block" : "none" }}>
              <Dashboard simRunning={simRunning} onToggleSim={toggleSim} />
            </div>
            
            {/* The new 3D Interactive Map */}
            <div style={{ display: activeTab === "map" ? "block" : "none", height: "100%" }}>
              <RadarView3D isActive={activeTab === "map"} />
            </div>
            
            <div style={{ display: activeTab === "patient" ? "block" : "none" }}>
              <PatientForm />
            </div>
            
            <div style={{ display: activeTab === "admin" ? "block" : "none" }}>
              <AdminPanel />
            </div>

          </div>
        </div>
      </div>
    </div>
  );
}