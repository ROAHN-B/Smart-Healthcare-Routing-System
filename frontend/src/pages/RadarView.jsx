import { useEffect, useRef, useCallback } from "react";

const API = import.meta.env.PROD ? window.location.origin : "http://localhost:8000";

const LAT_MIN = 12.85, LAT_MAX = 13.10;
const LON_MIN = 77.45, LON_MAX = 77.75;

export default function RadarView({ isActive }) {
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const dataRef = useRef({ hospitals: [], ambulances: [], patients: [] });
  const animFrameRef = useRef(null);
  
  // Track sequential patient numbers so we get "PATIENT-1", "PATIENT-2"
  const patientCounters = useRef({});
  const nextPatientNum = useRef(1);

  const project = (lat, lon, width, height) => {
    const padding = 50;
    const innerW = width - padding * 2;
    const innerH = height - padding * 2;
    const x = ((lon - LON_MIN) / (LON_MAX - LON_MIN)) * innerW + padding;
    const y = height - (((lat - LAT_MIN) / (LAT_MAX - LAT_MIN)) * innerH + padding);
    return { x, y };
  };

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const { width, height } = canvas;

    // Background
    ctx.fillStyle = "#0a0a0a"; 
    ctx.fillRect(0, 0, width, height);

    // Cyberpunk Grid
    ctx.strokeStyle = "#1a1a2e";
    ctx.lineWidth = 1;
    const gridSize = 50;
    for (let i = 0; i < width; i += gridSize) {
      ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i, height); ctx.stroke();
    }
    for (let i = 0; i < height; i += gridSize) {
      ctx.beginPath(); ctx.moveTo(0, i); ctx.lineTo(width, i); ctx.stroke();
    }

    const data = dataRef.current;

    // ── DRAW HOSPITALS ──────────────────────────────────────
    (data.hospitals || []).forEach((h, index) => {
      const { x, y } = project(h.lat, h.lon, width, height);
      const isFull = h.beds_available === 0;
      
      const boxSize = 24;
      ctx.fillStyle = isFull ? "#ef4444" : "#22c55e"; 
      ctx.shadowBlur = 15;
      ctx.shadowColor = ctx.fillStyle;
      
      // Draw outer box
      ctx.fillRect(x - boxSize/2, y - boxSize/2, boxSize, boxSize);
      
      // Draw inner medical cross
      ctx.fillStyle = "#000000";
      ctx.fillRect(x - 2, y - 8, 4, 16); // Vertical line
      ctx.fillRect(x - 8, y - 2, 16, 4); // Horizontal line

      ctx.shadowBlur = 0;
      ctx.fillStyle = "#ffffff";
      ctx.font = "bold 11px var(--mono-font, monospace)";
      ctx.fillText(`HOSPITAL-${index + 1} [${h.beds_available} BEDS]`, x + 18, y + 4);
    });

    // ── DRAW PATIENTS ───────────────────────────────────────
    const time = Date.now() / 150; 
    const pulseRadius = Math.abs(Math.sin(time)) * 6 + 6; 
    
    (data.patients || []).forEach(p => {
      if (p.status === "admitted") return;
      
      // Assign sequential ID if new
      if (!patientCounters.current[p.id]) {
        patientCounters.current[p.id] = nextPatientNum.current++;
      }
      const ptLabel = `PATIENT-${patientCounters.current[p.id]}`;
      
      const { x, y } = project(p.lat, p.lon, width, height);
      const isPickedUp = p.status === "picked_up";
      
      ctx.beginPath();
      ctx.arc(x, y, isPickedUp ? 6 : pulseRadius, 0, Math.PI * 2);
      ctx.fillStyle = isPickedUp ? "#eab308" : "#ef4444"; 
      ctx.shadowBlur = 20;
      ctx.shadowColor = ctx.fillStyle;
      ctx.fill();

      // If waiting, draw a tiny exclamation mark inside
      if (!isPickedUp) {
        ctx.fillStyle = "#ffffff";
        ctx.font = "bold 10px Arial";
        ctx.textAlign = "center";
        ctx.fillText("!", x, y + 3);
        ctx.textAlign = "left"; // reset
      }

      ctx.shadowBlur = 0;
      ctx.fillStyle = "#ffffff";
      ctx.font = "10px var(--mono-font, monospace)";
      ctx.fillText(ptLabel, x + 14, y + 4);
    });

    // ── DRAW AMBULANCES ─────────────────────────────────────
    (data.ambulances || []).forEach((a, index) => {
      const { x, y } = project(a.lat, a.lon, width, height);
      const isMoving = a.status === "en_route_to_patient" || a.status === "en_route_to_hospital";
      
      let destLabel = "";

      if (isMoving && a.assigned_patient) {
        const pt = data.patients.find(p => p.id === a.assigned_patient);
        if (pt) {
          let targetPos = null;
          let lineColor = "#3b82f6"; // Blue path to patient

          if (a.status === "en_route_to_patient") {
            targetPos = project(pt.lat, pt.lon, width, height);
            destLabel = `[TO: PT-${patientCounters.current[pt.id]}]`;
          } else if (a.status === "en_route_to_hospital" && a.target_hosp_id) {
            const hospIndex = data.hospitals.findIndex(h => h.id === a.target_hosp_id);
            const hosp = data.hospitals[hospIndex];
            if (hosp) {
              targetPos = project(hosp.lat, hosp.lon, width, height);
              lineColor = "#22c55e"; // Green path to hospital
              destLabel = `[DEST: HOSP-${hospIndex + 1}]`;
            }
          }

          // Draw the dotted path
          if (targetPos) {
            ctx.beginPath();
            ctx.moveTo(x, y);
            ctx.lineTo(targetPos.x, targetPos.y);
            ctx.strokeStyle = lineColor;
            ctx.lineWidth = 2;
            ctx.setLineDash([5, 5]);
            ctx.stroke();
            ctx.setLineDash([]);
          }
        }
      }

      ctx.fillStyle = isMoving ? "#3b82f6" : "#64748b"; 
      ctx.shadowBlur = isMoving ? 15 : 0;
      ctx.shadowColor = ctx.fillStyle;
      
      // Draw Ambulance as a sturdy triangle
      ctx.beginPath();
      ctx.moveTo(x, y - 12);
      ctx.lineTo(x + 10, y + 8);
      ctx.lineTo(x - 10, y + 8);
      ctx.closePath();
      ctx.fill();

      // Ambulance Label
      ctx.shadowBlur = 0;
      ctx.fillStyle = "#ffffff";
      ctx.font = "10px var(--mono-font, monospace)";
      const ambLabel = `AMB-${index + 1} ${destLabel}`;
      ctx.fillText(ambLabel, x + 14, y + 4);
    });

    animFrameRef.current = requestAnimationFrame(draw);
  }, []);

  useEffect(() => {
    if (!isActive) return;
    const resize = () => {
      if (canvasRef.current && canvasRef.current.parentElement) {
        canvasRef.current.width = canvasRef.current.parentElement.clientWidth;
        canvasRef.current.height = canvasRef.current.parentElement.clientHeight;
      }
    };
    resize();
    window.addEventListener("resize", resize);
    return () => window.removeEventListener("resize", resize);
  }, [isActive]);

  useEffect(() => {
    const connectWs = () => {
      try {
        const wsUrl = API.replace(/^http/, "ws") + "/ws/live";
        const ws = new WebSocket(wsUrl);
        ws.onmessage = (e) => {
          try {
            dataRef.current = JSON.parse(e.data);
          } catch (err) {}
        };
        ws.onclose = () => setTimeout(connectWs, 2000);
        wsRef.current = ws;
      } catch (err) {}
    };
    
    connectWs();
    animFrameRef.current = requestAnimationFrame(draw);

    return () => {
      if (wsRef.current) wsRef.current.close();
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    };
  }, [draw]);

  return (
    <div className="map-view-wrapper" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div className="map-topbar" style={{ padding: '10px 20px', borderBottom: '1px solid #333' }}>
        <span style={{color:"var(--accent)", fontFamily: 'var(--mono-font)'}}>SYSTEM.RADAR // REALTIME_DQN_TRACKING</span>
      </div>
      
      <div className="map-body" style={{ flex: 1, position: "relative", overflow: 'hidden' }}>
        <canvas ref={canvasRef} style={{ display: "block", width: "100%", height: "100%" }} />
        
        {/* ── SYSTEM LEGEND OVERLAY ── */}
        <div style={{
          position: 'absolute',
          top: '20px',
          right: '20px',
          backgroundColor: 'rgba(0, 0, 0, 0.8)',
          border: '1px solid var(--border-2, #333)',
          padding: '16px',
          borderRadius: '8px',
          fontFamily: 'var(--mono-font, monospace)',
          fontSize: '11px',
          color: 'var(--text, #ddd)',
          display: 'flex',
          flexDirection: 'column',
          gap: '12px',
          pointerEvents: 'none',
          boxShadow: '0 4px 12px rgba(0,0,0,0.5)',
          zIndex: 10
        }}>
          <div style={{ color: 'var(--text-3, #888)', fontWeight: 'bold', borderBottom: '1px solid var(--border-2, #333)', paddingBottom: '8px', marginBottom: '4px' }}>
            SYSTEM.LEGEND
          </div>
          
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div style={{ width: '12px', height: '12px', borderRadius: '50%', background: '#ef4444', boxShadow: '0 0 10px #ef4444', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <span style={{ fontSize: '9px', color: '#fff', fontWeight: 'bold' }}>!</span>
            </div>
            <span>AWAITING RESCUE</span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div style={{ width: '12px', height: '12px', borderRadius: '50%', background: '#eab308', boxShadow: '0 0 10px #eab308' }}></div>
            <span>SECURED / IN-TRANSIT</span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div style={{ width: '0', height: '0', borderLeft: '7px solid transparent', borderRight: '7px solid transparent', borderBottom: '12px solid #3b82f6', filter: 'drop-shadow(0 0 6px #3b82f6)' }}></div>
            <span>ACTIVE AMBULANCE</span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div style={{ width: '0', height: '0', borderLeft: '7px solid transparent', borderRight: '7px solid transparent', borderBottom: '12px solid #64748b' }}></div>
            <span>IDLE AMBULANCE</span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div style={{ width: '14px', height: '14px', background: '#22c55e', boxShadow: '0 0 10px #22c55e', position: 'relative', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <div style={{ position: 'absolute', width: '2px', height: '8px', background: '#000' }}></div>
              <div style={{ position: 'absolute', width: '8px', height: '2px', background: '#000' }}></div>
            </div>
            <span>HOSPITAL (OPEN)</span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div style={{ width: '14px', height: '14px', background: '#ef4444', boxShadow: '0 0 10px #ef4444', position: 'relative', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <div style={{ position: 'absolute', width: '2px', height: '8px', background: '#000' }}></div>
              <div style={{ position: 'absolute', width: '8px', height: '2px', background: '#000' }}></div>
            </div>
            <span>HOSPITAL (FULL)</span>
          </div>
        </div>
      </div>
    </div>
  );
}