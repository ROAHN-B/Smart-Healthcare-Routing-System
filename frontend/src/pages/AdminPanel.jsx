import { useState, useEffect } from "react";

const API = import.meta.env.PROD ? window.location.origin : "http://localhost:8000";

export default function AdminPanel() {
  const [data, setData] = useState({ hospitals: [], ambulances: [] });

  const fetchData = async () => {
    try {
      const res = await fetch(`${API}/get_live_tracking`);
      const json = await res.json();
      setData(json);
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => {
    fetchData();
    const iv = setInterval(fetchData, 3000);
    return () => clearInterval(iv);
  }, []);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "32px" }}>
      <div>
        <div className="page-title">SYSTEM.OVERRIDE</div>
        <div className="page-sub">MANUAL CONTROLS & RESOURCE MANAGEMENT</div>
      </div>

      <div className="section-card">
        <div className="section-title" style={{ marginBottom: "20px" }}>HOSPITAL_DB</div>
        <table className="os-table">
          <thead>
            <tr>
              <th>ID</th>
              <th>NAME</th>
              <th>GENERAL BEDS</th>
              <th>ICU BEDS</th>
            </tr>
          </thead>
          <tbody>
            {data.hospitals.map(h => (
              <tr key={h.id}>
                <td style={{ color: "var(--text-3)" }}>{h.id}</td>
                <td style={{ fontWeight: "bold" }}>{h.name}</td>
                <td>
                  <span style={{ 
                    fontFamily: "var(--dot-font)", fontSize: "18px", 
                    color: h.beds_available < 10 ? "var(--accent)" : "inherit" 
                  }}>
                    {h.beds_available} / {h.total_beds}
                  </span>
                </td>
                <td>
                  <span style={{ 
                    fontFamily: "var(--dot-font)", fontSize: "18px", 
                    color: h.icu_available < 3 ? "var(--accent)" : "inherit" 
                  }}>
                    {h.icu_available} / {h.icu_beds}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="section-card">
        <div className="section-title" style={{ marginBottom: "20px" }}>FLEET_DB</div>
        <table className="os-table">
          <thead>
            <tr>
              <th>ID</th>
              <th>CALLSIGN</th>
              <th>STATUS</th>
              <th>GPS LOC</th>
            </tr>
          </thead>
          <tbody>
            {data.ambulances.map(a => (
              <tr key={a.id}>
                <td style={{ color: "var(--text-3)" }}>{a.id}</td>
                <td style={{ fontWeight: "bold" }}>{a.name}</td>
                <td><span className={`status-tag ${a.status}`}>{a.status}</span></td>
                <td style={{ color: "var(--text-3)" }}>{a.lat.toFixed(4)}, {a.lon.toFixed(4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}