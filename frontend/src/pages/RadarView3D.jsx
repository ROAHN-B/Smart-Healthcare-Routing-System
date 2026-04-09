import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls";

const API = import.meta.env.PROD ? window.location.origin : "http://localhost:8000";

const LAT_MIN = 17.60, LAT_MAX = 17.75;
const LON_MIN = 75.85, LON_MAX = 76.00;
const MAP_SIZE = 100;

function createTextSprite(message) {
  const canvas = document.createElement('canvas');
  canvas.width = 512; canvas.height = 128;
  const context = canvas.getContext('2d');
  context.fillStyle = 'rgba(10, 10, 10, 0.8)';
  context.fillRect(0, 0, 512, 128);
  context.strokeStyle = '#00ffff';
  context.lineWidth = 4;
  context.strokeRect(0, 0, 512, 128);
  context.font = 'bold 44px monospace';
  context.textAlign = 'center'; context.textBaseline = 'middle';
  context.fillStyle = '#ffffff';
  context.fillText(message.toUpperCase(), 256, 64);
  const texture = new THREE.CanvasTexture(canvas);
  texture.minFilter = THREE.LinearFilter;
  const spriteMaterial = new THREE.SpriteMaterial({ map: texture, depthTest: false });
  const sprite = new THREE.Sprite(spriteMaterial);
  sprite.scale.set(6, 1.5, 1);
  return sprite;
}

// UPGRADED: Tall Skyscraper Hospital with an open "Drive-Through" base!
const createHospital = (color, name) => {
  const group = new THREE.Group();

  const mainColor = 0x0a1526;
  const buildMat = new THREE.MeshPhongMaterial({ color: mainColor, transparent: true, opacity: 0.95, shininess: 100 });
  
  // We name the glowing materials so we can easily change their color (Green/Red) in the animation loop
  const edgeMat = new THREE.LineBasicMaterial({ color: color, transparent: true, opacity: 0.8 });
  edgeMat.name = "hospitalColorMat";

  const crossMat = new THREE.MeshBasicMaterial({ color: color });
  crossMat.name = "hospitalColorMat";

  // 1. Four Pillars (Creates the open ER drop-off archway underneath)
  const pillarGeo = new THREE.BoxGeometry(0.8, 4, 0.8);
  const positions = [
    [-1.4, 2, -1.4], [1.4, 2, -1.4],
    [-1.4, 2, 1.4], [1.4, 2, 1.4]
  ];
  
  positions.forEach(pos => {
    const pillar = new THREE.Mesh(pillarGeo, buildMat);
    pillar.position.set(...pos);
    
    const pEdges = new THREE.EdgesGeometry(pillarGeo);
    const pLines = new THREE.LineSegments(pEdges, edgeMat);
    pLines.position.set(...pos);
    
    group.add(pillar, pLines);
  });

  // 2. Main Tower Building (Sits on top of the pillars)
  const bodyGeo = new THREE.BoxGeometry(3.6, 10, 3.6);
  const building = new THREE.Mesh(bodyGeo, buildMat);
  building.position.set(0, 9, 0); // Spans from y=4 to y=14

  const edges = new THREE.EdgesGeometry(bodyGeo);
  const lines = new THREE.LineSegments(edges, edgeMat);
  lines.position.set(0, 9, 0);

  // 3. Glowing Medical Cross in the center of the tower
  const vCross = new THREE.Mesh(new THREE.BoxGeometry(1.0, 3.5, 3.8), crossMat);
  vCross.position.set(0, 9, 0);
  const hCross = new THREE.Mesh(new THREE.BoxGeometry(3.5, 1.0, 3.8), crossMat);
  hCross.position.set(0, 9, 0);

  // 4. Helipad Roof & Label
  const roofGeo = new THREE.CylinderGeometry(1.5, 1.5, 0.2, 16);
  const roofMat = new THREE.MeshBasicMaterial({ color: 0x333333 });
  const roof = new THREE.Mesh(roofGeo, roofMat);
  roof.position.set(0, 14.1, 0);

  const label = createTextSprite(name);
  label.position.set(0, 16.5, 0);

  group.add(building, lines, vCross, hCross, roof, label);
  return group;
};

const createAmbulance = (color) => {
  const group = new THREE.Group();
  const core = new THREE.Mesh(new THREE.BoxGeometry(2.0, 1.0, 4.0), new THREE.MeshPhongMaterial({ color: 0x1a1a1a, shininess: 200 })); core.position.y = 1.0;
  const trimMat = new THREE.MeshBasicMaterial({ color: 0x00ffff }); 
  const sideTrimL = new THREE.Mesh(new THREE.BoxGeometry(0.15, 0.2, 3.8), trimMat); sideTrimL.position.set(-1.05, 0.8, 0);
  const sideTrimR = new THREE.Mesh(new THREE.BoxGeometry(0.15, 0.2, 3.8), trimMat); sideTrimR.position.set(1.05, 0.8, 0);
  const canopy = new THREE.Mesh(new THREE.BoxGeometry(1.6, 0.6, 1.4), new THREE.MeshBasicMaterial({ color: 0xccffff })); canopy.position.set(0, 1.8, -0.8);
  const lightL = new THREE.Mesh(new THREE.BoxGeometry(0.7, 0.3, 0.5), new THREE.MeshBasicMaterial({ color: 0xff0000 })); lightL.position.set(-0.5, 2.1, 0.8);
  const lightR = new THREE.Mesh(new THREE.BoxGeometry(0.7, 0.3, 0.5), new THREE.MeshBasicMaterial({ color: 0x0000ff })); lightR.position.set(0.5, 2.1, 0.8);
  const thrusterMat = new THREE.MeshPhongMaterial({ color: 0x111111 });
  const glowMat = new THREE.MeshBasicMaterial({ color: 0x00aaff }); 
  [[-1.2, 0.5, -1.3], [1.2, 0.5, -1.3], [-1.2, 0.5, 1.3], [1.2, 0.5, 1.3]].forEach(pos => {
    const housing = new THREE.Mesh(new THREE.CylinderGeometry(0.6, 0.5, 0.4, 16), thrusterMat); housing.position.set(...pos);
    const glow = new THREE.Mesh(new THREE.CylinderGeometry(0.4, 0.4, 0.45, 16), glowMat); glow.position.set(pos[0], pos[1]-0.05, pos[2]); 
    group.add(housing, glow);
  });
  group.add(core, sideTrimL, sideTrimR, canopy, lightL, lightR);
  group.scale.set(0.55, 0.55, 0.55); 
  return group;
};

export default function RadarView3D() {
  const containerRef = useRef(null);
  const dataRef = useRef({ hospitals: [], ambulances: [], patients: [] });
  const meshesRef = useRef({ hospitals: {}, patients: {}, ambulances: {} });
  const wsRef = useRef(null);

  const project3D = (lat, lon) => {
    const x = ((lon - LON_MIN) / (LON_MAX - LON_MIN) - 0.5) * MAP_SIZE;
    const z = -(((lat - LAT_MIN) / (LAT_MAX - LAT_MIN) - 0.5) * MAP_SIZE);
    return new THREE.Vector3(x, 0, z);
  };

  useEffect(() => {
    if (!containerRef.current) return;
    meshesRef.current = { hospitals: {}, patients: {}, ambulances: {} };

    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#050505");
    
    const initialWidth = containerRef.current.clientWidth || window.innerWidth;
    const initialHeight = containerRef.current.clientHeight || window.innerHeight;

    const camera = new THREE.PerspectiveCamera(45, initialWidth / initialHeight, 1, 1000);
    camera.position.set(0, 60, 70); 
    camera.lookAt(0, 0, 0);
    
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(initialWidth, initialHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    
    containerRef.current.innerHTML = ''; 
    containerRef.current.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.maxPolarAngle = Math.PI / 2 - 0.05; 

    scene.add(new THREE.AmbientLight(0xffffff, 2.0));
    const dirLight = new THREE.DirectionalLight(0xffffff, 3);
    dirLight.position.set(20, 50, 20);
    scene.add(dirLight);

    const ground = new THREE.Mesh(
        new THREE.PlaneGeometry(MAP_SIZE + 10, MAP_SIZE + 10),
        new THREE.MeshPhongMaterial({ color: 0x050505, depthWrite: false })
    );
    ground.rotation.x = -Math.PI / 2;
    scene.add(ground);

    const gridHelper = new THREE.GridHelper(MAP_SIZE, 20, 0x004488, 0x001133);
    gridHelper.position.y = 0.01;
    scene.add(gridHelper);

    const cityGroup = new THREE.Group();
    const buildMat = new THREE.MeshPhongMaterial({ color: 0x0a0f1a, transparent: true, opacity: 0.95, shininess: 80 });
    const edgeMat = new THREE.LineBasicMaterial({ color: 0x0088ff, transparent: true, opacity: 0.4 });
    
    for (let x = -MAP_SIZE/2 + 2.5; x < MAP_SIZE/2; x += 5) {
        for (let z = -MAP_SIZE/2 + 2.5; z < MAP_SIZE/2; z += 5) {
            if (Math.random() > 0.1) { 
                const h = 4 + Math.random() * 12;
                const geo = new THREE.BoxGeometry(3.2, h, 3.2);
                const mesh = new THREE.Mesh(geo, buildMat);
                mesh.position.set(x, h/2, z);
                
                const edges = new THREE.EdgesGeometry(geo);
                const lines = new THREE.LineSegments(edges, edgeMat);
                lines.position.set(x, h/2, z);
                
                cityGroup.add(mesh, lines);
            }
        }
    }
    scene.add(cityGroup);

    const resizeObserver = new ResizeObserver((entries) => {
      for (let entry of entries) {
        const { width, height } = entry.contentRect;
        if (width > 0 && height > 0) {
          camera.aspect = width / height;
          camera.updateProjectionMatrix();
          renderer.setSize(width, height);
        }
      }
    });
    resizeObserver.observe(containerRef.current);

    fetch(`${API}/get_live_tracking`)
      .then(res => res.json())
      .then(data => { if (data && data.hospitals) dataRef.current = data; })
      .catch(err => console.error("Initial fetch failed:", err));

    let animationFrameId;
    let time = 0;

    const animate = () => {
      animationFrameId = requestAnimationFrame(animate);
      controls.update();
      time += 0.05;

      const { hospitals = [], ambulances = [], patients = [] } = dataRef.current;
      
      hospitals.forEach(h => {
        const isAvailable = h.beds_available > 0;
        const colorHex = isAvailable ? 0x22ff22 : 0xff2222;

        if (!meshesRef.current.hospitals[h.id]) {
          const mesh = createHospital(colorHex, h.name);
          mesh.position.copy(project3D(h.lat, h.lon));
          scene.add(mesh);
          meshesRef.current.hospitals[h.id] = mesh;
        } else {
          // Dynamic color updating for full/available hospitals (safely searches for our named materials)
          const mesh = meshesRef.current.hospitals[h.id];
          mesh.traverse((child) => {
            if (child.material && child.material.name === "hospitalColorMat") {
                child.material.color.setHex(colorHex);
            }
          });
        }
      });

      const activePatientIds = new Set();

      patients.forEach(p => {
        if (p.status === "picked_up" || p.status === "admitted") return;
        
        activePatientIds.add(String(p.id));

        if (!meshesRef.current.patients[p.id]) {
          const mesh = new THREE.Mesh(
            new THREE.SphereGeometry(0.8, 16, 16), 
            new THREE.MeshBasicMaterial({ color: 0xff2222, wireframe: true })
          );
          scene.add(mesh);
          meshesRef.current.patients[p.id] = mesh;
        }

        const mesh = meshesRef.current.patients[p.id];
        mesh.position.copy(project3D(p.lat, p.lon));
        mesh.position.y = 1.0; 
        const scale = 1 + Math.sin(time) * 0.15;
        mesh.scale.set(scale, scale, scale);
        mesh.rotation.y += 0.02;
      });

      Object.keys(meshesRef.current.patients).forEach(id => {
        if (!activePatientIds.has(id)) {
          const mesh = meshesRef.current.patients[id];
          scene.remove(mesh);
          mesh.geometry.dispose();  
          mesh.material.dispose();  
          delete meshesRef.current.patients[id];
        }
      });

      ambulances.forEach(a => {
        if (!meshesRef.current.ambulances[a.id]) {
          const mesh = createAmbulance(0x22aaff);
          const startPos = project3D(a.lat, a.lon);
          mesh.position.set(startPos.x, 1.0, startPos.z);
          scene.add(mesh);
          meshesRef.current.ambulances[a.id] = mesh;
        }
        
        const m = meshesRef.current.ambulances[a.id];
        const targetPos = project3D(a.lat, a.lon);
        
        const lerpSpeed = 0.1; 
        const currentX = m.position.x;
        const currentZ = m.position.z;
        
        m.position.x += (targetPos.x - currentX) * lerpSpeed;
        m.position.z += (targetPos.z - currentZ) * lerpSpeed;
        
        const moveX = m.position.x - currentX;
        const moveZ = m.position.z - currentZ;
        
        if (Math.abs(moveX) > 0.0001 || Math.abs(moveZ) > 0.0001) {
            m.lookAt(m.position.x + moveX, m.position.y, m.position.z + moveZ);
        }
        
        m.position.y = 1.0 + Math.sin(time * 2 + a.id.charCodeAt(0)) * 0.2; 
      });

      renderer.render(scene, camera);
    };
    animate();

    const connectWS = () => {
      const wsUrl = window.location.hostname === 'localhost' 
        ? "ws://localhost:8000/ws/live" 
        : `${API.replace(/^http/, "ws")}/ws/live`.replace(/([^:]\/)\/+/g, "$1");

      const ws = new WebSocket(wsUrl);
      ws.onmessage = (e) => { 
        try { 
          const parsed = JSON.parse(e.data); 
          if (parsed.patients) {
            parsed.patients = parsed.patients.filter(p => p.status !== "admitted");
          }
          dataRef.current = parsed; 
        } catch(err){} 
      };
      ws.onclose = () => setTimeout(connectWS, 2000);
      wsRef.current = ws;
    };
    
    connectWS();

    return () => {
      resizeObserver.disconnect();
      cancelAnimationFrame(animationFrameId);
      renderer.dispose();
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%', minHeight: 'calc(100vh - 100px)', overflow: 'hidden', borderRadius: '8px' }}>
      <div ref={containerRef} style={{ position: 'absolute', top: 0, bottom: 0, left: 0, right: 0, cursor: 'grab', outline: 'none' }} />
      <div style={{
        position: 'absolute', top: '20px', right: '20px', backgroundColor: 'rgba(0, 0, 0, 0.85)',
        border: '1px solid #333', padding: '15px', borderRadius: '4px',
        fontFamily: 'monospace', fontSize: '11px', color: '#eee',
        display: 'flex', flexDirection: 'column', gap: '10px', pointerEvents: 'none'
      }}>
        <div style={{ color: '#888', fontWeight: 'bold', borderBottom: '1px solid #333', paddingBottom: '5px' }}>SYSTEM.LEGEND.3D</div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}><div style={{ width: '10px', height: '10px', borderRadius: '50%', border: '2px solid #ff2222', boxShadow: '0 0 5px #ff2222' }} /><span>PATIENT</span></div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}><div style={{ width: '0', height: '0', borderLeft: '6px solid transparent', borderRight: '6px solid transparent', borderBottom: '10px solid #22aaff' }} /><span>AMBULANCE</span></div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}><div style={{ width: '10px', height: '10px', background: '#22ff22', boxShadow: '0 0 5px #22ff22' }} /><span>HOSPITAL (OPEN)</span></div>
      </div>
    </div>
  );
}