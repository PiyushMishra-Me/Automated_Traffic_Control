import React, { useState, useEffect, useRef } from 'react';
import { 
  MapPin, 
  AlertTriangle, 
  CloudRain, 
  Navigation, 
  Layers, 
  ZoomIn, 
  ZoomOut, 
  RotateCcw, 
  ShieldAlert, 
  Compass,
  ArrowRight,
  Activity,
  Radio
} from 'lucide-react';

export default function LiveJunctionMap({ 
  junctions = [], 
  selectedJunction = 'J-01', 
  onSelectJunction, 
  incidents = [],
  activeAmbulances = [],
  weatherData = null,
  onOpenReportModal
}) {
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [hoveredJunction, setHoveredJunction] = useState(null);
  const [activeLayers, setActiveLayers] = useState({
    traffic: true,
    corridors: true,
    diversions: true,
    ambulances: true,
    weather: true
  });

  const svgRef = useRef(null);

  // Map geographic coords (lat: ~28.60-28.65, lng: ~77.19-77.24) to SVG viewBox (800x600)
  const minLat = 28.6000, maxLat = 28.6550;
  const minLng = 77.1900, maxLng = 77.2400;

  const projectCoords = (lat = 28.6139, lng = 77.2090) => {
    const x = ((lng - minLng) / (maxLng - minLng)) * 650 + 75;
    const y = (1 - (lat - minLat) / (maxLat - minLat)) * 480 + 60;
    return { x, y };
  };

  const handleMouseDown = (e) => {
    setIsDragging(true);
    setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
  };

  const handleMouseMove = (e) => {
    if (!isDragging) return;
    setPan({ x: e.clientX - dragStart.x, y: e.clientY - dragStart.y });
  };

  const handleMouseUp = () => setIsDragging(false);

  const resetView = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

  // Find active incidents for all junctions
  const activeIncidentsByJunction = {};
  incidents.forEach(inc => {
    if (inc.status === 'ACTIVE') {
      if (!activeIncidentsByJunction[inc.junction_id]) {
        activeIncidentsByJunction[inc.junction_id] = [];
      }
      activeIncidentsByJunction[inc.junction_id].push(inc);
    }
  });

  return (
    <div className="live-map-card">
      <div className="card-header">
        <div className="title-with-icon">
          <div className="pulse-icon-wrapper">
            <Radio className="icon-pulse text-cyan" size={20} />
            <span className="live-indicator-dot"></span>
          </div>
          <div>
            <h3>Live Urban Junction & Corridor Map</h3>
            <p className="card-subtitle">Real-time geospatial traffic telemetry, active incidents, & upstream diversion routing</p>
          </div>
        </div>

        <div className="map-toolbar">
          <div className="layer-toggles">
            <button 
              className={`layer-btn ${activeLayers.traffic ? 'active' : ''}`}
              onClick={() => setActiveLayers(l => ({ ...l, traffic: !l.traffic }))}
              title="Toggle Traffic Density"
            >
              <Activity size={14} /> Traffic
            </button>
            <button 
              className={`layer-btn ${activeLayers.diversions ? 'active' : ''}`}
              onClick={() => setActiveLayers(l => ({ ...l, diversions: !l.diversions }))}
              title="Toggle Detour Diversions"
            >
              <Navigation size={14} /> Diversions
            </button>
            <button 
              className={`layer-btn ${activeLayers.weather ? 'active' : ''}`}
              onClick={() => setActiveLayers(l => ({ ...l, weather: !l.weather }))}
              title="Toggle Weather Radar"
            >
              <CloudRain size={14} /> Weather
            </button>
          </div>

          <div className="zoom-controls">
            <button onClick={() => setZoom(z => Math.min(z + 0.25, 2.5))} title="Zoom In"><ZoomIn size={16} /></button>
            <button onClick={() => setZoom(z => Math.max(z - 0.25, 0.75))} title="Zoom Out"><ZoomOut size={16} /></button>
            <button onClick={resetView} title="Reset View"><RotateCcw size={16} /></button>
          </div>

          {onOpenReportModal && (
            <button className="report-incident-btn" onClick={onOpenReportModal}>
              <AlertTriangle size={15} /> Report Hazard / Crash
            </button>
          )}
        </div>
      </div>

      <div 
        className="map-canvas-container"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      >
        <svg 
          ref={svgRef}
          viewBox="0 0 800 600" 
          className="map-svg"
          style={{
            transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
            transformOrigin: 'center center',
            cursor: isDragging ? 'grabbing' : 'grab'
          }}
        >
          <defs>
            {/* Grid Pattern */}
            <pattern id="mapGrid" width="40" height="40" patternUnits="userSpaceOnUse">
              <path d="M 40 0 L 0 0 0 40" fill="none" stroke="rgba(255, 255, 255, 0.04)" strokeWidth="1" />
            </pattern>

            {/* Glowing Gradient for Normal Flow */}
            <linearGradient id="flowNormal" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#10b981" stopOpacity="0.8" />
              <stop offset="100%" stopColor="#06b6d4" stopOpacity="0.8" />
            </linearGradient>

            {/* Glowing Gradient for Diversion Flow */}
            <linearGradient id="flowDiversion" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#f59e0b" stopOpacity="1" />
              <stop offset="100%" stopColor="#ec4899" stopOpacity="1" />
            </linearGradient>

            {/* Accident Hazard Glow Filter */}
            <filter id="hazardGlow" x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur in="SourceGraphic" stdDeviation="6" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>

          {/* Map Base Grid */}
          <rect width="100%" height="100%" fill="#0a0f1d" />
          <rect width="100%" height="100%" fill="url(#mapGrid)" />

          {/* City District Blocks & Features */}
          <g className="map-city-zones" opacity="0.35">
            <path d="M 120,90 Q 280,60 440,110 T 700,80 L 720,260 Q 560,280 380,240 Z" fill="#0f1f38" stroke="#1e293b" strokeWidth="1" />
            <path d="M 80,310 Q 260,340 400,310 T 720,380 L 680,540 Q 420,510 140,550 Z" fill="#0f223d" stroke="#1e293b" strokeWidth="1" />
            {/* Waterfront Canal line */}
            <path d="M 50,560 Q 300,500 550,540 T 780,520" fill="none" stroke="#0369a1" strokeWidth="6" opacity="0.3" />
          </g>

          {/* Road Network Corridors */}
          {activeLayers.corridors && (
            <g className="map-roads">
              {/* Interconnecting Road Arterials */}
              {junctions.map((j) => {
                const p1 = projectCoords(j.latitude, j.longitude);
                return (j.connected_junctions || []).map((targetId) => {
                  const targetJ = junctions.find(tj => tj.junction_id === targetId);
                  if (!targetJ) return null;
                  const p2 = projectCoords(targetJ.latitude, targetJ.longitude);
                  const hasIncident = (activeIncidentsByJunction[j.junction_id] || []).length > 0;

                  return (
                    <g key={`road-${j.junction_id}-${targetId}`}>
                      {/* Background Road Base */}
                      <line 
                        x1={p1.x} y1={p1.y} 
                        x2={p2.x} y2={p2.y} 
                        stroke="#1e293b" 
                        strokeWidth="10" 
                        strokeLinecap="round" 
                      />
                      {/* Road Center Line */}
                      <line 
                        x1={p1.x} y1={p1.y} 
                        x2={p2.x} y2={p2.y} 
                        stroke="#334155" 
                        strokeWidth="6" 
                        strokeDasharray="4,6" 
                      />
                      {/* Active Traffic Flow Stream */}
                      {activeLayers.traffic && !hasIncident && (
                        <line 
                          x1={p1.x} y1={p1.y} 
                          x2={p2.x} y2={p2.y} 
                          stroke="url(#flowNormal)" 
                          strokeWidth="2.5" 
                          strokeDasharray="8,16" 
                          className="animated-flow-line" 
                        />
                      )}
                    </g>
                  );
                });
              })}
            </g>
          )}

          {/* Upstream Traffic Diversion Detour Polylines */}
          {activeLayers.diversions && incidents.filter(i => i.status === 'ACTIVE' && i.diversion_plan).map(inc => {
            const jSrc = junctions.find(j => j.junction_id === inc.junction_id);
            const jBypass = junctions.find(j => j.junction_id === inc.diversion_plan.bypass_junction_id);
            if (!jSrc || !jBypass) return null;

            const pSrc = projectCoords(jSrc.latitude, jSrc.longitude);
            const pBypass = projectCoords(jBypass.latitude, jBypass.longitude);

            // Arc detour path
            const dx = pBypass.x - pSrc.x;
            const dy = pBypass.y - pSrc.y;
            const cx = (pSrc.x + pBypass.x) / 2 - dy * 0.35;
            const cy = (pSrc.y + pBypass.y) / 2 + dx * 0.35;
            const d = `M ${pSrc.x},${pSrc.y} Q ${cx},${cy} ${pBypass.x},${pBypass.y}`;

            return (
              <g key={`detour-${inc.incident_id}`} className="diversion-corridor-group">
                <path 
                  d={d} 
                  fill="none" 
                  stroke="rgba(245, 158, 11, 0.3)" 
                  strokeWidth="12" 
                  strokeLinecap="round" 
                />
                <path 
                  d={d} 
                  fill="none" 
                  stroke="url(#flowDiversion)" 
                  strokeWidth="4" 
                  strokeDasharray="10,12" 
                  className="animated-diversion-line" 
                />
                {/* Detour Label */}
                <text 
                  x={cx} 
                  y={cy - 12} 
                  fill="#f59e0b" 
                  fontSize="10" 
                  fontWeight="bold" 
                  textAnchor="middle" 
                  className="detour-label-glow"
                >
                  DETOUR ➔ {inc.diversion_plan.recommended_reroute_corridor}
                </text>
              </g>
            );
          })}

          {/* Active Emergency Ambulance Green Wave Corridors */}
          {activeLayers.ambulances && activeAmbulances.filter(a => a.status !== 'MISSION_ACCOMPLISHED').map(amb => {
            const originJ = junctions.find(j => j.junction_id === amb.origin_junction_id);
            const destJ = junctions.find(j => j.junction_id === amb.destination_junction_id);
            if (!originJ || !destJ) return null;

            const p1 = projectCoords(originJ.latitude, originJ.longitude);
            const p2 = projectCoords(destJ.latitude, destJ.longitude);
            const midX = (p1.x + p2.x) / 2;
            const midY = (p1.y + p2.y) / 2 - 15;

            return (
              <g key={`amb-corridor-${amb.mission_id}`}>
                {/* Neon Green Glow Base */}
                <line 
                  x1={p1.x} y1={p1.y} 
                  x2={p2.x} y2={p2.y} 
                  stroke="rgba(16, 185, 129, 0.25)" 
                  strokeWidth="14" 
                  strokeLinecap="round" 
                />
                {/* Fast Animated Green Wave */}
                <line 
                  x1={p1.x} y1={p1.y} 
                  x2={p2.x} y2={p2.y} 
                  stroke="#10b981" 
                  strokeWidth="3.5" 
                  strokeDasharray="6,10" 
                  className="animated-greenwave-line" 
                />
                {/* Ambulance Vehicle Marker */}
                <g transform={`translate(${midX}, ${midY})`} className="ambulance-map-pin">
                  <circle r="14" fill="#065f46" stroke="#34d399" strokeWidth="2" className="pulse-greenwave" />
                  <text y="4" textAnchor="middle" fontSize="13">🚑</text>
                  <text y="-18" textAnchor="middle" fill="#34d399" fontSize="9" fontWeight="800" className="font-mono">
                    {amb.mission_id} ({amb.criticality === 'CRITICAL_LIFE_THREATENING' ? 'P4-CRITICAL' : amb.criticality})
                  </text>
                </g>
              </g>
            );
          })}

          {/* Junction Nodes / Pins */}
          {junctions.map((j) => {
            const { x, y } = projectCoords(j.latitude, j.longitude);
            const isSelected = selectedJunction === j.junction_id;
            const jIncidents = activeIncidentsByJunction[j.junction_id] || [];
            const hasIncident = jIncidents.length > 0;
            const isHovered = hoveredJunction === j.junction_id;

            return (
              <g 
                key={j.junction_id} 
                className="junction-marker-group"
                transform={`translate(${x}, ${y})`}
                onClick={() => onSelectJunction && onSelectJunction(j.junction_id)}
                onMouseEnter={() => setHoveredJunction(j.junction_id)}
                onMouseLeave={() => setHoveredJunction(null)}
                style={{ cursor: 'pointer' }}
              >
                {/* Selection Rings */}
                {isSelected && (
                  <circle r="28" fill="none" stroke="#06b6d4" strokeWidth="2" opacity="0.6" className="pulsing-selection-ring" />
                )}

                {/* Incident Hazard Halo */}
                {hasIncident && (
                  <circle r="34" fill="rgba(239, 68, 68, 0.25)" stroke="#ef4444" strokeWidth="2" className="hazard-pulse-ring" filter="url(#hazardGlow)" />
                )}

                {/* Base Node Circle */}
                <circle 
                  r={isSelected ? 18 : 15} 
                  fill={hasIncident ? '#ef4444' : (isSelected ? '#0284c7' : '#1e293b')} 
                  stroke={isSelected ? '#38bdf8' : (hasIncident ? '#fca5a5' : '#475569')} 
                  strokeWidth={isSelected ? 3 : 2} 
                />

                {/* Node Center Icon / Text */}
                {hasIncident ? (
                  <text y="4" textAnchor="middle" fill="#ffffff" fontSize="11" fontWeight="bold">⚠️</text>
                ) : (
                  <text y="4" textAnchor="middle" fill="#ffffff" fontSize="10" fontWeight="bold">
                    {j.junction_id.replace('J-', '')}
                  </text>
                )}

                {/* Junction Title Label */}
                <text 
                  y={isSelected ? 32 : 28} 
                  textAnchor="middle" 
                  fill={isSelected ? '#38bdf8' : '#cbd5e1'} 
                  fontSize={isSelected ? "11" : "10"} 
                  fontWeight={isSelected ? "bold" : "normal"}
                  className="junction-node-label"
                >
                  {j.junction_id}
                </text>

                {/* Hover Info Tooltip in SVG */}
                {(isHovered || isSelected) && (
                  <g transform="translate(24, -40)" className="map-tooltip-group">
                    <rect width="180" height="74" rx="8" fill="#0f172a" stroke="#334155" strokeWidth="1" filter="drop-shadow(0 4px 6px rgba(0,0,0,0.5))" />
                    <text x="10" y="18" fill="#f8fafc" fontSize="11" fontWeight="bold">{j.name || j.junction_id}</text>
                    <text x="10" y="34" fill="#94a3b8" fontSize="9">{j.location || 'Urban Arterial'}</text>
                    <text x="10" y="50" fill={hasIncident ? '#f87171' : '#34d399'} fontSize="9" fontWeight="bold">
                      {hasIncident ? `⚠️ ${jIncidents.length} Active Incident(s)` : '● Normal Traffic Flow'}
                    </text>
                    <text x="10" y="64" fill="#38bdf8" fontSize="8">Click to Inspect Junction ➔</text>
                  </g>
                )}
              </g>
            );
          })}
        </svg>

        {/* Live Weather Overlay Chip */}
        {activeLayers.weather && weatherData && (
          <div className="map-weather-chip">
            <CloudRain size={16} className="text-cyan" />
            <div>
              <div className="weather-chip-title">{weatherData.condition} • {weatherData.temperature_c}°C</div>
              <div className="weather-chip-sub">Surface: {weatherData.road_surface} | Rain: {weatherData.precipitation_mm} mm/h</div>
            </div>
          </div>
        )}

        {/* Map Legend */}
        <div className="map-legend">
          <div className="legend-item"><span className="legend-dot bg-green"></span> Normal Traffic</div>
          <div className="legend-item"><span className="legend-dot bg-amber"></span> Detour Corridor</div>
          <div className="legend-item"><span className="legend-dot bg-red"></span> Accident / Blockage</div>
        </div>
      </div>
    </div>
  );
}
