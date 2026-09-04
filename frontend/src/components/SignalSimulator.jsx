import React, { useEffect, useRef, useState } from 'react';
import {
  AlertTriangle, Car, Clock3, Gauge, Pause, Play, RadioTower,
  RotateCcw, ShieldCheck, TrendingDown, TrendingUp, Layers,
  Plus, Trash2, ArrowRight, ArrowUp, ArrowDown, ArrowLeft,
  ShieldAlert, Zap, Compass, CheckCircle2, Sparkles, Grid, Radio
} from 'lucide-react';
import { api } from '../services/api';
import ManualSignalModal from './ManualSignalModal';

const APPROACHES = ['NORTH', 'SOUTH', 'EAST', 'WEST'];
const STEP_MS = 90; // wall-clock ms between animation frames

// Directional styling configuration
const DIR_CONFIG = {
  NORTH: { label: 'North', icon: ArrowUp, color: '#0284c7', bg: '#f0f9ff', border: '#bae6fd' },
  SOUTH: { label: 'South', icon: ArrowDown, color: '#d97706', bg: '#fffbeb', border: '#fde68a' },
  EAST: { label: 'East', icon: ArrowRight, color: '#16a34a', bg: '#f0fdf4', border: '#bbf7d0' },
  WEST: { label: 'West', icon: ArrowLeft, color: '#7c3aed', bg: '#faf5ff', border: '#e9d5ff' },
};

function SignalHead({ approach, color, isForcedRed, onToggleForcedRed }) {
  return (
    <div 
      className={`sig-head sig-${approach.toLowerCase()} ${isForcedRed ? 'forced-red-head' : ''}`}
      onClick={onToggleForcedRed}
      title={isForcedRed ? `Approach ${approach} is LOCKED RED. Click to release.` : `Click to manually force RED on ${approach}`}
    >
      <div className="sig-head-header">
        <span className="sig-label">{approach[0]}</span>
        {isForcedRed && <span className="sig-lock-dot" title="Manual RED Hold" />}
      </div>
      <div className="sig-lamps">
        <i className={`sig-dot red ${color === 'RED' || isForcedRed ? 'on pulse-red' : ''}`} />
        <i className={`sig-dot yellow ${color === 'YELLOW' && !isForcedRed ? 'on' : ''}`} />
        <i className={`sig-dot green ${color === 'GREEN' && !isForcedRed ? 'on' : ''}`} />
      </div>
    </div>
  );
}

export default function SignalSimulator({ 
  junctionId, 
  junctions = [], 
  onSelectJunction, 
  refreshKey,
  onRefresh
}) {
  // Mode: 'SINGLE' | 'NETWORK_STAR'
  const [simMode, setSimMode] = useState('SINGLE');

  // Single Junction State
  const [recommendation, setRecommendation] = useState(null);
  const [forcedRedApproaches, setForcedRedApproaches] = useState([]);
  const [sim, setSim] = useState(null);

  // Network / Star Topology State
  const [centralJunctionId, setCentralJunctionId] = useState(junctionId || 'J-01');
  const [directionalNeighbors, setDirectionalNeighbors] = useState({
    NORTH: null,
    SOUTH: null,
    EAST: null,
    WEST: null,
  });
  const [additionalJunctions, setAdditionalJunctions] = useState([]);
  const [corridorForcedRed, setCorridorForcedRed] = useState({}); // { [jId]: ['EAST'] }
  const [corridorSim, setCorridorSim] = useState(null);
  const [viewLayout, setViewLayout] = useState('STAR'); // 'STAR' | 'GRID'

  // Live Physical Police Override State
  const [isLiveModalOpen, setIsLiveModalOpen] = useState(false);
  const [liveOverride, setLiveOverride] = useState(null);

  const activeTargetJunctionId = simMode === 'SINGLE' ? (junctionId || 'J-01') : (centralJunctionId || 'J-01');

  const fetchLiveOverride = async () => {
    if (!activeTargetJunctionId) return;
    try {
      const data = await api.getManualSignalOverride(activeTargetJunctionId);
      setLiveOverride(data && data.active ? data : null);
    } catch (e) {
      console.warn('Failed to fetch live manual signal override', e);
    }
  };

  useEffect(() => {
    fetchLiveOverride();
    const interval = setInterval(fetchLiveOverride, 4000);
    return () => clearInterval(interval);
  }, [activeTargetJunctionId, refreshKey]);

  // Playback & Animation State
  const [stepIndex, setStepIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(2);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const stageRef = useRef(null);

  // Auto-wire surrounding neighbors based on central junction when it changes
  useEffect(() => {
    if (junctionId) {
      setCentralJunctionId(junctionId);
    }
  }, [junctionId]);

  useEffect(() => {
    const curJ = junctions.find(j => j.junction_id === centralJunctionId);
    const others = junctions.filter(j => j.junction_id !== centralJunctionId);
    
    // Map known Delhi default neighbors or pick from others
    const newNeighbors = { NORTH: null, SOUTH: null, EAST: null, WEST: null };
    
    if (centralJunctionId === 'J-01') {
      newNeighbors.NORTH = others.find(j => j.junction_id === 'J-03')?.junction_id || others[0]?.junction_id || null;
      newNeighbors.EAST = others.find(j => j.junction_id === 'J-02')?.junction_id || others[1]?.junction_id || null;
      newNeighbors.SOUTH = others.find(j => j.junction_id === 'J-04')?.junction_id || others[2]?.junction_id || null;
      newNeighbors.WEST = others.find(j => j.junction_id === 'J-05')?.junction_id || others[3]?.junction_id || null;
    } else {
      // Auto-assign any connected junctions
      const conn = curJ?.connected_junctions || [];
      if (conn[0]) newNeighbors.EAST = conn[0];
      if (conn[1]) newNeighbors.NORTH = conn[1];
      if (conn[2]) newNeighbors.SOUTH = conn[2];
      if (conn[3]) newNeighbors.WEST = conn[3];
    }
    
    setDirectionalNeighbors(newNeighbors);
  }, [centralJunctionId, junctions]);

  // Compute all active junctions in the network
  const allNetworkJunctionIds = Array.from(new Set([
    centralJunctionId,
    ...Object.values(directionalNeighbors).filter(Boolean),
    ...additionalJunctions,
  ]));

  // Load instantaneous recommendation for active junction
  const loadRecommendation = async () => {
    if (!junctionId) return;
    setError(null);
    try {
      setRecommendation(await api.getSignalRecommendation(junctionId));
    } catch (err) {
      setError(err.message);
    }
  };

  const handleOverrideApplied = () => {
    fetchLiveOverride();
    loadRecommendation();
    if (onRefresh) onRefresh();
  };

  // Reset simulation when junction, refreshKey, or mode changes
  useEffect(() => {
    setSim(null);
    setCorridorSim(null);
    setStepIndex(0);
    setPlaying(false);
    loadRecommendation();
  }, [junctionId, refreshKey, simMode]);

  // Drive animation timeline
  const activeTimelineLength = simMode === 'SINGLE' 
    ? (sim?.steps?.length || 0) 
    : (corridorSim?.steps?.length || 0);

  useEffect(() => {
    if (!playing || activeTimelineLength === 0) return undefined;
    const id = setInterval(() => {
      setStepIndex((i) => Math.min(i + speed, activeTimelineLength - 1));
    }, STEP_MS);
    return () => clearInterval(id);
  }, [playing, speed, activeTimelineLength]);

  // Stop playback when reaching end of timeline
  useEffect(() => {
    if (activeTimelineLength > 0 && playing && stepIndex >= activeTimelineLength - 1) {
      setPlaying(false);
    }
  }, [stepIndex, playing, activeTimelineLength]);

  // Toggle Forced RED on single junction
  const toggleForcedRedSingle = (app) => {
    setForcedRedApproaches(prev => {
      const next = prev.includes(app) ? prev.filter(a => a !== app) : [...prev, app];
      return next;
    });
  };

  // Toggle Forced RED on a specific junction in network
  const toggleForcedRedCorridor = (jId, app) => {
    setCorridorForcedRed(prev => {
      const currentList = prev[jId] || [];
      const updatedList = currentList.includes(app) 
        ? currentList.filter(a => a !== app) 
        : [...currentList, app];
      return { ...prev, [jId]: updatedList };
    });
  };

  // Assign or remove directional neighbor
  const handleSetDirectionNeighbor = (direction, targetJId) => {
    setDirectionalNeighbors(prev => ({
      ...prev,
      [direction]: targetJId || null,
    }));
  };

  // One-click select all 5 city matrix junctions
  const handleSelectAll5Junctions = () => {
    const j01 = junctions.find(j => j.junction_id === 'J-01')?.junction_id || junctions[0]?.junction_id;
    if (j01) setCentralJunctionId(j01);
    
    setDirectionalNeighbors({
      NORTH: junctions.find(j => j.junction_id === 'J-03')?.junction_id || null,
      SOUTH: junctions.find(j => j.junction_id === 'J-04')?.junction_id || null,
      EAST: junctions.find(j => j.junction_id === 'J-02')?.junction_id || null,
      WEST: junctions.find(j => j.junction_id === 'J-05')?.junction_id || null,
    });
  };

  // Run Single Junction Simulation
  const runSingleSimulation = async () => {
    if (!junctionId) return;
    setLoading(true);
    setError(null);
    try {
      const result = await api.simulateSignal(junctionId, 180, forcedRedApproaches);
      setSim(result);
      setStepIndex(0);
      setPlaying(true);
      requestAnimationFrame(() => stageRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' }));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Run Full Multi-Junction Network Simulation
  const runNetworkSimulation = async () => {
    if (allNetworkJunctionIds.length === 0) return;
    setLoading(true);
    setError(null);
    try {
      const result = await api.simulateCorridor({
        junctionIds: allNetworkJunctionIds,
        forcedRed: corridorForcedRed,
        horizonSeconds: 180,
      });
      setCorridorSim(result);
      setStepIndex(0);
      setPlaying(true);
      requestAnimationFrame(() => stageRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' }));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleRunSimulation = () => {
    if (simMode === 'SINGLE') {
      runSingleSimulation();
    } else {
      runNetworkSimulation();
    }
  };

  const togglePlay = () => {
    if (activeTimelineLength === 0) return;
    if (!playing && stepIndex >= activeTimelineLength - 1) setStepIndex(0);
    setPlaying((p) => !p);
  };

  const replay = () => {
    setStepIndex(0);
    setPlaying(true);
  };

  // Single simulation step data
  const step = sim ? sim.steps[Math.min(stepIndex, sim.steps.length - 1)] : null;
  const atEnd = activeTimelineLength > 0 ? stepIndex >= activeTimelineLength - 1 : false;
  const queueScale = sim ? Math.max(6, ...sim.per_approach.map((p) => p.max_queue)) : 6;
  const comparison = sim?.comparison;
  const waitScale = comparison ? Math.max(comparison.adaptive_avg_wait, comparison.fixed_avg_wait, 1) : 1;
  const improved = comparison ? comparison.improvement_pct >= 0 : true;

  // Corridor simulation step data
  const corridorStep = corridorSim ? corridorSim.steps[Math.min(stepIndex, corridorSim.steps.length - 1)] : null;
  const corridorComp = corridorSim?.corridor_comparison;
  const corridorImproved = corridorComp ? corridorComp.improvement_pct >= 0 : true;
  const corridorWaitScale = corridorComp ? Math.max(corridorComp.adaptive_avg_wait, corridorComp.fixed_avg_wait, 1) : 1;

  const phaseLabel = recommendation?.recommended_phase?.replaceAll('_', ' ') || 'CALCULATING';

  // Helper to render a compact junction card
  const renderJunctionNodeCard = (jId, roleLabel = 'Connected') => {
    if (!jId) return null;
    const jData = junctions.find(j => j.junction_id === jId);
    const jStep = corridorStep?.junctions?.[jId];
    const jForced = corridorForcedRed[jId] || [];

    return (
      <div className={`star-junc-box ${jId === centralJunctionId ? 'central-box' : ''} ${jForced.length > 0 ? 'has-overrides' : ''}`}>
        <div className="star-junc-head">
          <div>
            <span className="j-badge font-mono">{jId}</span>
            <strong className="star-junc-name">{jData?.name || jId}</strong>
          </div>
          <span className="role-tag font-mono">{roleLabel}</span>
        </div>

        {jStep && (
          <>
            <div className="sim-cross mini-cross">
              <div className="sim-cell north">
                <SignalHead 
                  approach="NORTH" 
                  color={jStep.lights.NORTH} 
                  isForcedRed={jForced.includes('NORTH')}
                  onToggleForcedRed={() => toggleForcedRedCorridor(jId, 'NORTH')}
                />
              </div>
              <div className="sim-cell west">
                <SignalHead 
                  approach="WEST" 
                  color={jStep.lights.WEST} 
                  isForcedRed={jForced.includes('WEST')}
                  onToggleForcedRed={() => toggleForcedRedCorridor(jId, 'WEST')}
                />
              </div>
              <div className="sim-cell center">
                <span className="sim-clock"><Clock3 size={10} /> {jStep.phase_time_remaining}s</span>
                <span className="phase-mini-txt">{jStep.phase_label.split(' ')[0]}</span>
              </div>
              <div className="sim-cell east">
                <SignalHead 
                  approach="EAST" 
                  color={jStep.lights.EAST} 
                  isForcedRed={jForced.includes('EAST')}
                  onToggleForcedRed={() => toggleForcedRedCorridor(jId, 'EAST')}
                />
              </div>
              <div className="sim-cell south">
                <SignalHead 
                  approach="SOUTH" 
                  color={jStep.lights.SOUTH} 
                  isForcedRed={jForced.includes('SOUTH')}
                  onToggleForcedRed={() => toggleForcedRedCorridor(jId, 'SOUTH')}
                />
              </div>
            </div>

            <div className="corridor-queues-list">
              {APPROACHES.map(a => (
                <div key={a} className="corridor-q-item">
                  <span className="q-label font-mono">{a[0]}:</span>
                  <div className="q-track">
                    <div 
                      className={`q-fill ${jStep.lights[a] === 'GREEN' ? 'flowing' : ''} ${jForced.includes(a) ? 'locked-red' : ''}`}
                      style={{ width: `${Math.min(100, (jStep.queues[a] / 15) * 100)}%` }}
                    />
                  </div>
                  <span className="q-val font-mono">{jStep.queues[a]}</span>
                </div>
              ))}
            </div>
          </>
        )}

        {/* Directional Red Overrides for this node */}
        <div className="node-overrides-mini">
          <span className="mini-lbl">Force RED:</span>
          {APPROACHES.map(a => {
            const isLocked = jForced.includes(a);
            return (
              <button
                key={a}
                type="button"
                className={`mini-red-pill ${isLocked ? 'locked' : ''}`}
                onClick={() => toggleForcedRedCorridor(jId, a)}
                title={`Toggle manual RED on ${jId} ${a}`}
              >
                {a[0]}
              </button>
            );
          })}
        </div>
      </div>
    );
  };

  // Helper to render animated vehicle platoons moving along a directional corridor arm
  const renderPlatoonsOnLink = (uId, dId, orientation = 'horizontal') => {
    if (!corridorStep?.transit) return null;
    const platoons = corridorStep.transit.filter(
      p => (p.upstream_junction_id === uId && p.downstream_junction_id === dId) ||
           (p.upstream_junction_id === dId && p.downstream_junction_id === uId)
    );

    return (
      <div className={`corridor-star-link ${orientation}`}>
        <div className="star-road-pipe">
          <div className="road-stripes" />
          {platoons.map((platoon, pIdx) => {
            const isForward = platoon.upstream_junction_id === uId;
            const pct = isForward ? platoon.progress_pct * 100 : (1 - platoon.progress_pct) * 100;
            const posStyle = orientation === 'horizontal' 
              ? { left: `${Math.min(92, Math.max(8, pct))}%` }
              : { top: `${Math.min(92, Math.max(8, pct))}%` };

            return (
              <div 
                key={pIdx} 
                className="moving-vehicle-platoon star-platoon"
                style={posStyle}
                title={`In transit: ${platoon.vehicles_in_transit} veh (${platoon.upstream_junction_id} ➔ ${platoon.downstream_junction_id})`}
              >
                <Car size={11} className="platoon-car-icon" />
                <span className="platoon-count-badge font-mono">{platoon.vehicles_in_transit}</span>
              </div>
            );
          })}
        </div>
        <span className="star-link-label font-mono">
          {platoons.length > 0 ? `⚡ ${platoons.reduce((a, b) => a + b.vehicles_in_transit, 0)} veh flowing` : 'Clear Flow'}
        </span>
      </div>
    );
  };

  return (
    <section className="signal-simulator animate-fade-in">
      {/* HEADER & MODE SWITCHER */}
      <div className="panel-header sim-header-wrapper">
        <div className="sim-title-group">
          <h2><RadioTower size={20} color="#0284c7" /> 4-Way Multi-Junction &amp; City Matrix Simulator</h2>
          <span className="simulation-badge"><ShieldCheck size={13} /> Multi-Corridor Telemetry Sandbox</span>
        </div>

        {/* RIGHT CONTROLS: LIVE SIGNAL OVERRIDE & MODE TOGGLE */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap' }}>
          <button
            type="button"
            className={`btn-override-signal ${liveOverride && liveOverride.active ? 'active' : ''}`}
            onClick={() => setIsLiveModalOpen(true)}
            title="Police Command: Manually change traffic light on live junction"
            style={{ padding: '8px 14px', borderRadius: '10px' }}
          >
            <Radio size={14} className={liveOverride && liveOverride.active ? "animate-pulse" : ""} />
            {liveOverride && liveOverride.active 
              ? `🚨 Live Override: ${liveOverride.override_mode}` 
              : `Change Live Signal Light (${activeTargetJunctionId})`}
          </button>

          <div className="sim-mode-switcher">
            <button 
              type="button"
              className={`sim-mode-btn ${simMode === 'SINGLE' ? 'active' : ''}`}
              onClick={() => setSimMode('SINGLE')}
            >
              <Compass size={14} /> Single Intersection
            </button>
            <button 
              type="button"
              className={`sim-mode-btn ${simMode === 'NETWORK_STAR' ? 'active' : ''}`}
              onClick={() => setSimMode('NETWORK_STAR')}
            >
              <Layers size={14} /> 4-Way Multi-Junction Network ({allNetworkJunctionIds.length} Junctions)
            </button>
          </div>
        </div>
      </div>

      {error && <p className="form-error">{error}</p>}

      {/* =========================================================================
         SECTION 1: MANUAL DIRECTIONAL RED LIGHT OVERRIDE CONTROLS (SINGLE MODE)
         ========================================================================= */}
      {simMode === 'SINGLE' && (
        <div className="manual-override-container">
          {/* LIVE OVERRIDE ACTIVE BANNER */}
          {liveOverride && liveOverride.active && (
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '10px', padding: '10px 14px', background: 'rgba(239, 68, 68, 0.12)', border: '1px solid #ef4444', borderRadius: '8px', marginBottom: '12px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#b91c1c', fontSize: '0.8rem', fontWeight: 700 }}>
                <Radio size={16} className="text-red animate-pulse" />
                <span>
                  POLICE LIVE SIGNAL OVERRIDE ACTIVE: {liveOverride.reason} ({liveOverride.override_mode})
                </span>
              </div>
              <button
                type="button"
                className="btn-override-signal active"
                onClick={() => setIsLiveModalOpen(true)}
                style={{ padding: '4px 10px', fontSize: '0.74rem' }}
              >
                Manage Live Override
              </button>
            </div>
          )}

          <div className="manual-override-header">
            <div className="override-title">
              <ShieldAlert size={16} className={forcedRedApproaches.length > 0 ? "text-red animate-pulse" : "text-slate"} />
              <strong>Simulation Red Light Sandbox Overrides ({junctionId})</strong>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <button
                type="button"
                className={`btn-override-signal ${liveOverride && liveOverride.active ? 'active' : ''}`}
                onClick={() => setIsLiveModalOpen(true)}
                style={{ fontSize: '0.74rem', padding: '5px 10px' }}
                title="Open Live Physical Signal Override Modal"
              >
                <Radio size={13} className={liveOverride && liveOverride.active ? "animate-pulse" : ""} />
                {liveOverride && liveOverride.active ? 'Active Police Override' : 'Police Live Light Control'}
              </button>
              {forcedRedApproaches.length > 0 && (
                <span className="override-active-pill">
                  ⚠️ {forcedRedApproaches.length} APPROACH{forcedRedApproaches.length > 1 ? 'ES' : ''} FORCED RED
                </span>
              )}
            </div>
          </div>
          <p className="override-subtext">
            Click any direction button below (or click a signal head on the map) to lock that approach to a permanent <strong>RED light</strong> in the simulation sandbox. 
            Vehicles will accumulate in the queue without discharging, testing upstream bottleneck handling.
          </p>

          <div className="override-buttons-grid">
            {APPROACHES.map((app) => {
              const isLocked = forcedRedApproaches.includes(app);
              return (
                <button
                  key={app}
                  type="button"
                  className={`btn-override-approach ${isLocked ? 'locked-red' : ''}`}
                  onClick={() => toggleForcedRedSingle(app)}
                >
                  <span className="override-dot" />
                  <span className="override-label">{app} Approach</span>
                  <span className="override-status-tag font-mono">{isLocked ? 'LOCKED RED' : 'ADAPTIVE'}</span>
                </button>
              );
            })}

            <button
              type="button"
              className={`btn-override-approach all-red ${forcedRedApproaches.length === 4 ? 'locked-red' : ''}`}
              onClick={() => {
                if (forcedRedApproaches.length === 4) {
                  setForcedRedApproaches([]);
                } else {
                  setForcedRedApproaches([...APPROACHES]);
                }
              }}
            >
              <AlertTriangle size={14} />
              <span className="override-label">Emergency All-Red Lock</span>
              <span className="override-status-tag font-mono">{forcedRedApproaches.length === 4 ? 'ALL LOCKED' : 'FREEZE'}</span>
            </button>

            <button
              type="button"
              className={`btn-override-approach ${liveOverride && liveOverride.active ? 'locked-red' : ''}`}
              style={{ border: '1px solid rgba(239, 68, 68, 0.4)', background: liveOverride && liveOverride.active ? 'rgba(239, 68, 68, 0.2)' : 'rgba(239, 68, 68, 0.05)' }}
              onClick={() => setIsLiveModalOpen(true)}
              title="Police Command: Enforce live physical traffic signal change"
            >
              <Radio size={14} className={liveOverride && liveOverride.active ? "text-red animate-pulse" : "text-red"} />
              <span className="override-label">Police Live Signal Light</span>
              <span className="override-status-tag font-mono" style={{ color: '#ef4444' }}>
                {liveOverride && liveOverride.active ? 'OVERRIDE ON' : 'MANUAL'}
              </span>
            </button>
          </div>
        </div>
      )}

      {/* =========================================================================
         SECTION 2: 4-WAY STAR & MULTI-JUNCTION TOPOLOGY CONFIGURATOR (NETWORK MODE)
         ========================================================================= */}
      {simMode === 'NETWORK_STAR' && (
        <div className="corridor-stacker-container">
          <div className="corridor-stacker-header">
            <div>
              <div className="override-title">
                <Layers size={16} className="text-cyan" />
                <strong>4-Way Multi-Junction Star &amp; Network Matrix Topology</strong>
              </div>
              <p className="override-subtext">
                Select a <strong>Central Junction</strong> and attach connected surrounding junctions to its <strong>North, South, East, and West</strong> sides. 
                Discharged vehicle waves from any approach propagate across the physical road network and enter downstream queues!
              </p>
            </div>

            <div className="star-quick-actions">
              <button 
                type="button"
                className="btn-quick-auto-5"
                onClick={handleSelectAll5Junctions}
                title="Automatically configure all 5 monitored Delhi city junctions"
              >
                <Sparkles size={14} /> Select All 5 Monitored Junctions
              </button>
            </div>
          </div>

          {/* 4-DIRECTIONAL SURROUNDING JUNCTION SELECTORS */}
          <div className="directional-topology-grid">
            {/* CENTRAL JUNCTION SELECTOR */}
            <div className="central-junc-selector-box">
              <span className="topo-dir-badge central">🏛️ Central Hub Junction</span>
              <select 
                className="form-select font-bold"
                value={centralJunctionId}
                onChange={(e) => setCentralJunctionId(e.target.value)}
              >
                {junctions.map(j => (
                  <option key={j.junction_id} value={j.junction_id}>
                    {j.junction_id} — {j.name}
                  </option>
                ))}
              </select>
            </div>

            {/* 4 DIRECTIONAL SURROUND SLOTS */}
            {APPROACHES.map(dir => {
              const cfg = DIR_CONFIG[dir];
              const Icon = cfg.icon;
              const currentNeighbor = directionalNeighbors[dir];
              const availableForDir = junctions.filter(j => j.junction_id !== centralJunctionId);

              return (
                <div key={dir} className="topo-slot-card" style={{ borderColor: cfg.border, background: cfg.bg }}>
                  <div className="slot-top-row">
                    <span className="topo-dir-badge" style={{ color: cfg.color }}>
                      <Icon size={13} /> {cfg.label} Side
                    </span>
                    {currentNeighbor && (
                      <button 
                        type="button" 
                        className="btn-clear-slot" 
                        onClick={() => handleSetDirectionNeighbor(dir, null)}
                        title={`Disconnect ${dir} junction`}
                      >
                        <Trash2 size={11} />
                      </button>
                    )}
                  </div>

                  <select
                    className="form-select compact-select"
                    value={currentNeighbor || ''}
                    onChange={(e) => handleSetDirectionNeighbor(dir, e.target.value)}
                  >
                    <option value="">+ Add {cfg.label} Neighbor...</option>
                    {availableForDir.map(j => (
                      <option key={j.junction_id} value={j.junction_id}>
                        {j.junction_id} — {j.name}
                      </option>
                    ))}
                  </select>
                </div>
              );
            })}
          </div>

          {/* ACTIVE MONITORED NETWORK COUNTER */}
          <div className="network-summary-bar">
            <div className="net-stat">
              <span>Active Network Size:</span>
              <strong className="text-cyan font-mono">{allNetworkJunctionIds.length} Junctions Monitored</strong>
            </div>
            <div className="net-stat">
              <span>Connected Corridor Links:</span>
              <strong className="text-emerald font-mono">
                {Object.values(directionalNeighbors).filter(Boolean).length} Bidirectional Corridors
              </strong>
            </div>
          </div>
        </div>
      )}

      {/* =========================================================================
         SIMULATION LAUNCH & DECISION STRIP
         ========================================================================= */}
      <div className="signal-decision">
        <div>
          <span className="metric-lbl">Target Topology</span>
          <strong>
            {simMode === 'SINGLE' 
              ? `${junctionId} Single Intersection` 
              : `4-Way Star Network (${allNetworkJunctionIds.join(' + ')})`}
          </strong>
        </div>

        {simMode === 'SINGLE' ? (
          <div>
            <span className="metric-lbl"><Clock3 size={13} /> Recommended Next Phase</span>
            <strong>{phaseLabel} ({recommendation?.green_duration_seconds ?? 30}s)</strong>
          </div>
        ) : (
          <div>
            <span className="metric-lbl"><Zap size={13} /> Simultaneous Propagation</span>
            <strong>{allNetworkJunctionIds.length * 4} Synchronized Sensor Approaches</strong>
          </div>
        )}

        <button 
          className="btn-primary btn-launch-sim" 
          disabled={loading} 
          onClick={handleRunSimulation}
        >
          <Play size={15} /> {loading ? 'Computing Simulation…' : simMode === 'SINGLE' ? 'Run Adaptive Simulation' : `Run Full Network Simulation (${allNetworkJunctionIds.length} Junctions)`}
        </button>
      </div>

      {/* =========================================================================
         VIEW A: ANIMATED SINGLE JUNCTION VISUAL SIMULATION
         ========================================================================= */}
      {simMode === 'SINGLE' && sim && step && (
        <div className="sim-stage" ref={stageRef}>
          {sim.seeded_demo && (
            <div className="traffic-alert info sim-demo-note">
              <AlertTriangle size={14} /> No processed video feeds yet — showing calibrated synthetic baseline observations.
            </div>
          )}

          {forcedRedApproaches.length > 0 && (
            <div className="traffic-alert critical sim-demo-note">
              <ShieldAlert size={15} /> Manual Red Light Lock active on: <strong>{forcedRedApproaches.join(', ')}</strong>. Approaching vehicles cannot clear.
            </div>
          )}

          <div className="sim-viz">
            <div className="sim-cross">
              <div className="sim-cell north">
                <SignalHead 
                  approach="NORTH" 
                  color={step.lights.NORTH} 
                  isForcedRed={forcedRedApproaches.includes('NORTH')}
                  onToggleForcedRed={() => toggleForcedRedSingle('NORTH')}
                />
              </div>
              <div className="sim-cell west">
                <SignalHead 
                  approach="WEST" 
                  color={step.lights.WEST} 
                  isForcedRed={forcedRedApproaches.includes('WEST')}
                  onToggleForcedRed={() => toggleForcedRedSingle('WEST')}
                />
              </div>
              <div className="sim-cell center">
                <span className="sim-phase">{step.phase_label}</span>
                <span className="sim-clock"><Clock3 size={12} /> {step.t}s / {sim.total_seconds}s</span>
                <span className="sim-countdown">{step.phase_time_remaining}s left</span>
              </div>
              <div className="sim-cell east">
                <SignalHead 
                  approach="EAST" 
                  color={step.lights.EAST} 
                  isForcedRed={forcedRedApproaches.includes('EAST')}
                  onToggleForcedRed={() => toggleForcedRedSingle('EAST')}
                />
              </div>
              <div className="sim-cell south">
                <SignalHead 
                  approach="SOUTH" 
                  color={step.lights.SOUTH} 
                  isForcedRed={forcedRedApproaches.includes('SOUTH')}
                  onToggleForcedRed={() => toggleForcedRedSingle('SOUTH')}
                />
              </div>
            </div>

            <div className="sim-lanes">
              {APPROACHES.map((a) => {
                const isLocked = forcedRedApproaches.includes(a);
                return (
                  <div className={`sim-lane ${isLocked ? 'lane-locked' : ''}`} key={a}>
                    <span className="sim-lane-name">
                      {a} {isLocked && <span className="text-red font-bold">🔴</span>}
                    </span>
                    <div className="sim-queue-track">
                      <div
                        className={`sim-queue-bar ${step.lights[a] === 'GREEN' && !isLocked ? 'flowing' : ''} ${isLocked ? 'locked-bar' : ''}`}
                        style={{ width: `${Math.min(100, (step.queues[a] / queueScale) * 100)}%` }}
                      />
                    </div>
                    <span className="sim-lane-count font-mono">{step.queues[a]} veh</span>
                  </div>
                );
              })}
              <div className="sim-served">
                <Car size={14} /> Total Vehicles Cleared: <strong className="font-mono">{step.served_total}</strong>
              </div>
            </div>
          </div>

          {/* PLAYBACK CONTROLS */}
          <div className="sim-controls">
            <button className="sim-btn" onClick={togglePlay}>
              {playing ? <Pause size={15} /> : <Play size={15} />} {playing ? 'Pause' : atEnd ? 'Play again' : 'Play'}
            </button>
            <button className="sim-btn ghost" onClick={replay}><RotateCcw size={14} /> Replay</button>
            <div className="sim-speed">
              <Gauge size={13} />
              {[1, 2, 4].map((s) => (
                <button key={s} className={`sim-speed-btn ${speed === s ? 'active' : ''}`} onClick={() => setSpeed(s)}>{s}×</button>
              ))}
            </div>
            <div className="sim-progress-track">
              <div className="sim-progress-fill" style={{ width: `${(stepIndex / (sim.steps.length - 1)) * 100}%` }} />
            </div>
          </div>

          {/* OUTCOME SCOREBOARD */}
          <div className="sim-summary">
            <div className="sim-summary-head">
              <span>Adaptive Control vs. Naive 30s Fixed Timer</span>
              <span className={`sim-improve ${improved ? 'good' : 'bad'}`}>
                {improved ? <TrendingDown size={15} /> : <TrendingUp size={15} />}
                {Math.abs(comparison.improvement_pct)}% {improved ? 'less waiting' : 'more waiting'}
              </span>
            </div>
            <div className="sim-compare">
              <div className="sim-compare-row">
                <span className="sim-compare-lbl">Adaptive</span>
                <div className="sim-compare-track">
                  <div className="sim-compare-bar adaptive" style={{ width: `${(comparison.adaptive_avg_wait / waitScale) * 100}%` }} />
                </div>
                <span className="sim-compare-val">{comparison.adaptive_avg_wait}s avg wait</span>
              </div>
              <div className="sim-compare-row">
                <span className="sim-compare-lbl">Fixed timer</span>
                <div className="sim-compare-track">
                  <div className="sim-compare-bar fixed" style={{ width: `${(comparison.fixed_avg_wait / waitScale) * 100}%` }} />
                </div>
                <span className="sim-compare-val">{comparison.fixed_avg_wait}s avg wait</span>
              </div>
            </div>

            <div className="sim-served-compare">
              Vehicles cleared — adaptive <strong className="text-emerald">{comparison.adaptive_served}</strong> · fixed <strong className="text-muted">{comparison.fixed_served}</strong>
            </div>

            <table className="sim-table">
              <thead>
                <tr><th>Approach</th><th>Arrived</th><th>Cleared</th><th>Peak Queue</th><th>Avg Wait</th><th>Manual Override</th></tr>
              </thead>
              <tbody>
                {sim.per_approach.map((p) => (
                  <tr key={p.approach}>
                    <td><strong>{p.approach}</strong></td>
                    <td>{p.arrivals}</td>
                    <td>{p.served}</td>
                    <td>{p.max_queue}</td>
                    <td>{p.avg_wait}s</td>
                    <td>
                      {forcedRedApproaches.includes(p.approach) ? (
                        <span className="badge-critical font-mono font-bold">LOCKED RED</span>
                      ) : (
                        <span className="badge-minor font-mono">NORMAL</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* =========================================================================
         VIEW B: ANIMATED 4-WAY STAR & CITY MATRIX MULTI-JUNCTION CANVAS
         ========================================================================= */}
      {simMode === 'NETWORK_STAR' && corridorSim && corridorStep && (
        <div className="sim-stage" ref={stageRef}>
          {/* NETWORK TELEMETRY BANNER */}
          <div className="corridor-telemetry-banner">
            <div className="telemetry-pill">
              <Zap size={14} className="text-amber" />
              <span>Corridor In-Transit Vehicles: <strong>{corridorStep.transit?.reduce((acc, p) => acc + p.vehicles_in_transit, 0) || 0}</strong></span>
            </div>
            <div className="telemetry-pill">
              <CheckCircle2 size={14} className="text-emerald" />
              <span>Total Network Discharges: <strong>{corridorStep.corridor_served_total} veh</strong></span>
            </div>
            <div className="telemetry-pill">
              <Clock3 size={14} className="text-cyan" />
              <span>Simulated Time: <strong>{corridorStep.t}s / {corridorSim.total_seconds}s</strong></span>
            </div>
          </div>

          {/* 4-WAY COMPASS STAR NETWORK CANVAS */}
          <div className="star-network-canvas">
            {/* ROW 1: NORTH SURROUNDING JUNCTION */}
            <div className="star-row north-row">
              {directionalNeighbors.NORTH ? (
                <>
                  {renderJunctionNodeCard(directionalNeighbors.NORTH, 'North Corridor')}
                  {renderPlatoonsOnLink(centralJunctionId, directionalNeighbors.NORTH, 'vertical')}
                </>
              ) : (
                <div className="star-slot-placeholder">
                  <span>No North Junction Connected</span>
                </div>
              )}
            </div>

            {/* ROW 2: WEST JUNCTION ➔ CENTRAL JUNCTION ➔ EAST JUNCTION */}
            <div className="star-row center-row">
              {/* WEST */}
              {directionalNeighbors.WEST ? (
                renderJunctionNodeCard(directionalNeighbors.WEST, 'West Corridor')
              ) : (
                <div className="star-slot-placeholder mini">
                  <span>No West Link</span>
                </div>
              )}

              {/* WEST LINK PIPE */}
              {directionalNeighbors.WEST && renderPlatoonsOnLink(centralJunctionId, directionalNeighbors.WEST, 'horizontal')}

              {/* CENTRAL JUNCTION NODE */}
              {renderJunctionNodeCard(centralJunctionId, '🏛️ Central Matrix Hub')}

              {/* EAST LINK PIPE */}
              {directionalNeighbors.EAST && renderPlatoonsOnLink(centralJunctionId, directionalNeighbors.EAST, 'horizontal')}

              {/* EAST */}
              {directionalNeighbors.EAST ? (
                renderJunctionNodeCard(directionalNeighbors.EAST, 'East Corridor')
              ) : (
                <div className="star-slot-placeholder mini">
                  <span>No East Link</span>
                </div>
              )}
            </div>

            {/* ROW 3: SOUTH SURROUNDING JUNCTION */}
            <div className="star-row south-row">
              {directionalNeighbors.SOUTH ? (
                <>
                  {renderPlatoonsOnLink(centralJunctionId, directionalNeighbors.SOUTH, 'vertical')}
                  {renderJunctionNodeCard(directionalNeighbors.SOUTH, 'South Corridor')}
                </>
              ) : (
                <div className="star-slot-placeholder">
                  <span>No South Junction Connected</span>
                </div>
              )}
            </div>
          </div>

          {/* PLAYBACK CONTROLS */}
          <div className="sim-controls">
            <button className="sim-btn" onClick={togglePlay}>
              {playing ? <Pause size={15} /> : <Play size={15} />} {playing ? 'Pause' : atEnd ? 'Play again' : 'Play'}
            </button>
            <button className="sim-btn ghost" onClick={replay}><RotateCcw size={14} /> Replay</button>
            <div className="sim-speed">
              <Gauge size={13} />
              {[1, 2, 4].map((s) => (
                <button key={s} className={`sim-speed-btn ${speed === s ? 'active' : ''}`} onClick={() => setSpeed(s)}>{s}×</button>
              ))}
            </div>
            <div className="sim-progress-track">
              <div className="sim-progress-fill" style={{ width: `${(stepIndex / (corridorSim.steps.length - 1)) * 100}%` }} />
            </div>
          </div>

          {/* NETWORK SCOREBOARD */}
          <div className="sim-summary">
            <div className="sim-summary-head">
              <span>Coordinated Multi-Junction City Matrix vs. Uncoordinated Timers</span>
              <span className={`sim-improve ${corridorImproved ? 'good' : 'bad'}`}>
                {corridorImproved ? <TrendingDown size={15} /> : <TrendingUp size={15} />}
                {Math.abs(corridorComp.improvement_pct)}% {corridorImproved ? 'less city-wide delay' : 'more city-wide delay'}
              </span>
            </div>

            <div className="sim-compare">
              <div className="sim-compare-row">
                <span className="sim-compare-lbl">Network Adaptive</span>
                <div className="sim-compare-track">
                  <div className="sim-compare-bar adaptive" style={{ width: `${(corridorComp.adaptive_avg_wait / corridorWaitScale) * 100}%` }} />
                </div>
                <span className="sim-compare-val">{corridorComp.adaptive_avg_wait}s avg delay</span>
              </div>
              <div className="sim-compare-row">
                <span className="sim-compare-lbl">Uncoordinated Fixed</span>
                <div className="sim-compare-track">
                  <div className="sim-compare-bar fixed" style={{ width: `${(corridorComp.fixed_avg_wait / corridorWaitScale) * 100}%` }} />
                </div>
                <span className="sim-compare-val">{corridorComp.fixed_avg_wait}s avg delay</span>
              </div>
            </div>

            <div className="sim-served-compare">
              Cross-Corridor Dynamic Handoffs: <strong className="text-cyan">{corridorSim.corridor_handoff_count} veh</strong> · Total Network Throughput: <strong className="text-emerald">{corridorComp.adaptive_served} veh</strong>
            </div>
          </div>
        </div>
      )}

      {/* RATIONALE & ALERT LOGS */}
      <p className="help-text">
        {simMode === 'SINGLE' 
          ? (sim?.rationale || recommendation?.rationale || 'Ready to run single intersection simulation.')
          : (corridorSim?.rationale || 'Ready to run 4-way multi-junction city network simulation.')}
      </p>

      {/* LIVE PHYSICAL SIGNAL OVERRIDE MODAL */}
      <ManualSignalModal
        isOpen={isLiveModalOpen}
        junctionId={activeTargetJunctionId}
        currentOverride={liveOverride}
        onClose={() => setIsLiveModalOpen(false)}
        onOverrideApplied={handleOverrideApplied}
      />
    </section>
  );
}
