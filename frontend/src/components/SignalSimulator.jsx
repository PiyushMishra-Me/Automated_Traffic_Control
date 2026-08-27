import React, { useEffect, useRef, useState } from 'react';
import {
  AlertTriangle, Car, Clock3, Gauge, Pause, Play, RadioTower,
  RotateCcw, ShieldCheck, TrendingDown, TrendingUp,
} from 'lucide-react';
import { api } from '../services/api';

const APPROACHES = ['NORTH', 'SOUTH', 'EAST', 'WEST'];
const STEP_MS = 90; // wall-clock ms between animation frames (speed advances >1 step per frame)

function SignalHead({ approach, color }) {
  return (
    <div className={`sig-head sig-${approach.toLowerCase()}`}>
      <span className="sig-label">{approach[0]}</span>
      <div className="sig-lamps">
        <i className={`sig-dot red ${color === 'RED' ? 'on' : ''}`} />
        <i className={`sig-dot yellow ${color === 'YELLOW' ? 'on' : ''}`} />
        <i className={`sig-dot green ${color === 'GREEN' ? 'on' : ''}`} />
      </div>
    </div>
  );
}

export default function SignalSimulator({ junctionId, refreshKey }) {
  const [recommendation, setRecommendation] = useState(null);
  const [sim, setSim] = useState(null);
  const [stepIndex, setStepIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(2);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const stageRef = useRef(null);

  // Instantaneous recommendation (shown on load, before any simulation is run).
  const loadRecommendation = async () => {
    if (!junctionId) return;
    setError(null);
    try {
      setRecommendation(await api.getSignalRecommendation(junctionId));
    } catch (err) {
      setError(err.message);
    }
  };

  // Reset everything whenever the junction or a data refresh changes.
  useEffect(() => {
    setSim(null);
    setStepIndex(0);
    setPlaying(false);
    loadRecommendation();
  }, [junctionId, refreshKey]);

  // Drive the animation: advance `speed` steps every STEP_MS while playing.
  useEffect(() => {
    if (!playing || !sim) return undefined;
    const id = setInterval(() => {
      setStepIndex((i) => Math.min(i + speed, sim.steps.length - 1));
    }, STEP_MS);
    return () => clearInterval(id);
  }, [playing, speed, sim]);

  // Stop when we reach the end of the timeline.
  useEffect(() => {
    if (sim && playing && stepIndex >= sim.steps.length - 1) setPlaying(false);
  }, [stepIndex, playing, sim]);

  const runSimulation = async () => {
    if (!junctionId) return;
    setLoading(true);
    setError(null);
    try {
      const result = await api.simulateSignal(junctionId);
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

  const togglePlay = () => {
    if (!sim) return;
    if (!playing && stepIndex >= sim.steps.length - 1) setStepIndex(0);
    setPlaying((p) => !p);
  };

  const replay = () => {
    setStepIndex(0);
    setPlaying(true);
  };

  const phaseLabel = recommendation?.recommended_phase?.replaceAll('_', ' ') || 'LOADING';
  const step = sim ? sim.steps[Math.min(stepIndex, sim.steps.length - 1)] : null;
  const atEnd = sim ? stepIndex >= sim.steps.length - 1 : false;
  const queueScale = sim ? Math.max(6, ...sim.per_approach.map((p) => p.max_queue)) : 6;
  const comparison = sim?.comparison;
  const waitScale = comparison ? Math.max(comparison.adaptive_avg_wait, comparison.fixed_avg_wait, 1) : 1;
  const improved = comparison ? comparison.improvement_pct >= 0 : true;

  return (
    <section className="signal-simulator">
      <div className="panel-header">
        <h2><RadioTower size={19} color="#38bdf8" /> Adaptive Signal Simulator</h2>
        <span className="simulation-badge"><ShieldCheck size={13} /> Simulation only</span>
      </div>

      {error && <p className="form-error">{error}</p>}

      {/* Instantaneous recommendation from the current junction state */}
      <div className="signal-decision">
        <div>
          <span className="metric-lbl">Recommended next phase</span>
          <strong>{phaseLabel}</strong>
        </div>
        <div>
          <span className="metric-lbl"><Clock3 size={13} /> Green duration</span>
          <strong>{recommendation ? `${recommendation.green_duration_seconds}s` : '—'}</strong>
        </div>
        <button className="btn-primary" disabled={loading} onClick={runSimulation}>
          <Play size={15} /> {loading ? 'Running…' : 'Run simulation'}
        </button>
      </div>
      <div className="signal-scores">
        <span>North/South demand <strong>{recommendation?.north_south_score ?? '—'}</strong></span>
        <span>East/West demand <strong>{recommendation?.east_west_score ?? '—'}</strong></span>
      </div>

      {/* Animated time-stepped simulation */}
      {sim && step && (
        <div className="sim-stage" ref={stageRef}>
          {sim.seeded_demo && (
            <div className="traffic-alert info sim-demo-note">
              <AlertTriangle size={14} /> No processed observations yet — showing an illustrative demo scenario.
            </div>
          )}

          <div className="sim-viz">
            {/* 4-way intersection with live signal heads */}
            <div className="sim-cross">
              <div className="sim-cell north"><SignalHead approach="NORTH" color={step.lights.NORTH} /></div>
              <div className="sim-cell west"><SignalHead approach="WEST" color={step.lights.WEST} /></div>
              <div className="sim-cell center">
                <span className="sim-phase">{step.phase_label}</span>
                <span className="sim-clock"><Clock3 size={12} /> {step.t}s / {sim.total_seconds}s</span>
                <span className="sim-countdown">{step.phase_time_remaining}s left</span>
              </div>
              <div className="sim-cell east"><SignalHead approach="EAST" color={step.lights.EAST} /></div>
              <div className="sim-cell south"><SignalHead approach="SOUTH" color={step.lights.SOUTH} /></div>
            </div>

            {/* Per-approach queues draining in real time */}
            <div className="sim-lanes">
              {APPROACHES.map((a) => (
                <div className="sim-lane" key={a}>
                  <span className="sim-lane-name">{a}</span>
                  <div className="sim-queue-track">
                    <div
                      className={`sim-queue-bar ${step.lights[a] === 'GREEN' ? 'flowing' : ''}`}
                      style={{ width: `${Math.min(100, (step.queues[a] / queueScale) * 100)}%` }}
                    />
                  </div>
                  <span className="sim-lane-count">{step.queues[a]}</span>
                </div>
              ))}
              <div className="sim-served"><Car size={14} /> Vehicles cleared: <strong>{step.served_total}</strong></div>
            </div>
          </div>

          {/* Playback controls */}
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

          {/* Outcome scoreboard: adaptive vs fixed-timer baseline */}
          <div className="sim-summary">
            <div className="sim-summary-head">
              <span>Adaptive control vs. fixed {`30s`} timer</span>
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
              Vehicles cleared — adaptive <strong>{comparison.adaptive_served}</strong> · fixed <strong>{comparison.fixed_served}</strong>
            </div>

            <table className="sim-table">
              <thead>
                <tr><th>Approach</th><th>Arrived</th><th>Cleared</th><th>Peak queue</th><th>Avg wait</th></tr>
              </thead>
              <tbody>
                {sim.per_approach.map((p) => (
                  <tr key={p.approach}>
                    <td>{p.approach}</td><td>{p.arrivals}</td><td>{p.served}</td><td>{p.max_queue}</td><td>{p.avg_wait}s</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      <p className="help-text">{sim?.rationale || recommendation?.rationale || 'Calculating the current recommendation…'}</p>

      <div className="alert-list">
        {recommendation?.alerts?.map((alert, index) => (
          <div className={`traffic-alert ${alert.severity.toLowerCase()}`} key={index}>
            <AlertTriangle size={15} /> {alert.message}
          </div>
        ))}
      </div>
    </section>
  );
}
