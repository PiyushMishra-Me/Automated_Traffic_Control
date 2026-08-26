import React, { useEffect, useState } from 'react';
import { AlertTriangle, Clock3, Play, RadioTower, ShieldCheck } from 'lucide-react';
import { api } from '../services/api';

export default function SignalSimulator({ junctionId, refreshKey }) {
  const [recommendation, setRecommendation] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const load = async (simulate = false) => {
    if (!junctionId) return;
    setLoading(true); setError(null);
    try { setRecommendation(simulate ? await api.simulateSignal(junctionId) : await api.getSignalRecommendation(junctionId)); }
    catch (err) { setError(err.message); }
    finally { setLoading(false); }
  };
  useEffect(() => { load(); }, [junctionId, refreshKey]);
  const phaseLabel = recommendation?.recommended_phase?.replaceAll('_', ' ') || 'LOADING';
  return <section className="signal-simulator"><div className="panel-header"><h2><RadioTower size={19} color="#38bdf8" /> Adaptive Signal Simulator</h2><span className="simulation-badge"><ShieldCheck size={13} /> Simulation only</span></div>{error ? <p className="form-error">{error}</p> : <><div className="signal-decision"><div><span className="metric-lbl">Recommended next phase</span><strong>{phaseLabel}</strong></div><div><span className="metric-lbl"><Clock3 size={13} /> Green duration</span><strong>{recommendation ? `${recommendation.green_duration_seconds}s` : '—'}</strong></div><button className="btn-primary" disabled={loading} onClick={() => load(true)}><Play size={15} /> Run simulation</button></div><div className="signal-scores"><span>North/South demand <strong>{recommendation?.north_south_score ?? '—'}</strong></span><span>East/West demand <strong>{recommendation?.east_west_score ?? '—'}</strong></span></div><p className="help-text">{recommendation?.rationale || 'Calculating the current recommendation…'}</p><div className="alert-list">{recommendation?.alerts?.map((alert, index) => <div className={`traffic-alert ${alert.severity.toLowerCase()}`} key={index}><AlertTriangle size={15} /> {alert.message}</div>)}</div></>}</section>;
}
