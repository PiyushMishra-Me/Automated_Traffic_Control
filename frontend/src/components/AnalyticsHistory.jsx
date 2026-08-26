import React, { useEffect, useState } from 'react';
import { BarChart3, Loader2 } from 'lucide-react';
import { api } from '../services/api';

const APPROACHES = ['', 'NORTH', 'SOUTH', 'EAST', 'WEST'];

export default function AnalyticsHistory({ junctionId, refreshKey }) {
  const [approach, setApproach] = useState('');
  const [summary, setSummary] = useState(null);
  const [history, setHistory] = useState([]);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!junctionId) return;
    let active = true;
    setError(null);
    Promise.all([api.getAnalyticsSummary(junctionId, approach || null), api.getAnalyticsHistory(junctionId, approach || null, 8)])
      .then(([nextSummary, nextHistory]) => active && (setSummary(nextSummary), setHistory(nextHistory)))
      .catch((err) => active && setError(err.message));
    return () => { active = false; };
  }, [junctionId, approach, refreshKey]);

  const peak = Math.max(1, ...history.map((item) => item.vehicle_count || 0));
  return <section className="panel-card analytics-history">
    <div className="panel-header"><h2><BarChart3 size={18} color="#38bdf8" /> Historical Analytics</h2><select className="form-select compact-select" value={approach} onChange={(e) => setApproach(e.target.value)}>{APPROACHES.map((item) => <option key={item} value={item}>{item || 'All approaches'}</option>)}</select></div>
    {error ? <p className="form-error">{error}</p> : !summary ? <Loader2 className="animate-spin" size={20} /> : <><div className="summary-grid"><Metric label="Observations" value={summary.observations} /><Metric label="Avg. vehicles" value={summary.average_vehicle_count} /><Metric label="Peak vehicles" value={summary.peak_vehicle_count} /><Metric label="Avg. density" value={Number(summary.average_density).toFixed(2)} /><Metric label="Avg. queue" value={summary.average_queue_length} /><Metric label="Latest flow" value={Math.round(summary.latest_flow)} /></div><div className="history-chart">{history.length === 0 ? <p>No completed video observations yet.</p> : history.slice().reverse().map((item, index) => <div className="history-bar-item" key={`${item.timestamp}-${index}`} title={`${item.approach}: ${item.vehicle_count} active vehicles`}><div className="history-bar" style={{ height: `${Math.max(8, (item.vehicle_count / peak) * 100)}%` }} /><span>{item.approach?.slice(0, 1)}</span></div>)}</div><p className="help-text">Shows completed video observations, not a live traffic feed.</p></>}
  </section>;
}

function Metric({ label, value }) { return <div className="summary-metric"><span>{label}</span><strong>{value}</strong></div>; }
