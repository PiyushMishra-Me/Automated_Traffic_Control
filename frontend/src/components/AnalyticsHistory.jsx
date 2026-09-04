import React, { useEffect, useState } from 'react';
import { BarChart3, Loader2, MapPin, Filter } from 'lucide-react';
import { api } from '../services/api';

const APPROACHES = ['', 'NORTH', 'SOUTH', 'EAST', 'WEST'];

export default function AnalyticsHistory({ 
  junctionId, 
  junctions = [], 
  onSelectJunction, 
  refreshKey 
}) {
  const [approach, setApproach] = useState('');
  const [summary, setSummary] = useState(null);
  const [history, setHistory] = useState([]);
  const [error, setError] = useState(null);
  const [allJunctions, setAllJunctions] = useState(junctions);

  // Sync junctions or fallback fetch if empty
  useEffect(() => {
    if (junctions && junctions.length > 0) {
      setAllJunctions(junctions);
    } else {
      api.listJunctions()
        .then((list) => {
          if (list && list.length > 0) setAllJunctions(list);
        })
        .catch(() => {});
    }
  }, [junctions]);

  // Fetch analytics when junction or approach changes
  useEffect(() => {
    if (!junctionId) return;
    let active = true;
    setError(null);
    Promise.all([
      api.getAnalyticsSummary(junctionId, approach || null),
      api.getAnalyticsHistory(junctionId, approach || null, 8)
    ])
      .then(([nextSummary, nextHistory]) => {
        if (active) {
          setSummary(nextSummary);
          setHistory(nextHistory);
        }
      })
      .catch((err) => {
        if (active) setError(err.message);
      });

    return () => { active = false; };
  }, [junctionId, approach, refreshKey]);

  // Identify current junction metadata
  const currentJunction = (allJunctions || []).find((j) => j.junction_id === junctionId);
  const junctionName = currentJunction?.name || (junctionId ? `Junction ${junctionId}` : 'Connaught Place Outer Circle');
  const junctionLocation = currentJunction?.location || '';

  const peak = Math.max(1, ...history.map((item) => item.vehicle_count || 0));

  return (
    <section className="panel-card analytics-history">
      {/* Panel Header with Prominent Junction Name & Dual Controls */}
      <div className="panel-header analytics-header">
        <div className="analytics-header-title-wrap">
          <h2>
            <BarChart3 size={18} color="#38bdf8" /> 
            <span>Historical Analytics</span>
          </h2>
          
          {/* Prominent Junction Identifier Pill */}
          <div className="analytics-junction-chip" title={`Current Junction: ${junctionId} - ${junctionName}`}>
            <MapPin size={13} className="text-cyan" />
            <strong className="chip-junc-id">{junctionId || 'J-01'}</strong>
            <span className="chip-junc-name">— {junctionName}</span>
            {junctionLocation && <span className="chip-junc-loc">({junctionLocation})</span>}
          </div>
        </div>

        {/* Junction & Approach Selectors */}
        <div className="analytics-controls">
          {allJunctions && allJunctions.length > 0 && (
            <div className="analytics-control-group">
              <label className="analytics-label">
                <MapPin size={12} /> Junction:
              </label>
              <select 
                className="form-select compact-select junction-select" 
                value={junctionId || (allJunctions[0]?.junction_id)} 
                onChange={(e) => onSelectJunction && onSelectJunction(e.target.value)}
              >
                {allJunctions.map((j) => (
                  <option key={j.junction_id} value={j.junction_id}>
                    {j.junction_id} — {j.name} {j.location ? `(${j.location})` : ''}
                  </option>
                ))}
              </select>
            </div>
          )}

          <div className="analytics-control-group">
            <label className="analytics-label">
              <Filter size={12} /> Approach:
            </label>
            <select 
              className="form-select compact-select approach-select" 
              value={approach} 
              onChange={(e) => setApproach(e.target.value)}
            >
              {APPROACHES.map((item) => (
                <option key={item} value={item}>
                  {item ? `${item} Approach` : 'All approaches'}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Junction Telemetry Context Strip */}
      <div className="analytics-context-bar">
        <div className="context-left">
          <span className="context-pulse-indicator" />
          <span className="context-text">
            Telemetry Source: <strong>{junctionId || 'J-01'} — {junctionName}</strong> 
            {junctionLocation ? <span className="context-muted"> ({junctionLocation})</span> : null}
            <span className="context-sep">•</span>
            <span className="context-approach-badge">
              {approach ? `${approach} Directional Approach` : 'Combined 4-Way Approaches'}
            </span>
          </span>
        </div>
        <div className="context-right">
          <span className="context-obs-count">
            {summary ? `${summary.observations} Recorded Observations` : 'Loading metrics...'}
          </span>
        </div>
      </div>

      {error ? (
        <p className="form-error">{error}</p>
      ) : !summary ? (
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '40px 0', gap: '8px', color: 'var(--text-secondary)' }}>
          <Loader2 className="animate-spin" size={20} />
          <span>Loading historical observations for {junctionName}...</span>
        </div>
      ) : (
        <>
          <div className="summary-grid">
            <Metric label="Observations" value={summary.observations} />
            <Metric label="Avg. vehicles" value={summary.average_vehicle_count} />
            <Metric label="Peak vehicles" value={summary.peak_vehicle_count} />
            <Metric label="Avg. density" value={Number(summary.average_density).toFixed(2)} />
            <Metric label="Avg. queue" value={summary.average_queue_length} />
            <Metric label="Latest flow" value={Math.round(summary.latest_flow)} />
          </div>

          <div className="history-chart">
            {history.length === 0 ? (
              <p>No completed video observations yet for {junctionName} ({junctionId}).</p>
            ) : (
              history.slice().reverse().map((item, index) => (
                <div 
                  className="history-bar-item" 
                  key={`${item.timestamp}-${index}`} 
                  title={`${item.approach} Approach at ${junctionName}: ${item.vehicle_count} active vehicles`}
                >
                  <div 
                    className="history-bar" 
                    style={{ height: `${Math.max(8, (item.vehicle_count / peak) * 100)}%` }} 
                  />
                  <span>{item.approach?.slice(0, 1)}</span>
                </div>
              ))
            )}
          </div>

          <p className="help-text">
            Shows completed video observations for <strong>{junctionId} — {junctionName}</strong>, not a live traffic feed.
          </p>
        </>
      )}
    </section>
  );
}

function Metric({ label, value }) { 
  return (
    <div className="summary-metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  ); 
}
