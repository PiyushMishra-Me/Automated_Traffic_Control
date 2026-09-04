import React, { useState } from 'react';
import { 
  ShieldAlert, 
  AlertTriangle, 
  CheckCircle2, 
  Clock, 
  Navigation, 
  MapPin, 
  ExternalLink,
  ChevronDown,
  ChevronUp,
  RefreshCw,
  Plus,
  Radio,
  Zap
} from 'lucide-react';
import { api } from '../services/api';
import ManualSignalModal from './ManualSignalModal';

export default function IncidentManager({ 
  incidents = [], 
  onRefresh, 
  onOpenReportModal,
  selectedJunction,
  onSelectJunction
}) {
  const [filterJunctionOnly, setFilterJunctionOnly] = useState(false);
  const [expandedId, setExpandedId] = useState(null);
  const [resolvingId, setResolvingId] = useState(null);
  const [selectedIncidentForSignal, setSelectedIncidentForSignal] = useState(null);
  const [activeOverrides, setActiveOverrides] = useState({});

  const fetchOverrides = async () => {
    try {
      const data = await api.listActiveSignalOverrides();
      setActiveOverrides(data || {});
    } catch (e) {
      console.warn('Failed to load active overrides', e);
    }
  };

  React.useEffect(() => {
    fetchOverrides();
    const interval = setInterval(fetchOverrides, 5000);
    return () => clearInterval(interval);
  }, []);

  const displayedIncidents = incidents.filter(inc => {
    if (filterJunctionOnly && inc.junction_id !== selectedJunction) return false;
    return true;
  });

  const activeIncidents = displayedIncidents.filter(i => i.status === 'ACTIVE');
  const resolvedIncidents = displayedIncidents.filter(i => i.status === 'RESOLVED');

  const handleResolve = async (incidentId) => {
    setResolvingId(incidentId);
    try {
      await api.updateIncidentStatus(incidentId, 'RESOLVED');
      if (onRefresh) onRefresh();
    } catch (err) {
      console.error('Failed to resolve incident', err);
    } finally {
      setResolvingId(null);
    }
  };

  const toggleExpand = (id) => {
    setExpandedId(prev => prev === id ? null : id);
  };

  const getSeverityBadgeClass = (sev) => {
    switch (sev) {
      case 'CRITICAL_ROAD_BLOCKED': return 'badge-critical';
      case 'SEVERE': return 'badge-severe';
      case 'MODERATE': return 'badge-moderate';
      default: return 'badge-minor';
    }
  };

  return (
    <div className="incident-manager-card">
      <div className="card-header">
        <div className="title-with-icon">
          <ShieldAlert className="text-amber" size={20} />
          <div>
            <h3>Active Incidents & Upstream Traffic Diversion Hub</h3>
            <p className="card-subtitle">Real-time road obstruction containment, automated signal throttling, & detour clearance</p>
          </div>
        </div>

        <div className="incident-controls">
          <label className="checkbox-label">
            <input 
              type="checkbox" 
              checked={filterJunctionOnly} 
              onChange={(e) => setFilterJunctionOnly(e.target.checked)} 
            />
            <span>Filter for {selectedJunction} Only</span>
          </label>

          <button className="btn-secondary btn-sm" onClick={onRefresh} title="Refresh Incidents">
            <RefreshCw size={14} /> Refresh
          </button>

          <button className="btn-primary btn-sm" onClick={onOpenReportModal}>
            <Plus size={14} /> Report Incident
          </button>
        </div>
      </div>

      {activeIncidents.length === 0 ? (
        <div className="empty-incidents-state">
          <CheckCircle2 size={36} className="text-green opacity-70" />
          <h4>No Active Incidents Reported</h4>
          <p>All monitored road corridors and approaches are operating normally without detour diversions.</p>
        </div>
      ) : (
        <div className="incidents-list">
          {activeIncidents.map((inc) => {
            const isExpanded = expandedId === inc.incident_id;
            const plan = inc.diversion_plan;
            const activeOverride = activeOverrides[inc.junction_id];

            return (
              <div key={inc.incident_id} className="incident-card active-border">
                <div className="incident-card-top" onClick={() => toggleExpand(inc.incident_id)}>
                  <div className="incident-left">
                    <span className={`severity-chip ${getSeverityBadgeClass(inc.severity)}`}>
                      {inc.severity.replace('_', ' ')}
                    </span>
                    <span className="incident-junction-tag">
                      <MapPin size={13} /> {inc.junction_id} • {inc.approach}
                    </span>
                    <span className="incident-type-tag">{inc.incident_type.replace('_', ' ')}</span>
                    {activeOverride && activeOverride.active && (
                      <span className="override-incident-tag font-mono">
                        🚨 {activeOverride.override_mode === 'EMERGENCY_ALL_RED' 
                          ? 'ALL-RED FREEZE' 
                          : activeOverride.override_mode === 'HOLD_RED_APPROACH' 
                          ? `${activeOverride.forced_red_approaches?.join(', ') || inc.approach} LOCKED RED` 
                          : `${activeOverride.phase || 'PRIORITY'} GREEN`}
                      </span>
                    )}
                  </div>

                  <div className="incident-mid">
                    <span className="road-title">{inc.road_name}</span>
                    <span className="desc-preview">{inc.description}</span>
                  </div>

                  <div className="incident-right">
                    <span className="time-badge">
                      <Clock size={13} /> ~{inc.estimated_clearance_minutes}m est.
                    </span>

                    {/* BUTTON: MANUALLY CHANGE SIGNAL LIGHT FOR THIS EMERGENCY */}
                    <button 
                      type="button"
                      className={`btn-override-signal ${activeOverride && activeOverride.active ? 'active' : ''}`}
                      onClick={(e) => {
                        e.stopPropagation();
                        setSelectedIncidentForSignal(inc);
                      }}
                      title="Traffic Police: Manually change traffic light for this emergency"
                    >
                      <Radio size={13} className={activeOverride && activeOverride.active ? "animate-pulse" : ""} />
                      {activeOverride && activeOverride.active ? 'Signal Override Active' : 'Change Signal Light'}
                    </button>

                    <button 
                      className="btn-resolve"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleResolve(inc.incident_id);
                      }}
                      disabled={resolvingId === inc.incident_id}
                    >
                      {resolvingId === inc.incident_id ? 'Resolving...' : '✓ Resolve Incident'}
                    </button>
                    <button className="btn-icon-toggle">
                      {isExpanded ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
                    </button>
                  </div>
                </div>

                {isExpanded && (
                  <div className="incident-expanded-details">
                    {/* EMERGENCY POLICE SIGNAL DIRECTIVE BAR */}
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '10px', padding: '12px 16px', background: activeOverride && activeOverride.active ? 'rgba(239, 68, 68, 0.08)' : 'rgba(248, 250, 252, 0.9)', border: activeOverride && activeOverride.active ? '1px solid #fca5a5' : '1px solid #e2e8f0', borderRadius: '10px', marginBottom: '12px' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                        <div style={{ width: '32px', height: '32px', borderRadius: '8px', background: activeOverride && activeOverride.active ? 'rgba(239, 68, 68, 0.2)' : 'rgba(15, 23, 42, 0.06)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                          <Radio size={16} className={activeOverride && activeOverride.active ? "text-red animate-pulse" : "text-slate"} />
                        </div>
                        <div>
                          <strong style={{ fontSize: '0.82rem', color: activeOverride && activeOverride.active ? '#b91c1c' : '#1e293b' }}>
                            {activeOverride && activeOverride.active 
                              ? `Manual Signal Directive Active on ${inc.junction_id}` 
                              : `Manual Signal Control Available for ${inc.junction_id}`}
                          </strong>
                          <div style={{ fontSize: '0.72rem', color: '#64748b' }}>
                            {activeOverride && activeOverride.active 
                              ? `Mode: ${activeOverride.override_mode} • ${activeOverride.reason}` 
                              : `Police can lock ${inc.approach} approach to RED or enforce 4-way ALL-RED freeze during emergency response.`}
                          </div>
                        </div>
                      </div>
                      <button
                        type="button"
                        className={`btn-override-signal ${activeOverride && activeOverride.active ? 'active' : ''}`}
                        onClick={() => setSelectedIncidentForSignal(inc)}
                      >
                        <Zap size={13} />
                        {activeOverride && activeOverride.active ? 'Manage Active Signal' : 'Change Signal Light'}
                      </button>
                    </div>
                    <div className="diversion-summary-box">
                      <div className="diversion-summary-header">
                        <Navigation size={16} className="text-cyan" />
                        <strong>Automated Upstream Traffic Diversion Strategy</strong>
                      </div>
                      <div className="strategy-text">{plan.signal_timing_strategy}</div>
                      
                      <div className="diversion-steps-container">
                        <h6>Traffic Control Center Actions:</h6>
                        <div className="steps-timeline">
                          {plan.steps.map((st) => (
                            <div key={st.step_number} className="timeline-step">
                              <div className="step-num">{st.step_number}</div>
                              <div className="step-content">
                                <div className="step-instruction">{st.instruction}</div>
                                <div className="step-action-tag font-mono">🔧 {st.signal_action}</div>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>

                    {inc.photo_base64 && (
                      <div className="incident-photo-evidence-container">
                        <div className="evidence-header">
                          <span className="badge-verified-onscene">📸 Verified On-Scene Live Capture</span>
                          <span className="evidence-time">{inc.capture_timestamp ? new Date(inc.capture_timestamp).toLocaleString() : ''}</span>
                        </div>
                        <img src={inc.photo_base64} alt="Live Incident Evidence" className="incident-evidence-photo" />
                      </div>
                    )}

                    <div className="reporter-meta">
                      Reported by: <strong>{inc.reported_by}</strong> • Timestamp: {new Date(inc.reported_at).toLocaleTimeString()}
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {resolvedIncidents.length > 0 && (
        <div className="resolved-section">
          <h5 className="resolved-heading">Recent Resolved Incidents ({resolvedIncidents.length})</h5>
          <div className="resolved-list">
            {resolvedIncidents.slice(0, 3).map((r) => (
              <div key={r.incident_id} className="resolved-item">
                <span className="text-green">✓ {r.incident_id}</span>
                <span>{r.junction_id} ({r.approach}) — {r.road_name}</span>
                <span className="text-muted">{new Date(r.updated_at).toLocaleTimeString()}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* EMERGENCY MANUAL SIGNAL LIGHT OVERRIDE MODAL */}
      <ManualSignalModal
        isOpen={Boolean(selectedIncidentForSignal)}
        incident={selectedIncidentForSignal}
        currentOverride={selectedIncidentForSignal ? activeOverrides[selectedIncidentForSignal.junction_id] : null}
        onClose={() => setSelectedIncidentForSignal(null)}
        onOverrideApplied={() => {
          fetchOverrides();
          if (onRefresh) onRefresh();
        }}
      />
    </div>
  );
}
