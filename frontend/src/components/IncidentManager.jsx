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
  Plus
} from 'lucide-react';
import { api } from '../services/api';

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
                  </div>

                  <div className="incident-mid">
                    <span className="road-title">{inc.road_name}</span>
                    <span className="desc-preview">{inc.description}</span>
                  </div>

                  <div className="incident-right">
                    <span className="time-badge">
                      <Clock size={13} /> ~{inc.estimated_clearance_minutes}m est.
                    </span>
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

                {isExpanded && plan && (
                  <div className="incident-expanded-details">
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
    </div>
  );
}
