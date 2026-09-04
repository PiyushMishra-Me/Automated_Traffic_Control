import React, { useState, useEffect } from 'react';
import { 
  ShieldAlert, 
  AlertTriangle, 
  CheckCircle2, 
  Clock, 
  X, 
  Zap, 
  RotateCcw, 
  Radio, 
  ArrowRight,
  Sparkles,
  MapPin,
  ShieldCheck
} from 'lucide-react';
import { api } from '../services/api';

export default function ManualSignalModal({ 
  isOpen, 
  incident = null, 
  junctionId: propJunctionId = null,
  defaultApproach = 'NORTH',
  currentOverride,
  onClose, 
  onOverrideApplied 
}) {
  if (!isOpen) return null;
  if (!incident && !propJunctionId) return null;

  const junctionId = incident?.junction_id || propJunctionId || 'J-01';
  const [selectedApproach, setSelectedApproach] = useState(incident?.approach || defaultApproach || 'NORTH');
  const [selectedCorridor, setSelectedCorridor] = useState('NORTH_SOUTH');
  const approach = incident?.approach || selectedApproach;

  // Preset modes: 'HOLD_RED_APPROACH', 'EMERGENCY_ALL_RED', 'PRIORITY_GREEN', 'RESTORE_ADAPTIVE'
  const [selectedMode, setSelectedMode] = useState('HOLD_RED_APPROACH');
  const [durationSeconds, setDurationSeconds] = useState(180);
  const [customReason, setCustomReason] = useState(
    incident 
      ? `Police Emergency Override: ${incident.incident_type} on ${approach} approach (${incident.road_name})`
      : `Police Manual Signal Directive for ${junctionId}`
  );
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [successMsg, setSuccessMsg] = useState(null);

  useEffect(() => {
    if (incident) {
      setSelectedApproach(incident.approach || 'NORTH');
      setCustomReason(`Police Emergency Override: ${incident.incident_type} on ${incident.approach || 'NORTH'} approach (${incident.road_name})`);
    } else {
      setCustomReason(`Police Manual Signal Directive for ${junctionId}`);
    }
    setError(null);
    setSuccessMsg(null);
  }, [incident, junctionId]);

  const handleApplyOverride = async (e) => {
    if (e) e.preventDefault();
    setLoading(true);
    setError(null);
    setSuccessMsg(null);

    try {
      if (selectedMode === 'RESTORE_ADAPTIVE') {
        await api.clearManualSignalOverride(junctionId);
        setSuccessMsg(`Automated AI adaptive control successfully restored to ${junctionId}`);
        if (onOverrideApplied) onOverrideApplied(junctionId, null);
        setTimeout(() => {
          onClose();
        }, 1200);
        return;
      }

      let payload = {
        override_mode: selectedMode,
        reason: customReason,
        duration_seconds: parseInt(durationSeconds, 10),
        authorized_by: 'traffic_command',
        incident_id: incident?.incident_id || null,
        forced_red_approaches: []
      };

      if (selectedMode === 'HOLD_RED_APPROACH') {
        payload.forced_red_approaches = [approach];
      } else if (selectedMode === 'EMERGENCY_ALL_RED') {
        payload.phase = 'ALL_RED';
        payload.forced_red_approaches = ['NORTH', 'SOUTH', 'EAST', 'WEST'];
      } else if (selectedMode === 'PRIORITY_GREEN') {
        if (incident) {
          payload.phase = (approach === 'NORTH' || approach === 'SOUTH') 
            ? 'NORTH_SOUTH_GREEN' 
            : 'EAST_WEST_GREEN';
        } else {
          payload.phase = selectedCorridor === 'NORTH_SOUTH' 
            ? 'NORTH_SOUTH_GREEN' 
            : 'EAST_WEST_GREEN';
        }
      }

      const res = await api.setManualSignalOverride(junctionId, payload);
      setSuccessMsg(`Manual signal light directive successfully applied to ${junctionId}!`);
      if (onOverrideApplied) onOverrideApplied(junctionId, res);
      setTimeout(() => {
        onClose();
      }, 1200);
    } catch (err) {
      setError(err.message || 'Failed to apply manual signal change');
    } finally {
      setLoading(false);
    }
  };

  const isNScorridor = incident 
    ? (approach === 'NORTH' || approach === 'SOUTH')
    : (selectedCorridor === 'NORTH_SOUTH');

  return (
    <div className="modal-backdrop animate-fade-in" onClick={onClose}>
      <div 
        className="auth-modal-content animate-slide-up" 
        onClick={(e) => e.stopPropagation()}
        style={{ width: 'min(580px, 100%)', border: '1px solid rgba(239, 68, 68, 0.35)', boxShadow: '0 25px 60px rgba(239, 68, 68, 0.2), 0 10px 30px rgba(0, 0, 0, 0.9)' }}
      >
        {/* MODAL HEADER */}
        <div className="modal-header">
          <div className="title-with-icon">
            <div style={{ padding: '8px', borderRadius: '8px', background: 'rgba(239, 68, 68, 0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <Radio size={22} className="text-red animate-pulse" />
            </div>
            <div>
              <h3 style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                Emergency Manual Signal Override
                <span style={{ fontSize: '0.65rem', padding: '2px 8px', borderRadius: '99px', background: 'rgba(239, 68, 68, 0.2)', color: '#f87171', border: '1px solid rgba(239, 68, 68, 0.4)' }}>
                  POLICE DISPATCH
                </span>
              </h3>
              <p className="card-subtitle">
                Direct manual traffic light override for <strong>{junctionId}</strong>
                {incident ? ` (${approach} Approach)` : ''}
              </p>
            </div>
          </div>
          <button type="button" className="close-btn" onClick={onClose} title="Close Dialog">
            <X size={16} />
          </button>
        </div>

        {/* CONTEXT BANNER */}
        {incident ? (
          <div style={{ background: 'rgba(15, 23, 42, 0.7)', border: '1px solid rgba(255, 255, 255, 0.08)', borderRadius: '10px', padding: '12px 14px', marginBottom: '16px' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '4px' }}>
              <span style={{ fontSize: '0.75rem', fontWeight: 700, color: '#f87171', display: 'flex', alignItems: 'center', gap: '5px' }}>
                <AlertTriangle size={13} /> Reported Emergency: {incident.incident_type?.replace('_', ' ')}
              </span>
              <span style={{ fontSize: '0.72rem', background: '#3b82f6', color: '#fff', padding: '2px 8px', borderRadius: '6px', fontWeight: 800 }}>
                {junctionId} • {approach}
              </span>
            </div>
            <div style={{ fontSize: '0.78rem', color: '#cbd5e1', fontWeight: 600 }}>
              {incident.road_name}
            </div>
            <div style={{ fontSize: '0.72rem', color: '#94a3b8', marginTop: '3px' }}>
              {incident.description}
            </div>
          </div>
        ) : (
          <div style={{ background: 'rgba(15, 23, 42, 0.7)', border: '1px solid rgba(255, 255, 255, 0.08)', borderRadius: '10px', padding: '12px 14px', marginBottom: '16px' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '4px' }}>
              <span style={{ fontSize: '0.75rem', fontWeight: 700, color: '#38bdf8', display: 'flex', alignItems: 'center', gap: '5px' }}>
                <MapPin size={13} /> Junction Control Console: {junctionId}
              </span>
              <span style={{ fontSize: '0.72rem', background: '#0284c7', color: '#fff', padding: '2px 8px', borderRadius: '6px', fontWeight: 800 }}>
                LIVE TELEMETRY
              </span>
            </div>
            <div style={{ fontSize: '0.74rem', color: '#94a3b8', marginTop: '2px' }}>
              Direct manual physical traffic light override from the Traffic Simulation &amp; Operations Command.
            </div>
          </div>
        )}

        {/* ACTIVE OVERRIDE NOTICE IF ANY */}
        {currentOverride && currentOverride.active && (
          <div style={{ background: 'rgba(245, 158, 11, 0.12)', border: '1px solid rgba(245, 158, 11, 0.3)', borderRadius: '8px', padding: '8px 12px', marginBottom: '14px', fontSize: '0.75rem', color: '#fbbf24', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <ShieldAlert size={16} style={{ flexShrink: 0 }} />
            <div>
              <strong>Active Override On {junctionId}:</strong> {currentOverride.override_mode} ({currentOverride.reason})
            </div>
          </div>
        )}

        {error && (
          <div className="error-banner" style={{ marginBottom: '12px' }}>
            <AlertTriangle size={15} />
            <span>{error}</span>
          </div>
        )}

        {successMsg && (
          <div style={{ background: 'rgba(16, 185, 129, 0.15)', border: '1px solid rgba(16, 185, 129, 0.3)', borderRadius: '8px', padding: '10px 14px', color: '#34d399', fontSize: '0.78rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '14px' }}>
            <CheckCircle2 size={16} />
            <span>{successMsg}</span>
          </div>
        )}

        <form onSubmit={handleApplyOverride}>
          {/* TACTICAL OVERRIDE PRESETS */}
          <div style={{ marginBottom: '16px' }}>
            <label className="form-label" style={{ marginBottom: '8px', display: 'block' }}>
              Select Tactical Signal Light Action:
            </label>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '10px' }}>
              {/* ACTION 1: HOLD RED ON INCIDENT APPROACH */}
              <button
                type="button"
                onClick={() => setSelectedMode('HOLD_RED_APPROACH')}
                style={{
                  textAlign: 'left',
                  padding: '12px 14px',
                  borderRadius: '10px',
                  border: selectedMode === 'HOLD_RED_APPROACH' ? '2px solid #ef4444' : '1px solid rgba(255, 255, 255, 0.1)',
                  background: selectedMode === 'HOLD_RED_APPROACH' ? 'rgba(239, 68, 68, 0.18)' : 'rgba(255, 255, 255, 0.04)',
                  cursor: 'pointer',
                  transition: 'all 0.15s ease'
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: '#ef4444', fontWeight: 800, fontSize: '0.82rem' }}>
                  <span style={{ width: '10px', height: '10px', borderRadius: '50%', background: '#ef4444', display: 'inline-block' }} />
                  Lock {approach} to RED
                </div>
                <div style={{ fontSize: '0.7rem', color: '#94a3b8', marginTop: '4px', lineHeight: 1.35 }}>
                  Blocks oncoming traffic from entering the collision/hazard zone.
                </div>
              </button>

              {/* ACTION 2: EMERGENCY ALL-RED INTERSECTION FREEZE */}
              <button
                type="button"
                onClick={() => setSelectedMode('EMERGENCY_ALL_RED')}
                style={{
                  textAlign: 'left',
                  padding: '12px 14px',
                  borderRadius: '10px',
                  border: selectedMode === 'EMERGENCY_ALL_RED' ? '2px solid #dc2626' : '1px solid rgba(255, 255, 255, 0.1)',
                  background: selectedMode === 'EMERGENCY_ALL_RED' ? 'rgba(220, 38, 38, 0.25)' : 'rgba(255, 255, 255, 0.04)',
                  cursor: 'pointer',
                  transition: 'all 0.15s ease'
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: '#f87171', fontWeight: 800, fontSize: '0.82rem' }}>
                  <AlertTriangle size={14} color="#ef4444" />
                  ALL-RED Freeze (4-Way)
                </div>
                <div style={{ fontSize: '0.7rem', color: '#94a3b8', marginTop: '4px', lineHeight: 1.35 }}>
                  Full intersection halt for trauma ambulance or fire rescue clearance.
                </div>
              </button>

              {/* ACTION 3: PRIORITY EVACUATION GREEN */}
              <button
                type="button"
                onClick={() => setSelectedMode('PRIORITY_GREEN')}
                style={{
                  textAlign: 'left',
                  padding: '12px 14px',
                  borderRadius: '10px',
                  border: selectedMode === 'PRIORITY_GREEN' ? '2px solid #10b981' : '1px solid rgba(255, 255, 255, 0.1)',
                  background: selectedMode === 'PRIORITY_GREEN' ? 'rgba(16, 185, 129, 0.18)' : 'rgba(255, 255, 255, 0.04)',
                  cursor: 'pointer',
                  transition: 'all 0.15s ease'
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: '#10b981', fontWeight: 800, fontSize: '0.82rem' }}>
                  <span style={{ width: '10px', height: '10px', borderRadius: '50%', background: '#10b981', display: 'inline-block' }} />
                  Force {isNScorridor ? 'N-S' : 'E-W'} GREEN
                </div>
                <div style={{ fontSize: '0.7rem', color: '#94a3b8', marginTop: '4px', lineHeight: 1.35 }}>
                  Prioritizes green light window to rapidly evacuate vehicles from corridor.
                </div>
              </button>

              {/* ACTION 4: RESTORE AUTOMATED AI */}
              <button
                type="button"
                onClick={() => setSelectedMode('RESTORE_ADAPTIVE')}
                style={{
                  textAlign: 'left',
                  padding: '12px 14px',
                  borderRadius: '10px',
                  border: selectedMode === 'RESTORE_ADAPTIVE' ? '2px solid #06b6d4' : '1px solid rgba(255, 255, 255, 0.1)',
                  background: selectedMode === 'RESTORE_ADAPTIVE' ? 'rgba(6, 182, 212, 0.18)' : 'rgba(255, 255, 255, 0.04)',
                  cursor: 'pointer',
                  transition: 'all 0.15s ease'
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: '#06b6d4', fontWeight: 800, fontSize: '0.82rem' }}>
                  <RotateCcw size={13} color="#06b6d4" />
                  Restore Adaptive AI
                </div>
                <div style={{ fontSize: '0.7rem', color: '#94a3b8', marginTop: '4px', lineHeight: 1.35 }}>
                  Clears manual overrides and returns junction to neural traffic control.
                </div>
              </button>
            </div>
          </div>

          {/* DIRECTION PICKER IF HOLDING RED */}
          {selectedMode === 'HOLD_RED_APPROACH' && (
            <div style={{ marginBottom: '14px', padding: '10px 14px', background: 'rgba(239, 68, 68, 0.08)', borderRadius: '8px', border: '1px solid rgba(239, 68, 68, 0.2)' }}>
              <label className="form-label" style={{ marginBottom: '6px', fontSize: '0.72rem', color: '#fca5a5' }}>
                Target Approach to Lock to RED:
              </label>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '6px' }}>
                {['NORTH', 'SOUTH', 'EAST', 'WEST'].map((app) => (
                  <button
                    key={app}
                    type="button"
                    onClick={() => setSelectedApproach(app)}
                    style={{
                      padding: '6px 8px',
                      fontSize: '0.74rem',
                      fontWeight: selectedApproach === app ? 800 : 500,
                      borderRadius: '6px',
                      border: selectedApproach === app ? '1px solid #ef4444' : '1px solid rgba(255, 255, 255, 0.1)',
                      background: selectedApproach === app ? '#ef4444' : 'rgba(255, 255, 255, 0.05)',
                      color: '#ffffff',
                      cursor: 'pointer'
                    }}
                  >
                    {app}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* CORRIDOR PICKER IF PRIORITY GREEN IN STANDALONE MODE */}
          {selectedMode === 'PRIORITY_GREEN' && !incident && (
            <div style={{ marginBottom: '14px', padding: '10px 14px', background: 'rgba(16, 185, 129, 0.08)', borderRadius: '8px', border: '1px solid rgba(16, 185, 129, 0.2)' }}>
              <label className="form-label" style={{ marginBottom: '6px', fontSize: '0.72rem', color: '#6ee7b7' }}>
                Select Green Wave Corridor:
              </label>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '8px' }}>
                <button
                  type="button"
                  onClick={() => setSelectedCorridor('NORTH_SOUTH')}
                  style={{
                    padding: '7px 10px',
                    fontSize: '0.74rem',
                    fontWeight: selectedCorridor === 'NORTH_SOUTH' ? 800 : 500,
                    borderRadius: '6px',
                    border: selectedCorridor === 'NORTH_SOUTH' ? '1px solid #10b981' : '1px solid rgba(255, 255, 255, 0.1)',
                    background: selectedCorridor === 'NORTH_SOUTH' ? '#10b981' : 'rgba(255, 255, 255, 0.05)',
                    color: '#ffffff',
                    cursor: 'pointer'
                  }}
                >
                  North - South Corridor
                </button>
                <button
                  type="button"
                  onClick={() => setSelectedCorridor('EAST_WEST')}
                  style={{
                    padding: '7px 10px',
                    fontSize: '0.74rem',
                    fontWeight: selectedCorridor === 'EAST_WEST' ? 800 : 500,
                    borderRadius: '6px',
                    border: selectedCorridor === 'EAST_WEST' ? '1px solid #10b981' : '1px solid rgba(255, 255, 255, 0.1)',
                    background: selectedCorridor === 'EAST_WEST' ? '#10b981' : 'rgba(255, 255, 255, 0.05)',
                    color: '#ffffff',
                    cursor: 'pointer'
                  }}
                >
                  East - West Corridor
                </button>
              </div>
            </div>
          )}

          {/* DURATION SELECTOR (ONLY IF NOT RESTORING AI) */}
          {selectedMode !== 'RESTORE_ADAPTIVE' && (
            <div className="form-group" style={{ marginBottom: '14px' }}>
              <label className="form-label" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <Clock size={13} className="text-cyan" />
                Override Hold Duration:
              </label>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '8px' }}>
                {[
                  { lbl: '60s (1 min)', val: 60 },
                  { lbl: '180s (3 mins)', val: 180 },
                  { lbl: '300s (5 mins)', val: 300 },
                  { lbl: '600s (10 mins)', val: 600 },
                ].map((d) => (
                  <button
                    key={d.val}
                    type="button"
                    onClick={() => setDurationSeconds(d.val)}
                    style={{
                      padding: '6px 10px',
                      borderRadius: '8px',
                      fontSize: '0.72rem',
                      fontWeight: durationSeconds === d.val ? 800 : 500,
                      border: durationSeconds === d.val ? '1px solid #38bdf8' : '1px solid rgba(255, 255, 255, 0.12)',
                      background: durationSeconds === d.val ? 'rgba(56, 189, 248, 0.2)' : 'rgba(255, 255, 255, 0.05)',
                      color: durationSeconds === d.val ? '#ffffff' : '#94a3b8',
                      cursor: 'pointer'
                    }}
                  >
                    {d.lbl}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* JUSTIFICATION REASON */}
          <div className="form-group" style={{ marginBottom: '18px' }}>
            <label className="form-label">Official Command Justification:</label>
            <input 
              type="text" 
              className="form-input" 
              value={customReason} 
              onChange={(e) => setCustomReason(e.target.value)} 
              required 
            />
          </div>

          {/* MODAL ACTIONS */}
          <div className="modal-actions">
            <button 
              type="button" 
              className="btn-secondary" 
              onClick={onClose}
              disabled={loading}
            >
              Cancel
            </button>
            <button 
              type="submit" 
              className={selectedMode === 'RESTORE_ADAPTIVE' ? 'btn-primary' : 'btn-danger'}
              disabled={loading}
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '8px',
                padding: '9px 22px',
                fontWeight: 800,
                fontSize: '0.84rem',
                cursor: loading ? 'not-allowed' : 'pointer'
              }}
            >
              {loading ? (
                <span>Transmitting Directive...</span>
              ) : (
                <>
                  <Zap size={15} />
                  <span>
                    {selectedMode === 'RESTORE_ADAPTIVE' ? 'Restore AI Adaptive Control' : 'Enforce Signal Light Override'}
                  </span>
                  <ArrowRight size={15} />
                </>
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
