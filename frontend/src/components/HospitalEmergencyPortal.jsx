import React, { useState, useEffect } from 'react';
import { 
  Activity, 
  Plus, 
  Send, 
  CheckCircle2, 
  Clock, 
  AlertOctagon, 
  Flame, 
  Navigation, 
  ShieldAlert, 
  MapPin, 
  Phone, 
  Truck,
  RotateCcw,
  Sparkles,
  ArrowRight,
  ShieldCheck,
  Building2,
  Lock
} from 'lucide-react';
import { api } from '../services/api';

const CRITICALITY_CONFIG = {
  CRITICAL_LIFE_THREATENING: {
    label: 'CRITICAL (Priority 4)',
    badgeClass: 'crit-p4',
    desc: 'Cardiac arrest, structure fire, Golden Hour ICU. Full immediate Green Wave override.',
    color: '#ef4444'
  },
  HIGH: {
    label: 'HIGH (Priority 3)',
    badgeClass: 'crit-p3',
    desc: 'Severe fractures, 2nd-alarm fire hazard, emergency surgery. Expedited green corridor.',
    color: '#f97316'
  },
  MEDIUM: {
    label: 'MEDIUM (Priority 2)',
    badgeClass: 'crit-p2',
    desc: 'Urgent medical care, small hazard response. Priority green window extension.',
    color: '#eab308'
  },
  LOW: {
    label: 'LOW (Priority 1)',
    badgeClass: 'crit-p1',
    desc: 'Non-emergency patient transfer / equipment shift. Standard green flow.',
    color: '#22c55e'
  }
};

export default function HospitalEmergencyPortal({ userSession, onMissionUpdated }) {
  const [selectedDepartment, setSelectedDepartment] = useState('HOSPITAL'); // 'HOSPITAL', 'FIRE_RESCUE', 'POLICE_DISASTER'
  const [missions, setMissions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [isRegisterOpen, setIsRegisterOpen] = useState(false);

  // Form State
  const [agencyType, setAgencyType] = useState('HOSPITAL');
  const [hospitalName, setHospitalName] = useState(userSession?.organization_name || 'Apollo Trauma & Emergency Center');
  const [vehicleId, setVehicleId] = useState('DL-01-AMB-911');
  const [driverContact, setDriverContact] = useState('+91 98765 43210');
  const [criticality, setCriticality] = useState('CRITICAL_LIFE_THREATENING');
  const [patientCondition, setPatientCondition] = useState('Acute myocardial infarction (Cardiac Arrest) - Immediate ICU transport required');
  const [victimLocation, setVictimLocation] = useState('Central Ring Road Crossing (near J-01)');
  const [originJunction, setOriginJunction] = useState('J-04');
  const [destinationJunction, setDestinationJunction] = useState('J-02');
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);

  const fetchMissions = async () => {
    try {
      setLoading(true);
      const list = await api.listAmbulances();
      setMissions(list);
    } catch (err) {
      console.error('Failed to load emergency missions', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchMissions();
    const timer = setInterval(fetchMissions, 5000);
    return () => clearInterval(timer);
  }, []);

  const handleOpenRegisterForDept = (dept) => {
    setAgencyType(dept);
    if (dept === 'HOSPITAL') {
      setHospitalName(userSession?.organization_name || 'Apollo Trauma & Emergency Center');
      setVehicleId('DL-01-AMB-911');
      setPatientCondition('Acute myocardial infarction (Cardiac Arrest) - Immediate ICU transport required');
      setOriginJunction('J-04');
      setDestinationJunction('J-02');
    } else if (dept === 'FIRE_RESCUE') {
      setHospitalName('Delhi Fire & Rescue Station No. 4');
      setVehicleId('DL-01-FIRE-77');
      setPatientCondition('Category 2 Commercial Structure Fire — Immediate Water Tender & Hydraulic Rescue');
      setOriginJunction('J-05');
      setDestinationJunction('J-01');
    } else {
      setHospitalName('National Disaster Response Force (NDRF)');
      setVehicleId('DL-01-NDRF-09');
      setPatientCondition('Critical Disaster Relief & Emergency Convoy Protocol');
      setOriginJunction('J-03');
      setDestinationJunction('J-02');
    }
    setIsRegisterOpen(true);
  };

  const handleRegisterMission = async (e) => {
    e.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      const payload = {
        agency_type: agencyType,
        hospital_name: hospitalName,
        ambulance_vehicle_id: vehicleId,
        driver_contact: driverContact,
        criticality,
        patient_condition: patientCondition,
        victim_location: victimLocation,
        origin_junction_id: originJunction,
        destination_junction_id: destinationJunction
      };
      await api.registerAmbulance(payload);
      setIsRegisterOpen(false);
      fetchMissions();
      if (onMissionUpdated) onMissionUpdated();
    } catch (err) {
      setError(err.message || 'Failed to dispatch emergency mission');
    } finally {
      setSubmitting(false);
    }
  };

  const handleStatusProgress = async (missionId, nextStatus) => {
    try {
      await api.updateAmbulanceStatus(missionId, nextStatus);
      fetchMissions();
      if (onMissionUpdated) onMissionUpdated();
    } catch (err) {
      alert('Failed to update mission status: ' + err.message);
    }
  };

  const activeMissions = missions.filter(m => m.status !== 'MISSION_ACCOMPLISHED' && m.status !== 'CANCELLED');
  const completedMissions = missions.filter(m => m.status === 'MISSION_ACCOMPLISHED');

  return (
    <div className="hospital-portal-container animate-fade-in">
      {/* EMERGENCY SERVICES HERO */}
      <div className="portal-hero hospital-theme">
        <div className="hero-left">
          <div className="portal-badge">
            <Activity size={16} /> Integrated Emergency Services Dispatch Network
          </div>
          <h2>Emergency Green Wave &amp; Priority Corridor Command</h2>
          <p>
            Encrypted emergency response network for <strong>Hospitals &amp; Trauma Centers</strong>, 
            <strong> Fire &amp; Rescue Services</strong>, and <strong> Disaster Relief Convoys</strong>. 
            Synchronizes traffic signals for zero-delay emergency transit.
          </p>
        </div>
        <div className="hero-right">
          <button 
            className="btn-dispatch-emergency"
            onClick={() => handleOpenRegisterForDept(selectedDepartment)}
          >
            <Plus size={18} /> Launch Emergency Mission
          </button>
        </div>
      </div>

      {/* DEPARTMENT SELECTOR TABS */}
      <div className="emergency-dept-bar">
        <button 
          className={`dept-btn ${selectedDepartment === 'HOSPITAL' ? 'active hosp' : ''}`}
          onClick={() => setSelectedDepartment('HOSPITAL')}
        >
          <Activity size={16} />
          <div>
            <div className="dept-title">🏥 Hospital Trauma Emergency</div>
            <div className="dept-sub">Ambulance missions &amp; ICU transfers</div>
          </div>
        </button>

        <button 
          className={`dept-btn ${selectedDepartment === 'FIRE_RESCUE' ? 'active fire' : ''}`}
          onClick={() => setSelectedDepartment('FIRE_RESCUE')}
        >
          <Flame size={16} />
          <div>
            <div className="dept-title">🚒 Fire &amp; Rescue Services</div>
            <div className="dept-sub">Structure fires &amp; hazardous spills</div>
          </div>
        </button>

        <button 
          className={`dept-btn ${selectedDepartment === 'POLICE_DISASTER' ? 'active disaster' : ''}`}
          onClick={() => setSelectedDepartment('POLICE_DISASTER')}
        >
          <Building2 size={16} />
          <div>
            <div className="dept-title">🚓 Disaster &amp; Police Convoy</div>
            <div className="dept-sub">Disaster relief &amp; tactical evacuation</div>
          </div>
        </button>
      </div>

      {/* REGISTRATION MODAL */}
      {isRegisterOpen && (
        <div className="modal-backdrop">
          <div className="incident-modal-content hospital-modal">
            <div className="modal-header">
              <div className="title-with-icon">
                <Truck size={24} className="text-emerald" />
                <div>
                  <h3>
                    {agencyType === 'HOSPITAL' && '🏥 Dispatch Hospital Emergency Ambulance'}
                    {agencyType === 'FIRE_RESCUE' && '🚒 Dispatch Fire & Rescue Emergency Tender'}
                    {agencyType === 'POLICE_DISASTER' && '🚓 Dispatch Disaster Relief & Convoy'}
                  </h3>
                  <p className="card-subtitle">Calculates optimal transit path and pre-clears signal green corridors</p>
                </div>
              </div>
              <button className="close-btn" onClick={() => setIsRegisterOpen(false)}>×</button>
            </div>

            {error && <div className="error-banner">{error}</div>}

            <form onSubmit={handleRegisterMission} className="incident-form">
              <div className="form-grid">
                <div className="form-group">
                  <label className="form-label">Emergency Department / Agency</label>
                  <select className="form-select" value={agencyType} onChange={(e) => setAgencyType(e.target.value)}>
                    <option value="HOSPITAL">🏥 Hospital Trauma Care</option>
                    <option value="FIRE_RESCUE">🚒 Fire &amp; Rescue Department</option>
                    <option value="POLICE_DISASTER">🚓 Disaster Relief / Police Escort</option>
                  </select>
                </div>

                <div className="form-group">
                  <label className="form-label">Dispatching Station / Hospital Name</label>
                  <input 
                    type="text" 
                    className="form-input"
                    value={hospitalName} 
                    onChange={(e) => setHospitalName(e.target.value)} 
                    placeholder="e.g. Apollo Hospital Trauma Hub"
                    required 
                  />
                </div>

                <div className="form-group">
                  <label className="form-label">Vehicle Call Sign / Registration</label>
                  <input 
                    type="text" 
                    className="form-input font-mono"
                    value={vehicleId} 
                    onChange={(e) => setVehicleId(e.target.value)} 
                    placeholder="DL-01-AMB-911"
                    required 
                  />
                </div>

                <div className="form-group">
                  <label className="form-label">Driver / Officer Contact</label>
                  <input 
                    type="text" 
                    className="form-input font-mono"
                    value={driverContact} 
                    onChange={(e) => setDriverContact(e.target.value)} 
                    placeholder="+91 98765 43210"
                    required 
                  />
                </div>

                <div className="form-group">
                  <label className="form-label">Emergency Criticality Urgency</label>
                  <select className="form-select" value={criticality} onChange={(e) => setCriticality(e.target.value)}>
                    <option value="CRITICAL_LIFE_THREATENING">🚨 Priority 4 — CRITICAL (Immediate Green Wave Override)</option>
                    <option value="HIGH">🔴 Priority 3 — HIGH (Expedited Green Corridor)</option>
                    <option value="MEDIUM">🟡 Priority 2 — MEDIUM (Priority Green Window Extension)</option>
                    <option value="LOW">🟢 Priority 1 — LOW (Standard Priority Flow)</option>
                  </select>
                </div>

                <div className="form-group">
                  <label className="form-label">Deployment Origin Junction</label>
                  <select className="form-select" value={originJunction} onChange={(e) => setOriginJunction(e.target.value)}>
                    <option value="J-04">J-04 (Hospital / South Sector Hub)</option>
                    <option value="J-01">J-01 (Central Plaza Arterial)</option>
                    <option value="J-02">J-02 (East Arterial Flyover)</option>
                    <option value="J-03">J-03 (North Boulevard Station)</option>
                    <option value="J-05">J-05 (West Fire &amp; Rescue HQ)</option>
                  </select>
                </div>

                <div className="form-group full-width">
                  <label className="form-label">Target Destination Junction</label>
                  <select className="form-select" value={destinationJunction} onChange={(e) => setDestinationJunction(e.target.value)}>
                    <option value="J-02">J-02 (East Specialized Trauma / Industrial Sector)</option>
                    <option value="J-01">J-01 (Central Metropolitan Center)</option>
                    <option value="J-03">J-03 (North University Hospital / Fire Zone)</option>
                    <option value="J-04">J-04 (South General Hospital)</option>
                    <option value="J-05">J-05 (West Commercial Complex)</option>
                  </select>
                </div>
              </div>

              <div className="form-group full-width" style={{ marginTop: '10px' }}>
                <label className="form-label">Emergency Incident / Victim Location</label>
                <input 
                  type="text" 
                  className="form-input"
                  value={victimLocation} 
                  onChange={(e) => setVictimLocation(e.target.value)} 
                  placeholder="e.g. Ring Road Crossing opposite Metro Station"
                  required 
                />
              </div>

              <div className="form-group full-width">
                <label className="form-label">Emergency Clinical Condition / Hazard Brief</label>
                <textarea 
                  rows="2" 
                  className="form-textarea"
                  value={patientCondition} 
                  onChange={(e) => setPatientCondition(e.target.value)} 
                  placeholder="Describe patient vitals, cardiac/trauma tier, fire alarm level, or hazard scope..."
                  required 
                />
              </div>

              <div className="modal-actions">
                <button type="button" className="btn-secondary" onClick={() => setIsRegisterOpen(false)}>Cancel</button>
                <button type="submit" className="btn-emerald" disabled={submitting}>
                  {submitting ? 'Dispatching & Preempting Signals...' : '⚡ Launch Emergency Mission & Activate Green Wave'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* ACTIVE MISSIONS GRID */}
      <div className="missions-section">
        <div className="section-title-row">
          <div>
            <h3>Active Emergency Missions ({activeMissions.length})</h3>
            <p className="card-subtitle">Live green corridor telemetry and multi-unit priority conflict arbitrations</p>
          </div>
          <button className="btn-refresh-sm" onClick={fetchMissions}><RotateCcw size={14} /> Refresh</button>
        </div>

        {activeMissions.length === 0 ? (
          <div className="empty-missions-card">
            <CheckCircle2 size={40} className="text-emerald opacity-75" />
            <h4>No Active Emergency Units in Transit</h4>
            <p>All emergency corridors are clear. Ready to dispatch units on demand with instant Green Wave overrides.</p>
          </div>
        ) : (
          <div className="missions-grid">
            {activeMissions.map((m) => {
              const cfg = CRITICALITY_CONFIG[m.criticality] || CRITICALITY_CONFIG.MEDIUM;
              const conflict = m.conflict_resolution;
              const isFire = m.agency_type === 'FIRE_RESCUE' || m.ambulance_vehicle_id.includes('FIRE');
              const isPolice = m.agency_type === 'POLICE_DISASTER' || m.ambulance_vehicle_id.includes('NDRF');

              return (
                <div key={m.mission_id} className="mission-card active-mission-glow">
                  <div className="mission-header">
                    <div className="mission-left">
                      <span className={`criticality-badge ${cfg.badgeClass}`}>
                        {cfg.label}
                      </span>
                      <span className="mission-id font-mono">{m.mission_id}</span>
                      <span className="vehicle-id font-mono">
                        {isFire ? '🚒 ' : isPolice ? '🚓 ' : '🚑 '}
                        {m.ambulance_vehicle_id}
                      </span>
                    </div>

                    <div className="mission-status-pill font-mono">
                      ● {m.status.replace(/_/g, ' ')}
                    </div>
                  </div>

                  <div className="mission-patient-block">
                    <div className="patient-cond font-bold">{m.patient_condition}</div>
                    <div className="patient-loc"><MapPin size={13} /> Scene: {m.victim_location}</div>
                  </div>

                  {/* Conflict Resolution Banner */}
                  {conflict && conflict.has_conflict && (
                    <div className="conflict-resolution-banner">
                      <AlertOctagon size={16} className="text-amber flex-shrink-0" />
                      <div>
                        <strong>Multi-Unit Intersection Conflict Arbitrated:</strong>
                        <p>{conflict.strategy}</p>
                      </div>
                    </div>
                  )}

                  {/* Route Green Wave Timeline */}
                  <div className="route-timeline-box">
                    <div className="timeline-title font-mono text-cyan">
                      <Navigation size={14} /> Synchronized Green Wave Corridor:
                    </div>
                    <div className="timeline-nodes-flex">
                      {m.route_corridor.map((node, idx) => {
                        const isCurrent = idx === m.active_node_index;
                        const isPast = idx < m.active_node_index;
                        return (
                          <div key={node.junction_id} className={`timeline-node ${isCurrent ? 'current' : isPast ? 'past' : 'upcoming'}`}>
                            <div className="node-circle font-mono">{node.junction_id}</div>
                            <div className="node-detail">
                              <span className="corridor-name">{node.corridor_name}</span>
                              <span className="app-tag font-mono">via {node.approach}</span>
                            </div>
                            {idx < m.route_corridor.length - 1 && <ArrowRight size={14} className="node-arrow" />}
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  {/* Dispatch Controls Progression */}
                  <div className="mission-action-bar">
                    <span className="driver-phone font-mono"><Phone size={13} /> Driver: {m.driver_contact}</span>

                    <div className="status-progress-buttons">
                      {m.status === 'DISPATCHED_TO_VICTIM' && (
                        <button 
                          className="btn-step-progress"
                          onClick={() => handleStatusProgress(m.mission_id, 'ON_SCENE_PICKUP')}
                        >
                          Arrived On Scene (Secure Incident)
                        </button>
                      )}

                      {m.status === 'ON_SCENE_PICKUP' && (
                        <button 
                          className="btn-step-progress"
                          onClick={() => handleStatusProgress(m.mission_id, 'TRANSIT_TO_HOSPITAL')}
                        >
                          En Route to Destination (Transit)
                        </button>
                      )}

                      {m.status === 'TRANSIT_TO_HOSPITAL' && (
                        <button 
                          className="btn-step-complete"
                          onClick={() => handleStatusProgress(m.mission_id, 'MISSION_ACCOMPLISHED')}
                        >
                          ✓ Destination Reached (Complete Mission)
                        </button>
                      )}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* COMPLETED MISSIONS */}
      {completedMissions.length > 0 && (
        <div className="completed-missions-drawer">
          <h4>Completed Emergency Operations ({completedMissions.length})</h4>
          <div className="completed-list">
            {completedMissions.map((m) => (
              <div key={m.mission_id} className="completed-item">
                <span className="font-mono text-cyan">{m.mission_id}</span>
                <span className="font-bold">{m.ambulance_vehicle_id}</span>
                <span>{m.patient_condition}</span>
                <span className="badge-accomplished font-mono">✓ MISSION SAFELY COMPLETED</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
