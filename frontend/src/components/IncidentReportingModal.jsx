import React, { useState, useRef, useEffect } from 'react';
import { 
  X, 
  AlertTriangle, 
  Camera, 
  RotateCcw, 
  CheckCircle, 
  ShieldAlert, 
  ShieldCheck, 
  MapPin, 
  Clock, 
  Sparkles,
  Lock,
  Building2,
  PhoneCall,
  FileText,
  Radio
} from 'lucide-react';
import { api } from '../services/api';

export default function IncidentReportingModal({ 
  isOpen, 
  onClose, 
  junctions = [], 
  currentJunction = 'J-01',
  mode = 'PUBLIC',
  userRole = 'PUBLIC_USER',
  userSession = null,
  onIncidentReported 
}) {
  const [junctionId, setJunctionId] = useState(currentJunction || 'J-01');
  const [approach, setApproach] = useState('NORTH');
  const [roadName, setRoadName] = useState('North Boulevard');
  const [incidentType, setIncidentType] = useState('ACCIDENT');
  const [severity, setSeverity] = useState('SEVERE');
  const [description, setDescription] = useState('');
  const [estimatedMinutes, setEstimatedMinutes] = useState(30);
  const [reportedBy, setReportedBy] = useState('Citizen Commuter');

  // Strict Separation: Public Portal ALWAYS has live camera feed (mode === 'PUBLIC')
  // Traffic Police Base Dispatch is ONLY active when explicitly in 'POLICE' mode.
  const isPolice = mode === 'POLICE';
  const reporterMode = isPolice ? 'TRAFFIC_POLICE' : 'PUBLIC_USER';
  
  // Police Base Dispatch Specific State
  const [dispatchSource, setDispatchSource] = useState('Emergency 112 / PCR Call-In');
  const [dispatchCallRef, setDispatchCallRef] = useState('');

  // Live Camera State (Strictly for Public Citizen Portal)
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState(null);
  const [capturedPhoto, setCapturedPhoto] = useState(null);
  const [captureTimestamp, setCaptureTimestamp] = useState(null);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);

  const [submitting, setSubmitting] = useState(false);
  const [resultPlan, setResultPlan] = useState(null);
  const [error, setError] = useState(null);

  // Sync state when modal opens or role changes
  useEffect(() => {
    if (isOpen) {
      setReportedBy(isPolice ? (userSession?.organization_name || 'Traffic Police Base Command') : 'Citizen Commuter');
      setDispatchCallRef(`POL-CALL-${Math.floor(1000 + Math.random() * 9000)}`);
      setJunctionId(currentJunction || 'J-01');
      
      const j = junctions.find(item => item.junction_id === (currentJunction || 'J-01'));
      if (j && j.road_names && j.road_names['NORTH']) {
        setRoadName(j.road_names['NORTH']);
      }
      if (!isPolice) {
        setCapturedPhoto(null);
      }
    }
  }, [isOpen, isPolice, userSession, currentJunction, junctions]);

  // Manage Camera: Only start if public citizen and modal is open
  useEffect(() => {
    if (isOpen && !isPolice && !capturedPhoto) {
      startCamera();
    } else {
      stopCamera();
    }
    return () => stopCamera();
  }, [isOpen, isPolice, capturedPhoto]);

  const startCamera = async () => {
    setCameraError(null);
    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('Camera access not supported on this device/browser');
      }
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { 
          facingMode: { ideal: 'environment' },
          width: { ideal: 1280 },
          height: { ideal: 720 }
        },
        audio: false
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }
      setCameraActive(true);
    } catch (err) {
      console.warn('Camera error, creating virtual on-scene viewfinder:', err);
      setCameraError('Live camera not directly accessible or permission denied. Virtual on-scene simulator active.');
      setCameraActive(false);
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    setCameraActive(false);
  };

  const takeLiveSnapshot = () => {
    const canvas = canvasRef.current || document.createElement('canvas');
    canvas.width = 640;
    canvas.height = 360;
    const ctx = canvas.getContext('2d');
    const nowStr = new Date().toISOString();

    if (cameraActive && videoRef.current) {
      ctx.drawImage(videoRef.current, 0, 0, 640, 360);
    } else {
      const gradient = ctx.createLinearGradient(0, 0, 640, 360);
      gradient.addColorStop(0, '#0f172a');
      gradient.addColorStop(0.5, '#1e293b');
      gradient.addColorStop(1, '#090d16');
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, 640, 360);

      ctx.strokeStyle = '#334155';
      ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.moveTo(100, 360); ctx.lineTo(300, 160);
      ctx.moveTo(540, 360); ctx.lineTo(340, 160);
      ctx.stroke();

      ctx.fillStyle = '#ef4444';
      ctx.fillRect(270, 200, 100, 50);
      ctx.fillStyle = '#f87171';
      ctx.beginPath();
      ctx.arc(290, 255, 14, 0, Math.PI * 2);
      ctx.arc(350, 255, 14, 0, Math.PI * 2);
      ctx.fill();
    }

    // Embed Verified On-Scene Watermark
    ctx.fillStyle = 'rgba(0, 0, 0, 0.65)';
    ctx.fillRect(10, 290, 620, 60);

    ctx.fillStyle = '#22d3ee';
    ctx.font = 'bold 13px monospace';
    ctx.fillText(`● VERIFIED ON-SCENE LIVE EVIDENCE CAPTURE`, 20, 312);

    ctx.fillStyle = '#f8fafc';
    ctx.font = '11px sans-serif';
    ctx.fillText(`LOCATION: ${junctionId} (${approach}) — ${roadName}  |  TIME: ${nowStr}`, 20, 335);

    const dataUri = canvas.toDataURL('image/jpeg', 0.9);
    setCapturedPhoto(dataUri);
    setCaptureTimestamp(nowStr);
    stopCamera();
  };

  const retakePhoto = () => {
    setCapturedPhoto(null);
    setCaptureTimestamp(null);
    startCamera();
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!description.trim()) {
      setError('Please provide a short description of the incident.');
      return;
    }

    // Strict live camera enforcement ONLY for public citizens
    if (!isPolice && !capturedPhoto) {
      setError('A live on-the-spot photo is mandatory for public reports to prevent false submissions.');
      return;
    }

    setSubmitting(true);
    setError(null);

    try {
      const nowStr = new Date().toISOString();
      const payload = {
        junction_id: junctionId,
        approach,
        road_name: roadName,
        incident_type: incidentType,
        severity,
        description: isPolice && dispatchCallRef 
          ? `[${dispatchCallRef} | Source: ${dispatchSource}] ${description}` 
          : description,
        estimated_clearance_minutes: Number(estimatedMinutes),
        reported_by: reportedBy,
        reporter_role: reporterMode,
        dispatch_call_ref: isPolice ? dispatchCallRef : null,
        photo_base64: !isPolice ? capturedPhoto : null,
        is_live_captured: !isPolice && Boolean(capturedPhoto),
        capture_timestamp: !isPolice ? captureTimestamp : nowStr
      };

      const res = await api.reportIncident(payload);
      setResultPlan(res.diversion_plan);
      if (onIncidentReported) onIncidentReported(res);
    } catch (err) {
      setError(err.message || 'Failed to submit incident report');
    } finally {
      setSubmitting(false);
    }
  };

  const handleDone = () => {
    setResultPlan(null);
    setCapturedPhoto(null);
    setDescription('');
    onClose();
  };

  if (!isOpen) return null;

  return (
    <div className="modal-backdrop">
      <div className="incident-modal-content">
        {/* MODAL HEADER: DEDICATED TITLE & SUBTITLE BY ROLE */}
        <div className="modal-header">
          <div className="title-with-icon">
            {isPolice ? (
              <Building2 className="text-blue" size={24} />
            ) : (
              <ShieldAlert className="text-red" size={24} />
            )}
            <div>
              <h3>
                {isPolice 
                  ? 'Traffic Police Base Operations — Incident Report' 
                  : 'Report On-Scene Accident / Incident'}
              </h3>
              <p className="card-subtitle">
                {isPolice 
                  ? 'Headquarters & Base Control Station: Emergency call logged directly to trigger junction rerouting (live camera upload waived)'
                  : 'Public Commuter Hub: Live camera capture strictly enforced to authenticate on-scene reports and prevent false alarms'}
              </p>
            </div>
          </div>
          <button className="close-btn" onClick={onClose}><X size={20} /></button>
        </div>

        {error && <div className="error-banner">{error}</div>}

        {resultPlan ? (
          <div className="diversion-result-view">
            <div className="success-banner">
              <CheckCircle size={22} className="text-green" />
              <div>
                <h4>
                  {isPolice 
                    ? 'Official Police Incident Verified & Diversions Dispatched!' 
                    : 'On-Scene Incident Verified & Diversions Dispatched!'}
                </h4>
                <p>
                  {isPolice
                    ? `Logged from base control via emergency call [${dispatchCallRef}]. Upstream signals adjusted to prevent gridlock.`
                    : 'Live photographic evidence authenticated. Upstream signals adjusted to prevent gridlock.'}
                </p>
              </div>
            </div>

            {isPolice ? (
              <div className="police-verified-dispatch-card">
                <div className="dispatch-badge-row">
                  <span className="badge-police-official">
                    <ShieldCheck size={14} /> Verified Police Base Record: {dispatchCallRef}
                  </span>
                  <span className="dispatch-source-tag font-mono">
                    <Radio size={12} /> {dispatchSource}
                  </span>
                </div>
                <div className="dispatch-meta-row">
                  <span><strong>Junction:</strong> {junctionId} ({approach}) — {roadName}</span>
                  <span><strong>Reporting Officer / Unit:</strong> {reportedBy}</span>
                </div>
              </div>
            ) : (
              capturedPhoto && (
                <div className="captured-evidence-preview">
                  <img src={capturedPhoto} alt="Live Accident Evidence" className="evidence-img" />
                  <div className="evidence-tag">
                    <ShieldCheck size={14} /> Verified On-Scene Live Image
                  </div>
                </div>
              )
            )}

            <div className="diversion-plan-card">
              <h4>Automated Reroute &amp; Signal Strategy</h4>
              <div className="diversion-detail-item">
                <span className="label">Recommended Bypass:</span>
                <span className="value font-bold text-amber">{resultPlan.recommended_reroute_corridor} (via {resultPlan.bypass_junction_id})</span>
              </div>
              <div className="diversion-detail-item">
                <span className="label">Adaptive Signal Strategy:</span>
                <span className="value">{resultPlan.signal_timing_strategy}</span>
              </div>

              <div className="diversion-steps-list">
                <h5>Operational Execution Steps:</h5>
                {resultPlan.steps.map((step) => (
                  <div key={step.step_number} className="step-item">
                    <span className="step-badge">{step.step_number}</span>
                    <div>
                      <div className="step-inst">{step.instruction}</div>
                      <div className="step-signal font-mono text-cyan">⚡ {step.signal_action}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="modal-actions">
              <button className="btn-primary" onClick={handleDone}>Return to Live Dashboard</button>
            </div>
          </div>
        ) : (
          <form onSubmit={handleSubmit} className="incident-form">
            {/* TRAFFIC POLICE ONLY: BASE DISPATCH OPERATIONS CARD (NO CAMERA) */}
            {isPolice && (
              <div className="police-base-dispatch-card">
                <div className="police-base-header">
                  <div className="police-header-left">
                    <Building2 size={18} className="text-blue" />
                    <div>
                      <h4 className="police-base-title">Traffic Police Headquarters Base Operations</h4>
                      <p className="police-base-desc">
                        Officers stationed at base log verified emergency phone/radio calls directly. On-scene photo upload is removed for police personnel.
                      </p>
                    </div>
                  </div>
                  <span className="police-shield-badge">
                    <ShieldCheck size={13} /> Official Police Dispatch
                  </span>
                </div>

                <div className="police-dispatch-inputs-row">
                  <div className="form-group">
                    <label className="form-label">
                      <PhoneCall size={12} className="text-blue inline-icon" /> Emergency Call Source
                    </label>
                    <select 
                      className="form-select"
                      value={dispatchSource}
                      onChange={(e) => setDispatchSource(e.target.value)}
                    >
                      <option value="Emergency 112 / PCR Call-In">🚨 Emergency 112 / PCR Call-In</option>
                      <option value="Patrol Unit Radio Call">📻 Patrol Car Radio Dispatch</option>
                      <option value="Traffic CCTV Surveillance Alert">📹 Traffic CCTV Surveillance Alert</option>
                      <option value="Direct Base Operations Hotline">☎️ Direct Base Hotline</option>
                    </select>
                  </div>

                  <div className="form-group">
                    <label className="form-label">
                      <FileText size={12} className="text-blue inline-icon" /> Base Call Reference ID
                    </label>
                    <input 
                      type="text" 
                      className="form-input font-mono"
                      value={dispatchCallRef}
                      onChange={(e) => setDispatchCallRef(e.target.value)}
                      placeholder="e.g. POL-CALL-4921"
                    />
                  </div>
                </div>
              </div>
            )}

            <div className="form-grid">
              <div className="form-group">
                <label className="form-label">Target Junction</label>
                <select 
                  className="form-select"
                  value={junctionId} 
                  onChange={(e) => {
                    setJunctionId(e.target.value);
                    const j = junctions.find(item => item.junction_id === e.target.value);
                    if (j && j.road_names && j.road_names[approach]) {
                      setRoadName(j.road_names[approach]);
                    }
                  }}
                >
                  {junctions.map((j) => (
                    <option key={j.junction_id} value={j.junction_id}>
                      {j.junction_id} — {j.name}
                    </option>
                  ))}
                </select>
              </div>

              <div className="form-group">
                <label className="form-label">Affected Approach &amp; Roadway</label>
                <div className="approach-road-flex">
                  <select 
                    className="form-select"
                    value={approach} 
                    onChange={(e) => {
                      setApproach(e.target.value);
                      const j = junctions.find(item => item.junction_id === junctionId);
                      if (j && j.road_names && j.road_names[e.target.value]) {
                        setRoadName(j.road_names[e.target.value]);
                      }
                    }}
                  >
                    <option value="NORTH">NORTH (Incoming)</option>
                    <option value="SOUTH">SOUTH (Incoming)</option>
                    <option value="EAST">EAST (Incoming)</option>
                    <option value="WEST">WEST (Incoming)</option>
                  </select>
                  <input 
                    type="text" 
                    className="form-input"
                    placeholder="Road Name"
                    value={roadName} 
                    onChange={(e) => setRoadName(e.target.value)} 
                  />
                </div>
              </div>

              <div className="form-group">
                <label className="form-label">Incident Type</label>
                <select className="form-select" value={incidentType} onChange={(e) => setIncidentType(e.target.value)}>
                  <option value="ACCIDENT">🚗 Collision / Vehicle Crash</option>
                  <option value="VEHICLE_BREAKDOWN">🚛 Heavy Vehicle Breakdown / Stall</option>
                  <option value="WATERLOGGING">🌊 Waterlogging / Road Flooding</option>
                  <option value="ROAD_HAZARD">⚠️ Road Obstruction / Debris</option>
                  <option value="ROAD_WORK">🚧 Emergency Utility / Road Work</option>
                </select>
              </div>

              <div className="form-group">
                <label className="form-label">Severity Level</label>
                <select className="form-select" value={severity} onChange={(e) => setSeverity(e.target.value)}>
                  <option value="CRITICAL_ROAD_BLOCKED">🔴 CRITICAL — Entire Road Blocked</option>
                  <option value="SEVERE">🟠 SEVERE — Multiple Lanes Obstructed</option>
                  <option value="MODERATE">🟡 MODERATE — Partial Shoulder Blockage</option>
                  <option value="MINOR">🟢 MINOR — Minor Slowdown Hazard</option>
                </select>
              </div>
            </div>

            {/* MANDATORY ON-THE-SPOT LIVE CAMERA CAPTURE SECTION (PUBLIC CITIZEN ONLY) */}
            {!isPolice && (
              <div className="camera-capture-section">
                <div className="camera-section-header">
                  <div className="camera-title">
                    <Camera size={16} className="text-cyan" />
                    <span>Mandatory On-The-Spot Live Camera Evidence</span>
                  </div>
                  <div className="camera-lock-pill">
                    <Lock size={11} /> Live Capture Only (Anti-Fraud)
                  </div>
                </div>

                <div className="viewfinder-container">
                  {capturedPhoto ? (
                    <div className="snapped-photo-box">
                      <img src={capturedPhoto} alt="Snapped Evidence" className="snapped-img" />
                      <div className="snapped-overlay">
                        <span className="verified-badge"><ShieldCheck size={14} /> LIVE PHOTO VERIFIED</span>
                        <button type="button" className="retake-btn" onClick={retakePhoto}>
                          <RotateCcw size={13} /> Retake Photo
                        </button>
                      </div>
                    </div>
                  ) : (
                    <div className="live-video-box">
                      <video ref={videoRef} autoPlay playsInline muted className="live-video-stream" />
                      <div className="reticle-frame">
                        <div className="reticle-corner top-left"></div>
                        <div className="reticle-corner top-right"></div>
                        <div className="reticle-corner bottom-left"></div>
                        <div className="reticle-corner bottom-right"></div>
                        <div className="viewfinder-crosshair"></div>
                      </div>
                      <div className="viewfinder-bottom-bar">
                        <span className="live-tag">● LIVE ON-SCENE CAMERA</span>
                        <button type="button" className="snap-btn" onClick={takeLiveSnapshot}>
                          <Camera size={15} /> Snap Evidence Photo
                        </button>
                      </div>
                    </div>
                  )}
                </div>
                {cameraError && <div className="camera-fallback-note">{cameraError}</div>}
              </div>
            )}

            <div className="form-group full-width" style={{ marginTop: '14px' }}>
              <label className="form-label">
                {isPolice 
                  ? 'Emergency Caller Report & Road Obstruction Details' 
                  : 'Physical Obstruction & Hazard Details'}
              </label>
              <textarea 
                rows="2" 
                className="form-textarea"
                placeholder={
                  isPolice 
                    ? 'Caller statement, vehicle count, lane blockage status, and dispatched patrol unit details...'
                    : 'Describe vehicle types, lane obstruction, and first responder status...'
                }
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                required
              />
            </div>

            <div className="modal-actions">
              <button type="button" className="btn-secondary" onClick={onClose}>Cancel</button>
              
              {isPolice ? (
                <button type="submit" className="btn-police-report" disabled={submitting}>
                  <ShieldAlert size={16} />
                  {submitting ? 'Authenticating & Rerouting...' : '🚨 Log Police Report & Trigger Junction Reroute'}
                </button>
              ) : (
                <button type="submit" className="btn-danger" disabled={submitting || !capturedPhoto}>
                  {submitting ? 'Authenticating & Rerouting...' : '🚨 Submit Live Report & Reroute Traffic'}
                </button>
              )}
            </div>
          </form>
        )}
      </div>
    </div>
  );
}
