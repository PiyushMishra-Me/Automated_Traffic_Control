import React, { useState } from 'react';
import { 
  ShieldAlert, 
  Camera, 
  MapPin, 
  AlertTriangle, 
  Compass, 
  CheckCircle2, 
  Navigation,
  Sparkles,
  ArrowRight,
  Route,
  Zap,
  Clock,
  Car,
  Bike,
  Bus,
  ShieldCheck,
  AlertOctagon,
  Layers,
  ChevronRight,
  TrendingDown
} from 'lucide-react';
import LiveJunctionMap from './LiveJunctionMap';
import WeatherWidget from './WeatherWidget';
import { api } from '../services/api';

export default function PublicCitizenPortal({ 
  junctions = [], 
  selectedJunction, 
  onSelectJunction, 
  incidents = [], 
  weatherData, 
  onOpenReportModal,
  onWeatherUpdated
}) {
  // City & Navigation State
  const [selectedCity, setSelectedCity] = useState('DELHI');
  const [navOrigin, setNavOrigin] = useState('J-04');
  const [navDestination, setNavDestination] = useState('J-02');
  const [vehicleType, setVehicleType] = useState('CAR');
  const [navLoading, setNavLoading] = useState(false);
  const [navResult, setNavResult] = useState(null);
  const [navError, setNavError] = useState(null);

  // Filter junctions by selected city
  const cityJunctions = junctions.filter(j => (j.city || 'DELHI').toUpperCase() === selectedCity.toUpperCase());

  // Handle City Change
  const handleCityChange = (cityKey) => {
    setSelectedCity(cityKey);
    const filtered = junctions.filter(j => (j.city || 'DELHI').toUpperCase() === cityKey.toUpperCase());
    if (filtered.length >= 2) {
      setNavOrigin(filtered[0].junction_id);
      setNavDestination(filtered[1].junction_id);
      if (onSelectJunction) onSelectJunction(filtered[0].junction_id);
    } else if (filtered.length === 1) {
      setNavOrigin(filtered[0].junction_id);
      setNavDestination(filtered[0].junction_id);
      if (onSelectJunction) onSelectJunction(filtered[0].junction_id);
    }
    setNavResult(null);
    setNavError(null);
  };

  const activeIncidents = incidents.filter(i => i.status === 'ACTIVE');

  const handleComputeRoute = async (e) => {
    if (e) e.preventDefault();
    if (navOrigin === navDestination) {
      setNavError('Origin and Destination cannot be the same junction.');
      return;
    }
    setNavLoading(true);
    setNavError(null);
    try {
      const payload = {
        origin_junction_id: navOrigin,
        destination_junction_id: navDestination,
        vehicle_type: vehicleType,
        avoid_high_congestion: true
      };
      const res = await api.calculateOptimalRoute(payload);
      setNavResult(res);
    } catch (err) {
      setNavError(err.message || 'Failed to calculate optimal route.');
    } finally {
      setNavLoading(false);
    }
  };

  return (
    <div className="public-portal-container animate-fade-in">
      {/* PUBLIC HERO & REPORT CALL TO ACTION */}
      <div className="public-hero-card">
        <div className="public-hero-left">
          <div className="public-tag">
            <Sparkles size={14} className="text-cyan" /> Public Commuter &amp; Road Safety Hub
          </div>
          <h2>Smart Commuter Routing &amp; Real-World Road Navigation</h2>
          <p>
            Find the <strong>real-world shortest &amp; traffic-optimized route</strong> across metropolitan cities, 
            avoid active accident choke points, and report road hazards on the spot.
          </p>
        </div>

        <div className="public-hero-right">
          <button className="btn-public-report-hero" onClick={onOpenReportModal}>
            <Camera size={20} />
            <div>
              <div className="btn-main-text">Report Crash / Hazard</div>
              <div className="btn-sub-text">Live Camera Snapshot Required</div>
            </div>
          </button>
        </div>
      </div>

      {/* SMART COMMUTER DYNAMIC ROUTE PLANNER */}
      <div className="commuter-route-planner-card">
        <div className="planner-header">
          <div className="title-with-icon">
            <Route size={22} className="text-cyan" />
            <div>
              <h3>Real-World Dynamic Shortest Path Navigator</h3>
              <p className="card-subtitle">
                Live vision AI traffic-weighted pathfinder • Real road distance • Active incident avoidance
              </p>
            </div>
          </div>
        </div>

        {/* City Selector Bar */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.25rem', flexWrap: 'wrap' }}>
          <span style={{ fontSize: '0.82rem', fontWeight: 800, color: '#475569' }}>Select Metropolitan City:</span>
          {[
            { key: 'DELHI', label: '🏛️ New Delhi' },
            { key: 'MUMBAI', label: '🌊 Mumbai' },
            { key: 'HYDERABAD', label: '💎 Hyderabad' },
            { key: 'BENGALURU', label: '🌳 Bengaluru' }
          ].map(c => (
            <button
              key={c.key}
              type="button"
              onClick={() => handleCityChange(c.key)}
              style={{
                padding: '0.35rem 0.75rem',
                borderRadius: '8px',
                fontSize: '0.8rem',
                fontWeight: selectedCity === c.key ? 800 : 600,
                background: selectedCity === c.key ? '#0284c7' : '#f8fafc',
                color: selectedCity === c.key ? '#ffffff' : '#334155',
                border: `1px solid ${selectedCity === c.key ? '#0284c7' : '#cbd5e1'}`,
                cursor: 'pointer',
                transition: 'all 0.15s ease'
              }}
            >
              {c.label}
            </button>
          ))}
        </div>

        <form onSubmit={handleComputeRoute} className="planner-form">
          <div className="planner-grid">
            <div className="planner-field">
              <label><MapPin size={14} className="text-emerald" /> Origin / Starting Location</label>
              <select value={navOrigin} onChange={(e) => setNavOrigin(e.target.value)}>
                {cityJunctions.length > 0 ? (
                  cityJunctions.map(j => (
                    <option key={j.junction_id} value={j.junction_id}>
                      {j.junction_id}: {j.name} ({j.location || 'Arterial'})
                    </option>
                  ))
                ) : (
                  <option value="J-01">J-01: Central Plaza Interchange</option>
                )}
              </select>
            </div>

            <div className="planner-field">
              <label><Compass size={14} className="text-cyan" /> Target Destination</label>
              <select value={navDestination} onChange={(e) => setNavDestination(e.target.value)}>
                {cityJunctions.length > 0 ? (
                  cityJunctions.map(j => (
                    <option key={j.junction_id} value={j.junction_id}>
                      {j.junction_id}: {j.name} ({j.location || 'Arterial'})
                    </option>
                  ))
                ) : (
                  <option value="J-02">J-02: Tech City Interchange</option>
                )}
              </select>
            </div>

            <div className="planner-field">
              <label><Car size={14} /> Commuter Vehicle Type</label>
              <select value={vehicleType} onChange={(e) => setVehicleType(e.target.value)}>
                <option value="CAR">🚗 Standard Car / Sedan</option>
                <option value="TWO_WHEELER">🛵 Two-Wheeler / Motorcycle</option>
                <option value="EV">⚡ Electric Vehicle (EV)</option>
                <option value="BUS_HEAVY">🚌 Transit Bus / Heavy Commuter</option>
              </select>
            </div>

            <div className="planner-action-col">
              <button type="submit" className="btn-find-route" disabled={navLoading}>
                {navLoading ? 'Computing Live Path...' : '⚡ Find Shortest Path'}
              </button>
            </div>
          </div>
        </form>

        {navError && <div className="error-banner" style={{ marginTop: '14px' }}>{navError}</div>}

        {/* ROUTE RESULTS OVERVIEW */}
        {navResult && (
          <div className="route-results-container animate-fade-in">
            {/* STATS BAR */}
            <div className="route-stats-row">
              <div className="route-stat-box primary">
                <span className="stat-label">Estimated Transit Time</span>
                <span className="stat-val font-mono">{navResult.estimated_travel_time_formatted}</span>
                <span className="stat-sub">Live Traffic Adjusted</span>
              </div>

              <div className="route-stat-box">
                <span className="stat-label">Total Distance</span>
                <span className="stat-val font-mono">{navResult.total_distance_km} km</span>
                <span className="stat-sub">Optimal Road Graph</span>
              </div>

              {navResult.delay_saved_seconds > 0 ? (
                <div className="route-stat-box green">
                  <span className="stat-label">Congestion Saved</span>
                  <span className="stat-val font-mono text-emerald">
                    +{Math.round(navResult.delay_saved_seconds / 60)} min faster
                  </span>
                  <span className="stat-sub">vs Jammed Direct Corridor</span>
                </div>
              ) : (
                <div className="route-stat-box">
                  <span className="stat-label">Flow Quality</span>
                  <span className="stat-val font-mono text-emerald">Optimal Flow</span>
                  <span className="stat-sub">Zero Artificial Bottlenecks</span>
                </div>
              )}
            </div>

            {/* EMERGENCY PRIORITY CORRIDOR WARNING */}
            {navResult.emergency_corridor_warnings && navResult.emergency_corridor_warnings.length > 0 && (
              <div className="emergency-route-alert-box">
                <div className="alert-badge-row">
                  <span className="badge-emergency-preempt">
                    <ShieldAlert size={14} /> EMERGENCY VEHICLE PRIORITY PREEMPTION ACTIVE
                  </span>
                </div>
                <div className="alert-text-content">
                  {navResult.emergency_corridor_warnings.map((warn, i) => (
                    <p key={i} className="warn-p">{warn}</p>
                  ))}
                  <div className="warn-instruction">
                    ℹ️ <em>Public Commuter Protocol: Traffic signals along this corridor will prioritize approaching emergency vehicles. Please maintain standard lane discipline and yield right-of-way.</em>
                  </div>
                </div>
              </div>
            )}

            {/* LOAD BALANCING & WEATHER ADVISORIES */}
            {(navResult.load_balancing_advisory || navResult.weather_impact_advisory) && (
              <div className="route-advisories-bar">
                {navResult.load_balancing_advisory && (
                  <div className="adv-chip load">
                    <TrendingDown size={14} /> {navResult.load_balancing_advisory}
                  </div>
                )}
                {navResult.weather_impact_advisory && (
                  <div className="adv-chip weather">
                    <Sparkles size={14} /> {navResult.weather_impact_advisory}
                  </div>
                )}
              </div>
            )}

            {/* STEP BY STEP TURN GUIDANCE */}
            <div className="turn-guidance-list">
              <h4>Turn-by-Turn Navigation Steps ({navResult.steps.length})</h4>
              <div className="steps-timeline">
                {navResult.steps.map((step) => {
                  return (
                    <div key={step.step_number} className={`step-item ${step.emergency_active ? 'step-emergency' : ''}`}>
                      <div className="step-num-circle font-mono">{step.step_number}</div>
                      <div className="step-content">
                        <div className="step-top-row">
                          <strong className="step-road">{step.road_name}</strong>
                          <span className="step-dist font-mono">{step.distance_km} km ({Math.round(step.eta_seconds / 60)} min)</span>
                        </div>
                        <p className="step-inst">{step.instruction}</p>
                        
                        <div className="step-badges-row">
                          <span className={`step-cong-badge ${step.congestion_level.toLowerCase()}`}>
                            Traffic: {step.congestion_level.replace('_', ' ')}
                          </span>
                          {step.emergency_active && (
                            <span className="step-emb-badge">
                              🚨 Active Emergency Corridor (Priority {step.emergency_priority})
                            </span>
                          )}
                          {step.advisory_notes && (
                            <span className="step-adv-text">{step.advisory_notes}</span>
                          )}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* WEATHER & ROAD TRACTION ADVISORY */}
      <WeatherWidget 
        junctionId={selectedJunction}
        weatherData={weatherData}
        onWeatherUpdated={onWeatherUpdated}
      />

      {/* LIVE ROAD & DETOUR MAP (WITH COMMUTER ROUTE HIGHLIGHT) */}
      <div className="public-map-wrapper">
        <div className="public-section-header">
          <div className="title-with-icon">
            <Compass size={20} className="text-cyan" />
            <div>
              <h3>Live Metropolitan Traffic Flow &amp; Navigation Map</h3>
              <p className="card-subtitle">
                {navResult 
                  ? `Showing active commuter route from ${navResult.origin_name} to ${navResult.destination_name} (Electric Blue Trail)` 
                  : 'Real-time road corridor congestion, active diversions, and weather radar'}
              </p>
            </div>
          </div>
        </div>

        <LiveJunctionMap 
          junctions={junctions}
          selectedJunction={selectedJunction}
          onSelectJunction={onSelectJunction}
          incidents={incidents}
          weatherData={weatherData}
          navigationRoute={navResult}
          onOpenReportModal={onOpenReportModal}
          selectedCity={selectedCity}
          onSelectCity={handleCityChange}
          onSetNavOrigin={(jId) => setNavOrigin(jId)}
          onSetNavDestination={(jId) => setNavDestination(jId)}
        />
      </div>

      {/* ACTIVE PUBLIC ROAD HAZARDS & ADVISORIES */}
      <div className="public-advisories-card">
        <div className="advisory-title-row">
          <div className="title-with-icon">
            <AlertTriangle size={20} className="text-amber" />
            <div>
              <h3>Active Road Advisories &amp; Commuter Alerts ({activeIncidents.length})</h3>
              <p className="card-subtitle">Verified incident blockages and recommended bypass routes</p>
            </div>
          </div>
          <button className="btn-report-sm" onClick={onOpenReportModal}>
            <Camera size={14} /> Submit New Report
          </button>
        </div>

        {activeIncidents.length === 0 ? (
          <div className="empty-advisories">
            <CheckCircle2 size={36} className="text-emerald opacity-75" />
            <h4>All Commuter Corridors Operating Smoothly</h4>
            <p>No active collisions, waterlogging, or detours reported across monitored junctions.</p>
          </div>
        ) : (
          <div className="public-advisories-grid">
            {activeIncidents.map(inc => (
              <div key={inc.incident_id} className="public-advisory-pill">
                <div className="advisory-top">
                  <span className="badge-hazard-type font-mono">{inc.incident_type.replace('_', ' ')}</span>
                  <span className="badge-junc font-mono"><MapPin size={11} /> {inc.junction_id} • {inc.approach}</span>
                </div>
                <div className="advisory-road font-bold">{inc.road_name}</div>
                <div className="advisory-desc">{inc.description}</div>
                {inc.diversion_plan && (
                  <div className="advisory-detour-route">
                    <Navigation size={13} className="text-amber" />
                    <span>Recommended Detour: <strong>{inc.diversion_plan.recommended_reroute_corridor}</strong></span>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
