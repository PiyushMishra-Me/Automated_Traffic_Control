import React from 'react';
import { 
  ShieldAlert, 
  Camera, 
  MapPin, 
  AlertTriangle, 
  CloudRain, 
  Compass, 
  CheckCircle2, 
  Navigation,
  Sparkles,
  ArrowRight,
  Info
} from 'lucide-react';
import LiveJunctionMap from './LiveJunctionMap';
import WeatherWidget from './WeatherWidget';

export default function PublicCitizenPortal({ 
  junctions = [], 
  selectedJunction, 
  onSelectJunction, 
  incidents = [], 
  weatherData, 
  onOpenReportModal,
  onWeatherUpdated
}) {
  const activeIncidents = incidents.filter(i => i.status === 'ACTIVE');

  return (
    <div className="public-portal-container animate-fade-in">
      {/* PUBLIC HERO & REPORT CALL TO ACTION */}
      <div className="public-hero-card">
        <div className="public-hero-left">
          <div className="public-tag">
            <Sparkles size={14} className="text-cyan" /> Citizen Traffic Safety &amp; Incident Portal
          </div>
          <h2>Real-Time Commuter Map &amp; Hazard Reporting</h2>
          <p>
            Help keep your city moving safely. Report road accidents or obstructions on the spot using your 
            <strong> live camera viewfinder</strong>. Our AI system instantly computes upstream detours to prevent gridlock.
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

      {/* WEATHER & ROAD TRACTION ADVISORY */}
      <WeatherWidget 
        junctionId={selectedJunction}
        weatherData={weatherData}
        onWeatherUpdated={onWeatherUpdated}
      />

      {/* LIVE ROAD & DETOUR MAP */}
      <div className="public-map-wrapper">
        <div className="public-section-header">
          <div className="title-with-icon">
            <Compass size={20} className="text-cyan" />
            <div>
              <h3>Live Metropolitan Traffic Flow &amp; Detour Map</h3>
              <p className="card-subtitle">Real-time road corridor congestion, active diversions, and weather radar</p>
            </div>
          </div>
        </div>

        <LiveJunctionMap 
          junctions={junctions}
          selectedJunction={selectedJunction}
          onSelectJunction={onSelectJunction}
          incidents={incidents}
          weatherData={weatherData}
          onOpenReportModal={onOpenReportModal}
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
