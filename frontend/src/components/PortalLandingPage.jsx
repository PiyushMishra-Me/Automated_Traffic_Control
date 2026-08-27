import React from 'react';
import { 
  User, 
  Activity, 
  Building2, 
  ShieldCheck, 
  Camera, 
  Flame, 
  Truck, 
  Radio, 
  ArrowRight, 
  Lock, 
  Unlock, 
  Cpu, 
  Sparkles,
  MapPin,
  Clock,
  ShieldAlert,
  Sliders
} from 'lucide-react';

export default function PortalLandingPage({ onSelectPortal }) {
  return (
    <div className="landing-page-container animate-fade-in">
      {/* HERO SECTION */}
      <div className="landing-hero">
        <div className="landing-pill">
          <Sparkles size={14} className="text-cyan" />
          <span>Next-Generation Intelligent Urban Traffic OS</span>
        </div>
        <h1 className="landing-title">
          Metropolitan Automated Traffic Control &amp; Emergency Preemption Network
        </h1>
        <p className="landing-subtitle">
          An integrated intelligent transportation system uniting <strong>Public Commuters</strong>, 
          <strong> Emergency First Responders</strong> (Hospital Trauma &amp; Fire Rescue), and 
          <strong> Government Traffic Police Command</strong>.
        </p>
      </div>

      {/* PORTAL PERSONA CARDS */}
      <div className="landing-portals-grid">
        {/* PORTAL 1: PUBLIC CITIZEN */}
        <div className="portal-card public-card" onClick={() => onSelectPortal('PUBLIC_USER')}>
          <div className="portal-card-top">
            <div className="portal-icon-box public-icon">
              <User size={28} />
            </div>
            <span className="portal-access-badge open">
              <Unlock size={11} /> Open Public Access
            </span>
          </div>

          <div className="portal-card-body">
            <h3>Public Citizen &amp; Commuter Portal</h3>
            <p>
              Report on-scene traffic accidents with <strong>mandatory live camera capture</strong>, 
              view live public road congestion maps, and receive real-time weather traction speed advisories.
            </p>

            <div className="portal-feature-list">
              <div className="feat-item"><Camera size={14} className="text-cyan" /> Live Camera On-Scene Hazard Reporting</div>
              <div className="feat-item"><MapPin size={14} className="text-cyan" /> Public Live Traffic Flow &amp; Detour Map</div>
              <div className="feat-item"><Clock size={14} className="text-cyan" /> Weather Speed Advisories &amp; Road Alerts</div>
            </div>
          </div>

          <div className="portal-card-footer">
            <button className="btn-portal-enter public-btn">
              Enter Public Portal <ArrowRight size={16} />
            </button>
          </div>
        </div>

        {/* PORTAL 2: EMERGENCY SERVICES (HOSPITAL & FIRE) */}
        <div className="portal-card emergency-card" onClick={() => onSelectPortal('HOSPITAL_DISPATCH')}>
          <div className="portal-card-top">
            <div className="portal-icon-box emergency-icon">
              <Activity size={28} />
            </div>
            <span className="portal-access-badge auth-needed">
              <Lock size={11} /> Encrypted &amp; Authenticated
            </span>
          </div>

          <div className="portal-card-body">
            <h3>Emergency Services Network</h3>
            <p>
              Dedicated encrypted infrastructure for <strong>Hospitals, Fire &amp; Rescue, and Disaster Convoys</strong>. 
              Launch emergency units with 4-tier criticality preemption and dynamic Green Wave corridors.
            </p>

            <div className="sub-agencies-badge-row">
              <span className="sub-badge hosp">🏥 Hospital Trauma</span>
              <span className="sub-badge fire">🚒 Fire &amp; Rescue</span>
              <span className="sub-badge police">🚓 Police / Disaster Convoy</span>
            </div>

            <div className="portal-feature-list">
              <div className="feat-item"><Truck size={14} className="text-emerald" /> Emergency Mission Dispatch &amp; Routing</div>
              <div className="feat-item"><Radio size={14} className="text-emerald" /> Priority 1–4 Criticality Signal Overrides</div>
              <div className="feat-item"><ShieldAlert size={14} className="text-emerald" /> Multi-Ambulance Conflict Arbitration</div>
            </div>
          </div>

          <div className="portal-card-footer">
            <button className="btn-portal-enter emergency-btn">
              Authenticate &amp; Enter Emergency Portal <ArrowRight size={16} />
            </button>
          </div>
        </div>

        {/* PORTAL 3: GOVERNMENT & POLICE COMMAND */}
        <div className="portal-card gov-card" onClick={() => onSelectPortal('GOVERNMENT_OFFICIAL')}>
          <div className="portal-card-top">
            <div className="portal-icon-box gov-icon">
              <Building2 size={28} />
            </div>
            <span className="portal-access-badge gov-needed">
              <ShieldCheck size={11} /> High Security Clearance
            </span>
          </div>

          <div className="portal-card-body">
            <h3>Traffic Police &amp; Operations Command</h3>
            <p>
              Master administrative command center with unrestricted control over 
              <strong> 4-approach YOLOv8 vision feeds</strong>, adaptive signal timing algorithms, 
              automated accident detours, and longitudinal audit telemetry.
            </p>

            <div className="portal-feature-list">
              <div className="feat-item"><Cpu size={14} className="text-blue" /> 5-Junction Master Matrix &amp; Camera Feeds</div>
              <div className="feat-item"><Sliders size={14} className="text-blue" /> Adaptive Signal Controller &amp; PCU Simulator</div>
              <div className="feat-item"><ShieldAlert size={14} className="text-blue" /> Upstream Detour Operations &amp; Weather Overrides</div>
            </div>
          </div>

          <div className="portal-card-footer">
            <button className="btn-portal-enter gov-btn">
              Authorize &amp; Enter Command Center <ArrowRight size={16} />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
