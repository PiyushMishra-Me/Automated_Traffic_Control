import React from 'react';
import { Activity, ShieldCheck, Cpu, AlertTriangle, CloudRain, Map } from 'lucide-react';

export default function Navbar({ activeIncidentCount = 0, onOpenReportModal }) {
  return (
    <header className="app-header">
      <div className="brand-section">
        <div className="brand-icon"><Activity size={24} color="#ffffff" /></div>
        <div className="brand-title">
          <h1>Intelligent Traffic Management &amp; Safety Control</h1>
          <p>Real-Time Vision AI, Geospatial City Map &amp; Adaptive Signal Automation</p>
        </div>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
        <span className="header-badge" style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          <Cpu size={14} /> YOLO11 + ByteTrack
        </span>
        <span className="header-badge" style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: '#06b6d4' }}>
          <Map size={14} /> Live City Grid
        </span>
        
        {activeIncidentCount > 0 ? (
          <button 
            className="header-badge badge-hazard-alert" 
            onClick={onOpenReportModal}
            style={{ 
              display: 'flex', 
              alignItems: 'center', 
              gap: '0.4rem', 
              background: 'rgba(239, 68, 68, 0.2)', 
              color: '#ef4444', 
              borderColor: 'rgba(239, 68, 68, 0.5)',
              cursor: 'pointer'
            }}
          >
            <AlertTriangle size={14} /> {activeIncidentCount} Active Incident(s)
          </button>
        ) : (
          <button 
            className="header-badge" 
            onClick={onOpenReportModal}
            style={{ 
              display: 'flex', 
              alignItems: 'center', 
              gap: '0.4rem', 
              background: 'rgba(245, 158, 11, 0.15)', 
              color: '#f59e0b', 
              borderColor: 'rgba(245, 158, 11, 0.3)',
              cursor: 'pointer'
            }}
          >
            <AlertTriangle size={14} /> + Report Incident
          </button>
        )}

        <span className="header-badge" style={{ background: 'rgba(16, 185, 129, 0.1)', color: '#10b981', borderColor: 'rgba(16, 185, 129, 0.3)' }}>
          <ShieldCheck size={14} /> Adaptive Active
        </span>
      </div>
    </header>
  );
}
