import React from 'react';
import { Activity, ShieldCheck, Cpu } from 'lucide-react';

export default function Navbar() {
  return (
    <header className="app-header">
      <div className="brand-section">
        <div className="brand-icon"><Activity size={24} color="#ffffff" /></div>
        <div className="brand-title">
          <h1>Intelligent Traffic Management System</h1>
          <p>Phase 3 — Adaptive Signal Simulation &amp; Traffic Analytics</p>
        </div>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
        <span className="header-badge" style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}><Cpu size={14} /> YOLOv8n + ByteTrack</span>
        <span className="header-badge" style={{ background: 'rgba(16, 185, 129, 0.1)', color: '#10b981', borderColor: 'rgba(16, 185, 129, 0.3)' }}><ShieldCheck size={14} /> Simulation Ready</span>
      </div>
    </header>
  );
}
