import React from 'react';
import { Activity, MapPin, RefreshCw, Sparkles } from 'lucide-react';

export default function DashboardHero({ junction, state, onRefresh }) {
  const level = state?.aggregate_level || 'AWAITING DATA';
  return (
    <section className="dashboard-hero">
      <div className="hero-orb hero-orb-one" /><div className="hero-orb hero-orb-two" />
      <div className="hero-content">
        <div className="eyebrow"><Sparkles size={14} /> Traffic operations workspace</div>
        <h2>{junction?.name || 'Traffic control centre'}</h2>
        <p><MapPin size={15} /> {junction?.location || 'Select a junction to begin monitoring'}</p>
      </div>
      <div className="hero-statuses">
        <div className="hero-stat"><span>Current load</span><strong>{state?.total_active_vehicles ?? 0}</strong><small>active vehicles</small></div>
        <div className="hero-stat"><span>Network status</span><strong className="status-value"><i /> {level}</strong><small>latest processed data</small></div>
        <button className="icon-text-button" onClick={onRefresh}><RefreshCw size={15} /> Refresh</button>
      </div>
    </section>
  );
}
