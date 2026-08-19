import React from 'react';
import { Compass, Video, Car, Bike, Bus, Truck } from 'lucide-react';

const LEVEL_CLASS_MAP = {
  LOW: 'badge-low',
  MEDIUM: 'badge-medium',
  HIGH: 'badge-high',
  'VERY HIGH': 'badge-very-high'
};

export default function ApproachFeedCard({ approachName, state }) {
  const level = state?.traffic_level || 'LOW';
  const badgeClass = LEVEL_CLASS_MAP[level] || 'badge-low';

  const classCounts = state?.class_counts || { car: 0, motorcycle: 0, bus: 0, truck: 0 };

  return (
    <div className="approach-card">
      <div className="approach-card-header">
        <div className="approach-title">
          <Compass size={18} color="#38bdf8" />
          <span>{approachName} APPROACH</span>
        </div>
        <span className={`badge-level ${badgeClass}`}>
          ● {level}
        </span>
      </div>

      <div className="video-container">
        {state?.annotated_video_url ? (
          <video 
            key={state.annotated_video_url}
            className="video-player" 
            controls 
            autoPlay 
            loop 
            muted
            playsInline
          >
            <source src={state.annotated_video_url} type="video/mp4" />
            Your browser does not support the video tag.
          </video>
        ) : (
          <div className="video-placeholder">
            <Video size={36} />
            <span>No video processed for this approach yet</span>
          </div>
        )}
      </div>

      <div className="metrics-strip">
        <div className="metric-cell">
          <div className="metric-val">{state ? state.vehicle_count : '—'}</div>
          <div className="metric-lbl">Active Vehicles</div>
        </div>
        <div className="metric-cell">
          <div className="metric-val" style={{ color: '#f59e0b' }}>
            {state ? `~${state.estimated_queue_length}` : '—'}
          </div>
          <div className="metric-lbl">Est. Queue</div>
        </div>
        <div className="metric-cell">
          <div className="metric-val" style={{ color: '#38bdf8' }}>
            {state ? Math.round(state.flow) : '—'}
          </div>
          <div className="metric-lbl">Flow Count</div>
        </div>
        <div className="metric-cell">
          <div className="metric-val" style={{ color: '#818cf8' }}>
            {state ? Number(state.density).toFixed(2) : '—'}
          </div>
          <div className="metric-lbl">Density Index</div>
        </div>
      </div>

      <div className="classes-strip">
        <div className="class-badge">
          <Car size={14} /> Cars: <span className="class-count">{classCounts.car || 0}</span>
        </div>
        <div className="class-badge">
          <Bike size={14} /> Bikes: <span className="class-count">{classCounts.motorcycle || 0}</span>
        </div>
        <div className="class-badge">
          <Bus size={14} /> Bus: <span className="class-count">{classCounts.bus || 0}</span>
        </div>
        <div className="class-badge">
          <Truck size={14} /> Truck: <span className="class-count">{classCounts.truck || 0}</span>
        </div>
      </div>
    </div>
  );
}
