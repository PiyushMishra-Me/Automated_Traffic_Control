import React from 'react';
import { Compass, Video, Car, Bike, Bus, Truck, Siren } from 'lucide-react';

const LEVEL_CLASS_MAP = {
  LOW: 'badge-low',
  MEDIUM: 'badge-medium',
  HIGH: 'badge-high',
  'VERY HIGH': 'badge-very-high'
};

export default function ApproachFeedCard({ approachName, state, liveStream, junctionId }) {
  const level = state?.traffic_level || 'LOW';
  const badgeClass = LEVEL_CLASS_MAP[level] || 'badge-low';

  const classCounts = state?.class_counts || { car: 0, motorcycle: 0, bus: 0, truck: 0 };
  const ambulanceCount = classCounts.ambulance || state?.ambulance_count || 0;

  const [streamError, setStreamError] = React.useState(false);
  const jId = junctionId || liveStream?.junction_id;
  const isLiveActive = Boolean(liveStream && liveStream.is_active && liveStream.stream_url);
  const liveInferenceUrl = jId ? `/api/videos/live/${jId}/${approachName}/annotated-stream` : null;

  // Reset stream error if live stream config changes
  React.useEffect(() => {
    setStreamError(false);
  }, [liveStream?.stream_url, jId]);

  return (
    <div className="approach-card">
      <div className="approach-card-header">
        <div className="approach-title">
          <Compass size={18} color="#38bdf8" />
          <span>{approachName} APPROACH</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          {isLiveActive && (
            <span className="badge-level" style={{ background: 'rgba(16, 185, 129, 0.15)', color: '#10b981', border: '1px solid rgba(16, 185, 129, 0.3)' }}>
              ● LIVE AI FEED
            </span>
          )}
          {ambulanceCount > 0 && (
            <span className="badge-level" style={{ background: 'rgba(239, 68, 68, 0.2)', color: '#f87171', border: '1px solid rgba(239, 68, 68, 0.5)' }}>
              ⚡ AMBULANCE DETECTED
            </span>
          )}
          <span className={`badge-level ${badgeClass}`}>
            ● {level}
          </span>
        </div>
      </div>

      <div className="video-container">
        {isLiveActive ? (
          <div style={{ position: 'relative', width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#000' }}>
            <img 
              src={!streamError && liveInferenceUrl ? liveInferenceUrl : liveStream.stream_url} 
              alt={`${approachName} Live Vision AI Feed`} 
              className="video-player"
              style={{ width: '100%', height: '100%', objectFit: 'contain' }}
              onError={() => {
                if (!streamError) setStreamError(true);
              }}
            />
            <div style={{ position: 'absolute', top: 10, left: 10, background: 'rgba(0,0,0,0.8)', color: '#10b981', padding: '4px 8px', borderRadius: 4, fontSize: '0.72rem', fontWeight: 800, border: '1px solid rgba(16,185,129,0.35)', display: 'flex', alignItems: 'center', gap: '5px' }}>
              <span style={{ width: 6, height: 6, borderRadius: '50%', background: '#10b981', display: 'inline-block' }} /> 
              {!streamError ? 'LIVE YOLO + AMBULANCE AI' : 'LIVE RAW STREAM'}
            </div>
          </div>
        ) : state?.annotated_video_url ? (
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
            <span>No video feed connected for {approachName} Approach</span>
            <span style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
              Upload an approach video or connect an RTSP/HTTP stream above.
            </span>
          </div>
        )}
      </div>

      <div className="metrics-strip">
        <div className="metric-cell">
          <div className="metric-val">{state ? state.vehicle_count : '—'}</div>
          <div className="metric-lbl">Peak Vehicles</div>
        </div>
        <div className="metric-cell">
          <div className="metric-val" style={{ color: '#f59e0b' }}>
            {state ? `~${state.estimated_queue_length}` : '—'}
          </div>
          <div className="metric-lbl">Peak Queue</div>
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
          <div className="metric-lbl">Avg. Density</div>
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
        {ambulanceCount > 0 && (
          <div className="class-badge" style={{ borderColor: 'rgba(239, 68, 68, 0.4)', background: 'rgba(239, 68, 68, 0.15)', color: '#f87171' }}>
            <Siren size={14} /> Ambulance: <span className="class-count" style={{ color: '#f87171' }}>{ambulanceCount}</span>
          </div>
        )}
      </div>
    </div>
  );
}
