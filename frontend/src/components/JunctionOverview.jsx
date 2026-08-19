import React from 'react';
import { Layers, ArrowUp, ArrowDown, ArrowLeft, ArrowRight } from 'lucide-react';

const LEVEL_CLASS_MAP = {
  LOW: 'badge-low',
  MEDIUM: 'badge-medium',
  HIGH: 'badge-high',
  'VERY HIGH': 'badge-very-high'
};

export default function JunctionOverview({ junctionState }) {
  const aggregateLevel = junctionState?.aggregate_level || 'LOW';
  const totalVehicles = junctionState?.total_active_vehicles || 0;

  const approaches = [
    { key: 'north', name: 'North Approach', icon: ArrowUp, data: junctionState?.north },
    { key: 'south', name: 'South Approach', icon: ArrowDown, data: junctionState?.south },
    { key: 'east', name: 'East Approach', icon: ArrowRight, data: junctionState?.east },
    { key: 'west', name: 'West Approach', icon: ArrowLeft, data: junctionState?.west },
  ];

  return (
    <div className="junction-overview-card">
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '1rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <div style={{ background: 'rgba(56, 189, 248, 0.15)', padding: '0.6rem', borderRadius: 'var(--radius-md)' }}>
            <Layers size={22} color="#38bdf8" />
          </div>
          <div>
            <h2 style={{ fontSize: '1.1rem', fontWeight: 700 }}>Junction Traffic State Aggregation</h2>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
              Unified 4-Way Approach State Matrix for Intersection: <strong>{junctionState?.junction_id || 'N/A'}</strong>
            </p>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <div style={{ textAlign: 'right' }}>
            <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', textTransform: 'uppercase', fontWeight: 600 }}>
              Total Active Vehicles
            </div>
            <div style={{ fontSize: '1.35rem', fontWeight: 800, color: '#38bdf8' }}>
              {totalVehicles}
            </div>
          </div>

          <div style={{ textAlign: 'right' }}>
            <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', textTransform: 'uppercase', fontWeight: 600 }}>
              Aggregate Status
            </div>
            <span className={`badge-level ${LEVEL_CLASS_MAP[aggregateLevel] || 'badge-low'}`}>
              ● {aggregateLevel}
            </span>
          </div>
        </div>
      </div>

      <div className="junction-state-grid">
        {approaches.map(({ key, name, icon: Icon, data }) => (
          <div 
            key={key} 
            style={{ 
              background: 'var(--bg-card)', 
              border: '1px solid var(--border-color)', 
              borderRadius: 'var(--radius-md)', 
              padding: '1rem' 
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
              <span style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontWeight: 600, fontSize: '0.85rem' }}>
                <Icon size={16} color="#38bdf8" /> {name}
              </span>
              <span className={`badge-level ${LEVEL_CLASS_MAP[data?.traffic_level || 'LOW'] || 'badge-low'}`} style={{ fontSize: '0.65rem' }}>
                {data?.traffic_level || 'NO FEED'}
              </span>
            </div>

            <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
              <div>Vehicles: <strong style={{ color: 'var(--text-primary)' }}>{data?.vehicle_count ?? '—'}</strong></div>
              <div>Est. Queue: <strong style={{ color: '#f59e0b' }}>{data ? `~${data.estimated_queue_length} veh` : '—'}</strong></div>
              <div>Flow: <strong style={{ color: '#38bdf8' }}>{data ? Math.round(data.flow) : '—'}</strong></div>
              <div>Density: <strong style={{ color: '#818cf8' }}>{data ? Number(data.density).toFixed(2) : '—'}</strong></div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
