import React, { useState, useEffect } from 'react';
import { 
  Building2, 
  ShieldCheck, 
  Radio, 
  Video, 
  Cpu, 
  AlertTriangle, 
  Activity, 
  Sliders, 
  RotateCcw, 
  Eye, 
  CheckCircle2, 
  Flame, 
  Lock,
  ChevronRight,
  Server,
  Map,
  BarChart3,
  CloudRain,
  ShieldAlert,
  Sparkles
} from 'lucide-react';
import LiveJunctionMap from './LiveJunctionMap';
import VideoUploader from './VideoUploader';
import ApproachFeedCard from './ApproachFeedCard';
import JunctionSelector from './JunctionSelector';
import JunctionOverview from './JunctionOverview';
import CountingLineEditor from './CountingLineEditor';
import SignalSimulator from './SignalSimulator';
import IncidentManager from './IncidentManager';
import AnalyticsHistory from './AnalyticsHistory';
import WeatherWidget from './WeatherWidget';
import { api } from '../services/api';

export default function GovernmentCommandPortal({ 
  junctions = [], 
  selectedJunction, 
  onSelectJunction, 
  userSession,
  onOpenReportModal,
  weatherData,
  onWeatherUpdated,
  analyticsRefresh,
  onRefreshAll,
  onJobCompleted
}) {
  const [govActiveTab, setGovActiveTab] = useState('overview'); // 'overview', 'map', 'vision', 'simulation', 'incidents', 'weather', 'analytics'
  const [junctionStates, setJunctionStates] = useState({});
  const [liveStreams, setLiveStreams] = useState({});
  const [activeAmbulances, setActiveAmbulances] = useState([]);
  const [activeIncidents, setActiveIncidents] = useState([]);
  const [loading, setLoading] = useState(false);

  const selectedJunctionData = junctions.find((j) => j.junction_id === selectedJunction);
  const currentJunctionState = junctionStates[selectedJunction];

  const fetchLiveStreams = async (jId) => {
    if (!jId) return;
    try {
      const streams = await api.getJunctionLiveStreams(jId);
      setLiveStreams(streams || {});
    } catch (e) {
      console.warn('Failed to load live streams', e);
    }
  };

  const fetchAllCommandData = async () => {
    try {
      setLoading(true);
      const [ambList, incList] = await Promise.all([
        api.listAmbulances(),
        api.listIncidents()
      ]);
      setActiveAmbulances(ambList.filter(a => a.status !== 'MISSION_ACCOMPLISHED'));
      setActiveIncidents(incList.filter(i => i.status === 'ACTIVE'));

      const statesObj = {};
      for (const j of junctions) {
        try {
          const s = await api.getJunctionState(j.junction_id);
          statesObj[j.junction_id] = s;
        } catch (e) {}
      }
      setJunctionStates(statesObj);
    } catch (err) {
      console.error('Failed to fetch command telemetry', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAllCommandData();
    const interval = setInterval(fetchAllCommandData, 6000);
    return () => clearInterval(interval);
  }, [junctions, analyticsRefresh]);

  useEffect(() => {
    if (selectedJunction) {
      fetchLiveStreams(selectedJunction);
    }
  }, [selectedJunction, analyticsRefresh]);

  const getLevelBadgeClass = (level) => {
    switch (level) {
      case 'VERY_HIGH': return 'badge-critical';
      case 'HIGH': return 'badge-severe';
      case 'MEDIUM': return 'badge-moderate';
      default: return 'badge-minor';
    }
  };

  return (
    <div className="gov-command-container animate-fade-in">
      {/* GOVERNMENT HERO */}
      <div className="portal-hero gov-theme">
        <div className="hero-left">
          <div className="portal-badge gov">
            <ShieldCheck size={16} /> Traffic Police &amp; Municipal Operations Command
          </div>
          <h2>Metropolitan Traffic Operating Command System</h2>
          <p>
            Full administrative access for <strong>{userSession?.organization_name || 'Traffic Police Operations'}</strong>. 
            Supervise city-wide YOLOv8n vision telemetry, execute manual signal overrides, manage upstream accident detours, 
            and supervise multi-agency emergency Green Waves.
          </p>
        </div>
        <div className="hero-right">
          <div className="command-stat-box">
            <div className="stat-num font-mono text-cyan">{junctions.length}</div>
            <div className="stat-lbl">Active Intersections</div>
          </div>
          <div className="command-stat-box">
            <div className="stat-num font-mono text-emerald">{activeAmbulances.length}</div>
            <div className="stat-lbl">Active Green Waves</div>
          </div>
          <div className="command-stat-box">
            <div className="stat-num font-mono text-red">{activeIncidents.length}</div>
            <div className="stat-lbl">Active Detours</div>
          </div>
        </div>
      </div>

      {/* GOVERNMENT MASTER MODULE NAVIGATION TABS */}
      <div className="dashboard-nav-tabs gov-nav-tabs">
        <button 
          className={`tab-btn ${govActiveTab === 'overview' ? 'active' : ''}`}
          onClick={() => setGovActiveTab('overview')}
        >
          <Building2 size={16} /> City Matrix Overview
        </button>

        <button 
          className={`tab-btn ${govActiveTab === 'map' ? 'active' : ''}`}
          onClick={() => setGovActiveTab('map')}
        >
          <Map size={16} /> Live Geospatial Map
        </button>

        <button 
          className={`tab-btn ${govActiveTab === 'vision' ? 'active' : ''}`}
          onClick={() => setGovActiveTab('vision')}
        >
          <Video size={16} /> Vision AI Feeds &amp; Uploader
        </button>

        <button 
          className={`tab-btn ${govActiveTab === 'simulation' ? 'active' : ''}`}
          onClick={() => setGovActiveTab('simulation')}
        >
          <Cpu size={16} /> Adaptive Signal Simulator
        </button>

        <button 
          className={`tab-btn ${govActiveTab === 'incidents' ? 'active' : ''}`}
          onClick={() => setGovActiveTab('incidents')}
        >
          <ShieldAlert size={16} /> Detour &amp; Incident Operations
          {activeIncidents.length > 0 && <span className="tab-badge-count">{activeIncidents.length}</span>}
        </button>

        <button 
          className={`tab-btn ${govActiveTab === 'weather' ? 'active' : ''}`}
          onClick={() => setGovActiveTab('weather')}
        >
          <CloudRain size={16} /> Weather &amp; Safety Bounds
        </button>

        <button 
          className={`tab-btn ${govActiveTab === 'analytics' ? 'active' : ''}`}
          onClick={() => setGovActiveTab('analytics')}
        >
          <BarChart3 size={16} /> Historical Analytics
        </button>
      </div>

      {/* TAB 1: MASTER CITY MATRIX OVERVIEW */}
      {govActiveTab === 'overview' && (
        <div className="tab-view-container animate-fade-in">
          <div className="command-grid-section">
            <div className="section-title-row">
              <div>
                <h3>Metropolitan 5-Junction Real-Time Matrix</h3>
                <p className="card-subtitle">AI vehicle load, congestion levels, and live emergency preemption states</p>
              </div>
              <button className="btn-refresh-sm" onClick={fetchAllCommandData}>
                <RotateCcw size={14} /> Refresh Matrix
              </button>
            </div>

            <div className="junction-matrix-grid">
              {junctions.map((j) => {
                const st = junctionStates[j.junction_id];
                const isSelected = selectedJunction === j.junction_id;
                const hasAmbulance = activeAmbulances.some(a => a.route_corridor.some(n => n.junction_id === j.junction_id));
                const hasIncident = activeIncidents.some(i => i.junction_id === j.junction_id);

                return (
                  <div 
                    key={j.junction_id} 
                    className={`junction-matrix-card ${isSelected ? 'selected-matrix' : ''}`}
                    onClick={() => onSelectJunction(j.junction_id)}
                  >
                    <div className="matrix-card-header">
                      <div>
                        <span className="j-badge font-mono">{j.junction_id}</span>
                        <strong className="j-name">{j.name}</strong>
                      </div>
                      {hasAmbulance && <span className="beacon-pill-amb">⚡ GREEN WAVE</span>}
                      {hasIncident && <span className="beacon-pill-inc">⚠️ DETOUR</span>}
                    </div>

                    <div className="matrix-stats-row">
                      <div className="matrix-stat">
                        <span className="label">Total Vehicles</span>
                        <span className="value font-mono">{st ? st.total_active_vehicles : '—'}</span>
                      </div>
                      <div className="matrix-stat">
                        <span className="label">Congestion</span>
                        <span className={`value-badge ${st ? getLevelBadgeClass(st.aggregate_level) : ''}`}>
                          {st ? st.aggregate_level : 'IDLE'}
                        </span>
                      </div>
                    </div>

                    <div className="matrix-approaches-mini">
                      <div className={`app-mini-pill ${st?.north ? 'active' : ''}`}>N: {st?.north?.vehicle_count || 0}</div>
                      <div className={`app-mini-pill ${st?.south ? 'active' : ''}`}>S: {st?.south?.vehicle_count || 0}</div>
                      <div className={`app-mini-pill ${st?.east ? 'active' : ''}`}>E: {st?.east?.vehicle_count || 0}</div>
                      <div className={`app-mini-pill ${st?.west ? 'active' : ''}`}>W: {st?.west?.vehicle_count || 0}</div>
                    </div>

                    <button 
                      className="btn-inspect-junction"
                      onClick={(e) => {
                        e.stopPropagation();
                        onSelectJunction(j.junction_id);
                        setGovActiveTab('vision');
                      }}
                    >
                      Open Vision AI Feeds <ChevronRight size={14} />
                    </button>
                  </div>
                );
              })}
            </div>
          </div>

          {/* DUAL COMMAND PANELS */}
          <div className="command-dual-panels">
            <div className="command-panel">
              <div className="panel-title">
                <Activity size={18} className="text-emerald" />
                <h4>Active Green Wave Emergency Corridors ({activeAmbulances.length})</h4>
              </div>
              {activeAmbulances.length === 0 ? (
                <div className="panel-empty-text">Zero emergency vehicles currently requesting signal preemption.</div>
              ) : (
                <div className="panel-items-list">
                  {activeAmbulances.map(a => (
                    <div key={a.mission_id} className="panel-log-item green-border">
                      <div className="log-top">
                        <span className="font-mono text-cyan font-bold">{a.mission_id}</span>
                        <span className="crit-pill font-mono">{a.criticality}</span>
                      </div>
                      <div className="log-desc">
                        <strong>{a.hospital_name}</strong> ({a.ambulance_vehicle_id}): {a.patient_condition}
                      </div>
                      <div className="log-route font-mono text-muted">
                        Corridor: {a.origin_junction_id} ➔ {a.destination_junction_id}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div className="command-panel">
              <div className="panel-title">
                <AlertTriangle size={18} className="text-red" />
                <h4>Active Road Hazard Detours ({activeIncidents.length})</h4>
              </div>
              {activeIncidents.length === 0 ? (
                <div className="panel-empty-text">All corridors clear. Zero road blockages reported.</div>
              ) : (
                <div className="panel-items-list">
                  {activeIncidents.map(inc => (
                    <div key={inc.incident_id} className="panel-log-item red-border">
                      <div className="log-top">
                        <span className="font-mono text-red font-bold">{inc.incident_id}</span>
                        <span className="font-mono">{inc.junction_id} • {inc.approach}</span>
                      </div>
                      <div className="log-desc">
                        <strong>{inc.road_name}</strong>: {inc.description}
                      </div>
                      <div className="log-route font-mono text-amber">
                        Detour: {inc.diversion_plan?.recommended_reroute_corridor}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* TAB 2: LIVE GEOSPATIAL MAP */}
      {govActiveTab === 'map' && (
        <div className="tab-view-container animate-fade-in">
          <LiveJunctionMap 
            junctions={junctions}
            selectedJunction={selectedJunction}
            onSelectJunction={onSelectJunction}
            incidents={activeIncidents}
            activeAmbulances={activeAmbulances}
            weatherData={weatherData}
            onOpenReportModal={onOpenReportModal}
          />
          <JunctionOverview junctionState={currentJunctionState} />
        </div>
      )}

      {/* TAB 3: VISION AI FEEDS & UPLOADER */}
      {govActiveTab === 'vision' && (
        <div className="tab-view-container animate-fade-in">
          <div className="top-controls-grid">
            <JunctionSelector 
              junctions={junctions}
              selectedJunction={selectedJunction}
              onSelectJunction={onSelectJunction}
              onRefresh={onRefreshAll}
            />

            <VideoUploader 
              junctionId={selectedJunction}
              junctions={junctions}
              onSelectJunction={onSelectJunction}
              onJobCompleted={onJobCompleted}
            />

            <CountingLineEditor
              junction={selectedJunctionData}
              onSaved={(updated) => onRefreshAll()}
            />
          </div>

          <section className="workspace-section">
            <div className="section-heading">
              <div>
                <span className="eyebrow">Vision AI Workspace</span>
                <h2>4-Approach Camera Feeds &amp; Detection Overlay ({selectedJunction})</h2>
              </div>
              <p>YOLOv8n neural detection, ByteTrack tracking IDs, and directional analytics.</p>
            </div>

            <div className="approaches-grid">
              <ApproachFeedCard approachName="NORTH" state={currentJunctionState?.north} liveStream={liveStreams?.NORTH} junctionId={selectedJunction} />
              <ApproachFeedCard approachName="SOUTH" state={currentJunctionState?.south} liveStream={liveStreams?.SOUTH} junctionId={selectedJunction} />
              <ApproachFeedCard approachName="EAST" state={currentJunctionState?.east} liveStream={liveStreams?.EAST} junctionId={selectedJunction} />
              <ApproachFeedCard approachName="WEST" state={currentJunctionState?.west} liveStream={liveStreams?.WEST} junctionId={selectedJunction} />
            </div>
          </section>
        </div>
      )}

      {/* TAB 4: ADAPTIVE SIGNAL SIMULATOR */}
      {govActiveTab === 'simulation' && (
        <div className="tab-view-container animate-fade-in">
          <SignalSimulator 
            junctionId={selectedJunction} 
            junctions={junctions}
            onSelectJunction={onSelectJunction}
            refreshKey={analyticsRefresh} 
          />
        </div>
      )}

      {/* TAB 5: DETOUR & INCIDENT OPERATIONS */}
      {govActiveTab === 'incidents' && (
        <div className="tab-view-container animate-fade-in">
          <IncidentManager 
            incidents={activeIncidents}
            onRefresh={fetchAllCommandData}
            onOpenReportModal={onOpenReportModal}
            selectedJunction={selectedJunction}
            onSelectJunction={onSelectJunction}
          />
        </div>
      )}

      {/* TAB 6: WEATHER & SAFETY BOUNDS */}
      {govActiveTab === 'weather' && (
        <div className="tab-view-container animate-fade-in">
          <WeatherWidget 
            junctionId={selectedJunction}
            weatherData={weatherData}
            onWeatherUpdated={onWeatherUpdated}
          />
        </div>
      )}

      {/* TAB 7: HISTORICAL ANALYTICS */}
      {govActiveTab === 'analytics' && (
        <div className="tab-view-container animate-fade-in">
          <AnalyticsHistory 
            junctionId={selectedJunction} 
            refreshKey={analyticsRefresh} 
          />
        </div>
      )}
    </div>
  );
}
