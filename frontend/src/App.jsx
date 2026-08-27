import React, { useState, useEffect } from 'react';
import { 
  Map, 
  Video, 
  Cpu, 
  ShieldAlert, 
  BarChart3, 
  Radio, 
  Plus, 
  Sparkles,
  Sliders
} from 'lucide-react';
import Navbar from './components/Navbar';
import JunctionSelector from './components/JunctionSelector';
import VideoUploader from './components/VideoUploader';
import ApproachFeedCard from './components/ApproachFeedCard';
import JunctionOverview from './components/JunctionOverview';
import AnalyticsHistory from './components/AnalyticsHistory';
import CountingLineEditor from './components/CountingLineEditor';
import SignalSimulator from './components/SignalSimulator';
import DashboardHero from './components/DashboardHero';
import LiveJunctionMap from './components/LiveJunctionMap';
import IncidentManager from './components/IncidentManager';
import IncidentReportingModal from './components/IncidentReportingModal';
import WeatherWidget from './components/WeatherWidget';
import { api } from './services/api';

export default function App() {
  const [junctions, setJunctions] = useState([]);
  const [selectedJunction, setSelectedJunction] = useState('J-01');
  const [junctionState, setJunctionState] = useState(null);
  const [incidents, setIncidents] = useState([]);
  const [weatherData, setWeatherData] = useState(null);
  const [isReportModalOpen, setIsReportModalOpen] = useState(false);
  const [analyticsRefresh, setAnalyticsRefresh] = useState(0);

  // Active Dashboard Navigation Tab
  const [activeTab, setActiveTab] = useState('map'); // 'map', 'vision', 'simulation', 'incidents', 'analytics'

  const selectedJunctionData = junctions.find((j) => j.junction_id === selectedJunction);
  const activeIncidentCount = incidents.filter(i => i.status === 'ACTIVE').length;

  const fetchJunctions = async () => {
    try {
      const list = await api.listJunctions();
      setJunctions(list);
      if (list.length > 0 && !selectedJunction) {
        setSelectedJunction(list[0].junction_id);
      }
    } catch (err) {
      console.error('Failed to load junctions', err);
    }
  };

  const fetchJunctionState = async (jId) => {
    if (!jId) return;
    try {
      const state = await api.getJunctionState(jId);
      setJunctionState(state);
    } catch (err) {
      console.error('Failed to fetch junction state', err);
    }
  };

  const fetchIncidents = async () => {
    try {
      const list = await api.listIncidents();
      setIncidents(list);
    } catch (err) {
      console.error('Failed to load incidents', err);
    }
  };

  const fetchWeather = async (jId) => {
    if (!jId) return;
    try {
      const data = await api.getJunctionWeather(jId);
      setWeatherData(data);
    } catch (err) {
      console.error('Failed to fetch weather', err);
    }
  };

  useEffect(() => {
    fetchJunctions();
    fetchIncidents();
  }, []);

  useEffect(() => {
    if (selectedJunction) {
      fetchJunctionState(selectedJunction);
      fetchWeather(selectedJunction);
    }
  }, [selectedJunction]);

  const handleRefreshAll = () => {
    fetchJunctions();
    fetchJunctionState(selectedJunction);
    fetchIncidents();
    fetchWeather(selectedJunction);
    setAnalyticsRefresh((c) => c + 1);
  };

  const handleJobCompleted = (job) => {
    // Refresh junction state after video processing completes
    fetchJunctionState(selectedJunction);
    setAnalyticsRefresh((current) => current + 1);
  };

  return (
    <div className="app-container">
      <Navbar 
        activeIncidentCount={activeIncidentCount} 
        onOpenReportModal={() => setIsReportModalOpen(true)} 
      />

      <DashboardHero
        junction={selectedJunctionData}
        state={junctionState}
        weatherData={weatherData}
        onRefresh={handleRefreshAll}
      />

      {/* Weather Telemetry & Road Traction Bar */}
      <WeatherWidget 
        junctionId={selectedJunction}
        weatherData={weatherData}
        onWeatherUpdated={(updated) => {
          setWeatherData(updated);
          handleRefreshAll();
        }}
      />

      {/* Sleek Minimalist Dashboard Navigation Tabs */}
      <div className="dashboard-nav-tabs">
        <button 
          className={`tab-btn ${activeTab === 'map' ? 'active' : ''}`}
          onClick={() => setActiveTab('map')}
        >
          <Map size={16} /> Live City Grid &amp; Map
        </button>

        <button 
          className={`tab-btn ${activeTab === 'vision' ? 'active' : ''}`}
          onClick={() => setActiveTab('vision')}
        >
          <Video size={16} /> Vision Feeds &amp; Video AI
        </button>

        <button 
          className={`tab-btn ${activeTab === 'simulation' ? 'active' : ''}`}
          onClick={() => setActiveTab('simulation')}
        >
          <Cpu size={16} /> Signal Simulator
        </button>

        <button 
          className={`tab-btn ${activeTab === 'incidents' ? 'active' : ''}`}
          onClick={() => setActiveTab('incidents')}
        >
          <ShieldAlert size={16} /> Incidents &amp; Diversions
          {activeIncidentCount > 0 && <span className="tab-badge-count">{activeIncidentCount}</span>}
        </button>

        <button 
          className={`tab-btn ${activeTab === 'analytics' ? 'active' : ''}`}
          onClick={() => setActiveTab('analytics')}
        >
          <BarChart3 size={16} /> Historical Analytics
        </button>
      </div>

      {/* TAB 1: LIVE GEOSPATIAL MAP */}
      {activeTab === 'map' && (
        <div className="tab-view-container animate-fade-in">
          <LiveJunctionMap 
            junctions={junctions}
            selectedJunction={selectedJunction}
            onSelectJunction={(id) => setSelectedJunction(id)}
            incidents={incidents}
            weatherData={weatherData}
            onOpenReportModal={() => setIsReportModalOpen(true)}
          />

          <JunctionOverview junctionState={junctionState} />
        </div>
      )}

      {/* TAB 2: VISION FEEDS & VIDEO PROCESSING */}
      {activeTab === 'vision' && (
        <div className="tab-view-container animate-fade-in">
          <div className="top-controls-grid">
            <JunctionSelector 
              junctions={junctions}
              selectedJunction={selectedJunction}
              onSelectJunction={setSelectedJunction}
              onRefresh={fetchJunctions}
            />

            <VideoUploader 
              junctionId={selectedJunction}
              onJobCompleted={handleJobCompleted}
            />

            <CountingLineEditor
              junction={selectedJunctionData}
              onSaved={(updated) => setJunctions((current) => current.map((j) => j.junction_id === updated.junction_id ? updated : j))}
            />
          </div>

          <section className="workspace-section">
            <div className="section-heading">
              <div>
                <span className="eyebrow">Vision AI Workspace</span>
                <h2>4-Approach Camera Feeds &amp; Detection Overlay</h2>
              </div>
              <p>Review real-time YOLOv8n object detection, ByteTrack tracking IDs, and directional analytics.</p>
            </div>

            <div className="approaches-grid">
              <ApproachFeedCard approachName="NORTH" state={junctionState?.north} />
              <ApproachFeedCard approachName="SOUTH" state={junctionState?.south} />
              <ApproachFeedCard approachName="EAST" state={junctionState?.east} />
              <ApproachFeedCard approachName="WEST" state={junctionState?.west} />
            </div>
          </section>
        </div>
      )}

      {/* TAB 3: ADAPTIVE SIGNAL SIMULATOR */}
      {activeTab === 'simulation' && (
        <div className="tab-view-container animate-fade-in">
          <SignalSimulator 
            junctionId={selectedJunction} 
            refreshKey={analyticsRefresh} 
          />
        </div>
      )}

      {/* TAB 4: INCIDENT & UPSTREAM DIVERSION HUB */}
      {activeTab === 'incidents' && (
        <div className="tab-view-container animate-fade-in">
          <IncidentManager 
            incidents={incidents}
            onRefresh={fetchIncidents}
            onOpenReportModal={() => setIsReportModalOpen(true)}
            selectedJunction={selectedJunction}
            onSelectJunction={setSelectedJunction}
          />
        </div>
      )}

      {/* TAB 5: HISTORICAL ANALYTICS */}
      {activeTab === 'analytics' && (
        <div className="tab-view-container animate-fade-in">
          <AnalyticsHistory 
            junctionId={selectedJunction} 
            refreshKey={analyticsRefresh} 
          />
        </div>
      )}

      {/* MANDATORY LIVE ON-THE-SPOT CAMERA CAPTURE MODAL */}
      <IncidentReportingModal 
        isOpen={isReportModalOpen}
        onClose={() => setIsReportModalOpen(false)}
        junctions={junctions}
        currentJunction={selectedJunction}
        onIncidentReported={(newInc) => {
          fetchIncidents();
          fetchJunctionState(selectedJunction);
          setAnalyticsRefresh(c => c + 1);
        }}
      />
    </div>
  );
}
