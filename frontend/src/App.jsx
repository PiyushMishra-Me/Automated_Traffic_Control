import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import JunctionSelector from './components/JunctionSelector';
import VideoUploader from './components/VideoUploader';
import ApproachFeedCard from './components/ApproachFeedCard';
import JunctionOverview from './components/JunctionOverview';
import { api } from './services/api';

export default function App() {
  const [junctions, setJunctions] = useState([]);
  const [selectedJunction, setSelectedJunction] = useState('J-01');
  const [junctionState, setJunctionState] = useState(null);
  const [loading, setLoading] = useState(false);

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

  useEffect(() => {
    fetchJunctions();
  }, []);

  useEffect(() => {
    if (selectedJunction) {
      fetchJunctionState(selectedJunction);
    }
  }, [selectedJunction]);

  const handleJobCompleted = (job) => {
    // Refresh junction state after a video completes processing
    fetchJunctionState(selectedJunction);
  };

  return (
    <div className="app-container">
      <Navbar />

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
      </div>

      <div style={{ marginBottom: '2rem' }}>
        <h2 style={{ fontSize: '1.15rem', fontWeight: 700, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <span>Approach Camera Feeds & Local Analytics</span>
          <span style={{ fontSize: '0.75rem', fontWeight: 500, color: 'var(--text-muted)' }}>
            (Independent Feed Per Road Approach)
          </span>
        </h2>

        <div className="approaches-grid">
          <ApproachFeedCard 
            approachName="NORTH" 
            state={junctionState?.north} 
          />
          <ApproachFeedCard 
            approachName="SOUTH" 
            state={junctionState?.south} 
          />
          <ApproachFeedCard 
            approachName="EAST" 
            state={junctionState?.east} 
          />
          <ApproachFeedCard 
            approachName="WEST" 
            state={junctionState?.west} 
          />
        </div>
      </div>

      <JunctionOverview junctionState={junctionState} />
    </div>
  );
}
