import React, { useState } from 'react';
import { MapPin, Plus, Check } from 'lucide-react';
import { api } from '../services/api';

export default function JunctionSelector({ junctions, selectedJunction, onSelectJunction, onRefresh }) {
  const [showModal, setShowModal] = useState(false);
  const [junctionId, setJunctionId] = useState('');
  const [name, setName] = useState('');
  const [location, setLocation] = useState('');
  const [loading, setLoading] = useState(false);

  const handleCreate = async (e) => {
    e.preventDefault();
    if (!junctionId.trim() || !name.trim()) return;

    try {
      setLoading(true);
      await api.createJunction({
        junction_id: junctionId.trim(),
        name: name.trim(),
        location: location.trim() || undefined
      });
      setShowModal(false);
      setJunctionId('');
      setName('');
      setLocation('');
      await onRefresh();
      onSelectJunction(junctionId.trim());
    } catch (err) {
      alert(`Error creating junction: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="panel-card">
      <div className="panel-header">
        <h2>
          <MapPin size={18} color="#38bdf8" /> Active Junction
        </h2>
        <button 
          className="btn-primary" 
          style={{ padding: '0.4rem 0.75rem', fontSize: '0.75rem' }}
          onClick={() => setShowModal(!showModal)}
        >
          <Plus size={14} /> New Junction
        </button>
      </div>

      <div className="form-group">
        <label className="form-label">Select Intersection / Junction:</label>
        <select 
          className="form-select"
          value={selectedJunction} 
          onChange={(e) => onSelectJunction(e.target.value)}
        >
          {junctions.map((j) => (
            <option key={j.junction_id} value={j.junction_id}>
              {j.junction_id} — {j.name} {j.location ? `(${j.location})` : ''}
            </option>
          ))}
        </select>
      </div>

      {showModal && (
        <form onSubmit={handleCreate} style={{ marginTop: '1rem', padding: '1rem', background: 'var(--bg-secondary)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)' }}>
          <h3 style={{ fontSize: '0.85rem', marginBottom: '0.75rem', color: 'var(--accent-cyan)' }}>Create New Intersection</h3>
          <div className="form-group">
            <label className="form-label">Junction ID (e.g. J-02):</label>
            <input 
              className="form-input" 
              placeholder="J-02" 
              value={junctionId} 
              onChange={(e) => setJunctionId(e.target.value)} 
              required 
            />
          </div>
          <div className="form-group">
            <label className="form-label">Intersection Name:</label>
            <input 
              className="form-input" 
              placeholder="MG Road Crossing" 
              value={name} 
              onChange={(e) => setName(e.target.value)} 
              required 
            />
          </div>
          <div className="form-group">
            <label className="form-label">Location (Optional):</label>
            <input 
              className="form-input" 
              placeholder="Sector 14 & Ring Road" 
              value={location} 
              onChange={(e) => setLocation(e.target.value)} 
            />
          </div>
          <div style={{ display: 'flex', gap: '0.5rem', justifyContent: 'flex-end' }}>
            <button 
              type="button" 
              className="btn-primary" 
              style={{ background: 'transparent', border: '1px solid var(--border-color)', color: 'var(--text-secondary)' }}
              onClick={() => setShowModal(false)}
            >
              Cancel
            </button>
            <button type="submit" className="btn-primary" disabled={loading}>
              <Check size={14} /> {loading ? 'Saving...' : 'Add Junction'}
            </button>
          </div>
        </form>
      )}
    </div>
  );
}
