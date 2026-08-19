import React, { useState, useEffect } from 'react';
import { Upload, Play, CheckCircle2, AlertCircle, Loader2 } from 'lucide-react';
import { api } from '../services/api';

export default function VideoUploader({ junctionId, onJobCompleted }) {
  const [approach, setApproach] = useState('NORTH');
  const [file, setFile] = useState(null);
  const [activeJob, setActiveJob] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Poll for job status when active
  useEffect(() => {
    if (!activeJob || activeJob.status === 'COMPLETED' || activeJob.status === 'FAILED') {
      return;
    }

    const interval = setInterval(async () => {
      try {
        const updated = await api.getJobStatus(activeJob.job_id);
        setActiveJob(updated);

        if (updated.status === 'COMPLETED') {
          clearInterval(interval);
          if (onJobCompleted) {
            onJobCompleted(updated);
          }
        } else if (updated.status === 'FAILED') {
          clearInterval(interval);
          setError(updated.message || 'Processing failed');
        }
      } catch (e) {
        console.error('Status poll error', e);
      }
    }, 1200);

    return () => clearInterval(interval);
  }, [activeJob, onJobCompleted]);

  const handleUpload = async (e) => {
    e.preventDefault();
    if (!file) {
      setError('Please select a traffic video file.');
      return;
    }
    setError(null);
    setLoading(true);

    try {
      const job = await api.uploadVideo(junctionId, approach, file);
      setActiveJob(job);
      setFile(null);
    } catch (err) {
      setError(err.message || 'Failed to start processing');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="panel-card">
      <div className="panel-header">
        <h2>
          <Upload size={18} color="#38bdf8" /> Approach Camera Video Ingest
        </h2>
        <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
          Target: {junctionId}
        </span>
      </div>

      <form onSubmit={handleUpload}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.5fr', gap: '1rem' }}>
          <div className="form-group">
            <label className="form-label">Road Approach / Camera:</label>
            <select 
              className="form-select"
              value={approach}
              onChange={(e) => setApproach(e.target.value)}
            >
              <option value="NORTH">NORTH Approach</option>
              <option value="SOUTH">SOUTH Approach</option>
              <option value="EAST">EAST Approach</option>
              <option value="WEST">WEST Approach</option>
            </select>
          </div>

          <div className="form-group">
            <label className="form-label">Traffic Video File (.mp4):</label>
            <input 
              type="file" 
              accept="video/mp4,video/avi,video/mov,video/mkv"
              className="form-input"
              onChange={(e) => setFile(e.target.files[0])}
            />
          </div>
        </div>

        {error && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--level-very-high)', fontSize: '0.8rem', marginBottom: '0.75rem' }}>
            <AlertCircle size={15} /> {error}
          </div>
        )}

        <button 
          type="submit" 
          className="btn-primary" 
          style={{ width: '100%', marginTop: '0.25rem' }}
          disabled={loading || (activeJob && (activeJob.status === 'PENDING' || activeJob.status === 'PROCESSING'))}
        >
          {loading ? (
            <>
              <Loader2 size={16} className="animate-spin" /> Uploading Video...
            </>
          ) : (
            <>
              <Play size={16} /> Process Approach Video (YOLOv8n + ByteTrack)
            </>
          )}
        </button>
      </form>

      {activeJob && (
        <div className="progress-container">
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem' }}>
            <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
              Approach: {activeJob.approach} ({activeJob.status})
            </span>
            <span style={{ color: 'var(--accent-cyan)' }}>{Math.round(activeJob.progress)}%</span>
          </div>

          <div className="progress-bar-bg">
            <div 
              className="progress-bar-fill" 
              style={{ width: `${activeJob.progress}%` }} 
            />
          </div>

          <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
            {activeJob.message || 'Running vehicle detection & trajectory tracking...'}
          </p>

          {activeJob.status === 'COMPLETED' && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: 'var(--level-low)', fontSize: '0.8rem', marginTop: '0.4rem' }}>
              <CheckCircle2 size={16} /> Inference complete & observation stored in MongoDB.
            </div>
          )}
        </div>
      )}
    </div>
  );
}
