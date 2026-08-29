import React, { useState, useEffect, useRef } from 'react';
import { 
  Upload, 
  Play, 
  CheckCircle2, 
  AlertCircle, 
  Loader2, 
  Video, 
  Radio, 
  Camera, 
  Zap, 
  Trash2, 
  ArrowUp, 
  ArrowDown, 
  ArrowRight, 
  ArrowLeft,
  Link as LinkIcon,
  Wifi,
  Film
} from 'lucide-react';
import { api } from '../services/api';

const APPROACH_CONFIG = [
  { key: 'NORTH', name: 'North', icon: ArrowUp, color: '#0284c7', bg: '#f0f9ff', border: '#bae6fd' },
  { key: 'SOUTH', name: 'South', icon: ArrowDown, color: '#d97706', bg: '#fffbeb', border: '#fde68a' },
  { key: 'EAST', name: 'East', icon: ArrowRight, color: '#16a34a', bg: '#f0fdf4', border: '#bbf7d0' },
  { key: 'WEST', name: 'West', icon: ArrowLeft, color: '#7c3aed', bg: '#faf5ff', border: '#e9d5ff' },
];

export default function VideoUploader({ junctionId, onJobCompleted }) {
  // Mode: 'UPLOAD' | 'LIVE_RTSP' | 'LIVE_CAMERA'
  const [ingestMode, setIngestMode] = useState('UPLOAD');

  // Approach Files & URLs
  const [files, setFiles] = useState({ NORTH: null, SOUTH: null, EAST: null, WEST: null });
  const [streamUrls, setStreamUrls] = useState({
    NORTH: 'rtsp://192.168.1.101:554/live/north',
    SOUTH: 'rtsp://192.168.1.102:554/live/south',
    EAST: 'rtsp://192.168.1.103:554/live/east',
    WEST: 'rtsp://192.168.1.104:554/live/west',
  });

  // Active Job states
  const [jobs, setJobs] = useState({ NORTH: null, SOUTH: null, EAST: null, WEST: null });
  const [globalLoading, setGlobalLoading] = useState(false);
  const [error, setError] = useState(null);
  const [successBanner, setSuccessBanner] = useState(null);

  // Live Webcam state
  const [webcamApproach, setWebcamApproach] = useState('NORTH');
  const [webcamActive, setWebcamActive] = useState(false);
  const videoRef = useRef(null);
  const streamRef = useRef(null);

  // Poll for active jobs
  useEffect(() => {
    const activeJobIds = Object.values(jobs)
      .filter(j => j && (j.status === 'PENDING' || j.status === 'PROCESSING'))
      .map(j => j.job_id);

    if (activeJobIds.length === 0) return;

    const interval = setInterval(async () => {
      try {
        const statuses = await api.getBatchJobStatus(activeJobIds);
        
        setJobs(prev => {
          const next = { ...prev };
          let anyCompleted = false;

          statuses.forEach(st => {
            if (st.approach && next[st.approach]) {
              next[st.approach] = st;
              if (st.status === 'COMPLETED' && prev[st.approach]?.status !== 'COMPLETED') {
                anyCompleted = true;
              }
            }
          });

          if (anyCompleted && onJobCompleted) {
            onJobCompleted();
          }

          return next;
        });
      } catch (err) {
        console.error('Batch status polling error:', err);
      }
    }, 1000);

    return () => clearInterval(interval);
  }, [jobs, onJobCompleted]);

  // Handle single file selection
  const handleFileChange = (approach, file) => {
    setFiles(prev => ({ ...prev, [approach]: file }));
    setError(null);
  };

  // Upload single approach
  const handleSingleUpload = async (approach) => {
    const file = files[approach];
    if (!file) {
      setError(`Select a video file for ${approach} approach first.`);
      return;
    }
    setError(null);

    try {
      const job = await api.uploadVideo(junctionId, approach, file);
      setJobs(prev => ({ ...prev, [approach]: job }));
    } catch (err) {
      setError(err.message || `Failed to process ${approach} video`);
    }
  };

  // Upload all approaches simultaneously
  const handleBatchUploadAll = async () => {
    const selectedCount = Object.values(files).filter(Boolean).length;
    if (selectedCount === 0) {
      setError('Please select at least one approach video file.');
      return;
    }
    setError(null);
    setGlobalLoading(true);

    try {
      const resp = await api.batchUploadVideos(junctionId, files);
      if (resp.jobs && resp.jobs.length > 0) {
        setJobs(prev => {
          const updated = { ...prev };
          resp.jobs.forEach(j => {
            updated[j.approach] = j;
          });
          return updated;
        });
        setSuccessBanner(`Processing started for ${resp.jobs.length} approaches.`);
        setTimeout(() => setSuccessBanner(null), 4000);
      }
    } catch (err) {
      setError(err.message || 'Simultaneous upload failed');
    } finally {
      setGlobalLoading(false);
    }
  };

  // Connect / Save Live Stream URL
  const handleSaveLiveStream = async (approach) => {
    const url = streamUrls[approach];
    try {
      await api.registerLiveStream({
        junction_id: junctionId,
        approach: approach,
        stream_type: 'RTSP',
        stream_url: url,
        is_active: true
      });
      setSuccessBanner(`Live stream connected for ${approach} Approach.`);
      setTimeout(() => setSuccessBanner(null), 3000);
      if (onJobCompleted) onJobCompleted();
    } catch (err) {
      setError(err.message || 'Failed to connect stream');
    }
  };

  // Live Webcam controls
  const startWebcam = async (approach) => {
    setWebcamApproach(approach);
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } },
        audio: false
      });
      streamRef.current = mediaStream;
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
      }
      setWebcamActive(true);
      setError(null);
      await api.registerLiveStream({
        junction_id: junctionId,
        approach: approach,
        stream_type: 'DEVICE_WEBCAM',
        stream_url: 'local://device_webcam',
        is_active: true
      });
    } catch (err) {
      setError('Could not access camera: ' + err.message);
    }
  };

  const stopWebcam = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop());
      streamRef.current = null;
    }
    setWebcamActive(false);
  };

  const activeJobCount = Object.values(jobs).filter(j => j && (j.status === 'PENDING' || j.status === 'PROCESSING')).length;
  const completedJobCount = Object.values(jobs).filter(j => j && j.status === 'COMPLETED').length;

  return (
    <div className="panel-card multi-ingest-panel">
      {/* COMPACT MODULAR HEADER */}
      <div className="ingest-header-row">
        <div className="ingest-title-group">
          <div className="ingest-icon-box">
            <Zap size={16} color="#38bdf8" />
          </div>
          <div>
            <h3>Approach Feeds &amp; Live Ingest</h3>
            <span className="target-sub">Junction: <strong>{junctionId}</strong></span>
          </div>
        </div>

        {/* MINIMAL SEGMENTED TABS */}
        <div className="modular-tab-bar">
          <button 
            type="button"
            className={`modular-tab ${ingestMode === 'UPLOAD' ? 'active' : ''}`}
            onClick={() => setIngestMode('UPLOAD')}
          >
            <Film size={13} /> 4-Way Ingest
          </button>
          <button 
            type="button"
            className={`modular-tab ${ingestMode === 'LIVE_RTSP' ? 'active' : ''}`}
            onClick={() => setIngestMode('LIVE_RTSP')}
          >
            <Wifi size={13} /> RTSP / IP
          </button>
          <button 
            type="button"
            className={`modular-tab ${ingestMode === 'LIVE_CAMERA' ? 'active' : ''}`}
            onClick={() => setIngestMode('LIVE_CAMERA')}
          >
            <Camera size={13} /> Webcam
          </button>
        </div>
      </div>

      {/* MINIMAL NOTICES */}
      {error && (
        <div className="minimal-alert error">
          <AlertCircle size={14} /> {error}
        </div>
      )}
      {successBanner && (
        <div className="minimal-alert success">
          <CheckCircle2 size={14} /> {successBanner}
        </div>
      )}

      {/* VIEW 1: 4-WAY COMPACT FILE INGEST */}
      {ingestMode === 'UPLOAD' && (
        <div className="modular-upload-layout">
          <div className="modular-cards-grid">
            {APPROACH_CONFIG.map(({ key, name, icon: Icon, color, bg, border }) => {
              const file = files[key];
              const job = jobs[key];
              const isProcessing = job && (job.status === 'PENDING' || job.status === 'PROCESSING');
              const isCompleted = job && job.status === 'COMPLETED';

              return (
                <div 
                  key={key} 
                  className="modular-approach-card" 
                  style={{ background: bg, borderColor: border }}
                >
                  <div className="card-top-row">
                    <span className="approach-lbl" style={{ color }}>
                      <Icon size={14} /> {name}
                    </span>
                    {isCompleted ? (
                      <span className="quiet-pill completed">Ready</span>
                    ) : isProcessing ? (
                      <span className="quiet-pill processing">{Math.round(job.progress)}%</span>
                    ) : (
                      <span className="quiet-pill idle">Idle</span>
                    )}
                  </div>

                  <div className="file-slot">
                    {file ? (
                      <div className="compact-file-chip">
                        <span className="chip-name" title={file.name}>{file.name}</span>
                        <button 
                          type="button"
                          className="chip-remove"
                          onClick={() => handleFileChange(key, null)}
                          disabled={isProcessing}
                        >
                          <Trash2 size={13} />
                        </button>
                      </div>
                    ) : (
                      <label className="compact-dropzone">
                        <Upload size={14} style={{ color }} />
                        <span>Select {name} .mp4</span>
                        <input 
                          type="file" 
                          accept="video/mp4,video/avi,video/mov,video/mkv"
                          style={{ display: 'none' }}
                          onChange={(e) => handleFileChange(key, e.target.files[0])}
                        />
                      </label>
                    )}
                  </div>

                  {isProcessing && (
                    <div className="slim-progress-track">
                      <div 
                        className="slim-progress-bar" 
                        style={{ width: `${job.progress}%` }} 
                      />
                    </div>
                  )}

                  <button 
                    type="button"
                    className="btn-card-action"
                    onClick={() => handleSingleUpload(key)}
                    disabled={!file || isProcessing || globalLoading}
                  >
                    {isProcessing ? (
                      <><Loader2 size={13} className="animate-spin" /> Ingesting...</>
                    ) : (
                      <><Play size={12} /> Ingest {name}</>
                    )}
                  </button>
                </div>
              );
            })}
          </div>

          <div className="modular-bottom-bar">
            <div className="bar-meta font-mono">
              Ready: <strong>{Object.values(files).filter(Boolean).length}/4</strong>
              {activeJobCount > 0 && <span className="text-cyan">• Active: {activeJobCount}</span>}
              {completedJobCount > 0 && <span className="text-emerald">• Done: {completedJobCount}</span>}
            </div>

            <button 
              type="button"
              className="btn-primary btn-batch-run"
              onClick={handleBatchUploadAll}
              disabled={globalLoading || Object.values(files).filter(Boolean).length === 0 || activeJobCount > 0}
            >
              {globalLoading ? (
                <>
                  <Loader2 size={14} className="animate-spin" /> Starting Parallel Ingest...
                </>
              ) : (
                <>
                  <Zap size={14} /> Process All Selected Feeds
                </>
              )}
            </button>
          </div>
        </div>
      )}

      {/* VIEW 2: MODULAR RTSP STREAMS */}
      {ingestMode === 'LIVE_RTSP' && (
        <div className="modular-rtsp-layout">
          <div className="modular-cards-grid">
            {APPROACH_CONFIG.map(({ key, name, icon: Icon, color, bg, border }) => (
              <div 
                key={key} 
                className="modular-approach-card"
                style={{ background: bg, borderColor: border }}
              >
                <div className="card-top-row">
                  <span className="approach-lbl" style={{ color }}>
                    <Icon size={14} /> {name} RTSP
                  </span>
                  <span className="clean-stream-pill">
                    <span className="green-dot" /> RTSP
                  </span>
                </div>

                <div className="rtsp-input-slot">
                  <input 
                    type="text" 
                    className="compact-input font-mono"
                    value={streamUrls[key]}
                    onChange={(e) => setStreamUrls(prev => ({ ...prev, [key]: e.target.value }))}
                    placeholder="rtsp://ip:port/live"
                  />
                </div>

                <button 
                  type="button"
                  className="btn-card-action"
                  onClick={() => handleSaveLiveStream(key)}
                >
                  <LinkIcon size={12} /> Connect Stream
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* VIEW 3: MODULAR LIVE WEBCAM */}
      {ingestMode === 'LIVE_CAMERA' && (
        <div className="modular-webcam-layout">
          <div className="webcam-toolbar">
            <div className="cam-select-wrap">
              <span>Target Approach:</span>
              <select 
                className="compact-select"
                value={webcamApproach}
                onChange={(e) => setWebcamApproach(e.target.value)}
                disabled={webcamActive}
              >
                <option value="NORTH">NORTH Approach</option>
                <option value="SOUTH">SOUTH Approach</option>
                <option value="EAST">EAST Approach</option>
                <option value="WEST">WEST Approach</option>
              </select>
            </div>

            {webcamActive ? (
              <button 
                type="button" 
                className="btn-danger-sm"
                onClick={stopWebcam}
              >
                Disconnect Camera
              </button>
            ) : (
              <button 
                type="button" 
                className="btn-primary-sm"
                onClick={() => startWebcam(webcamApproach)}
              >
                <Camera size={13} /> Start Live Camera
              </button>
            )}
          </div>

          <div className="webcam-viewport">
            <video 
              ref={videoRef} 
              autoPlay 
              playsInline 
              muted 
              className="webcam-elem"
              style={{ display: webcamActive ? 'block' : 'none' }}
            />
            {!webcamActive && (
              <div className="webcam-empty">
                <Camera size={28} color="var(--text-secondary)" />
                <span>Webcam inactive. Click above to stream live into {webcamApproach}.</span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
