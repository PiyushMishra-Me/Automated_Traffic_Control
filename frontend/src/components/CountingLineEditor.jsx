import React, { useEffect, useState } from 'react';
import { Crosshair, Save } from 'lucide-react';
import { api } from '../services/api';

const DEFAULTS = {
  NORTH: { p1: [0.1, 0.65], p2: [0.9, 0.65], orientation: 'horizontal' },
  SOUTH: { p1: [0.1, 0.35], p2: [0.9, 0.35], orientation: 'horizontal' },
  EAST: { p1: [0.35, 0.1], p2: [0.35, 0.9], orientation: 'vertical' },
  WEST: { p1: [0.65, 0.1], p2: [0.65, 0.9], orientation: 'vertical' },
};

export default function CountingLineEditor({ junction, onSaved }) {
  const [approach, setApproach] = useState('NORTH');
  const [line, setLine] = useState(DEFAULTS.NORTH);
  const [message, setMessage] = useState('');
  useEffect(() => { setLine(junction?.custom_counting_lines?.[approach] || DEFAULTS[approach]); setMessage(''); }, [junction, approach]);

  const updateCoordinate = (point, index, value) => {
    const next = Number(value);
    if (Number.isFinite(next)) setLine((current) => ({ ...current, [point]: current[point].map((item, i) => i === index ? next : item) }));
  };
  const save = async (event) => {
    event.preventDefault();
    if (!junction?.junction_id || [...line.p1, ...line.p2].some((value) => value < 0 || value > 1)) return setMessage('All coordinates must be between 0 and 1.');
    try { const updated = await api.updateCountingLines(junction.junction_id, { [approach]: line }); setMessage('Calibration saved for the next processed video.'); onSaved?.(updated); }
    catch (err) { setMessage(err.message); }
  };
  return <section className="panel-card calibration-card"><div className="panel-header"><h2><Crosshair size={18} color="#38bdf8" /> Camera Calibration</h2></div><form onSubmit={save}><div className="form-group"><label className="form-label">Approach</label><select className="form-select" value={approach} onChange={(e) => setApproach(e.target.value)}>{Object.keys(DEFAULTS).map((name) => <option key={name}>{name}</option>)}</select></div><div className="coordinate-grid">{['p1', 'p2'].flatMap((point) => ['X', 'Y'].map((axis, index) => <label className="form-label" key={`${point}-${axis}`}>{point.toUpperCase()} {axis}<input className="form-input" type="number" min="0" max="1" step="0.01" value={line[point][index]} onChange={(e) => updateCoordinate(point, index, e.target.value)} /></label>))}</div><p className="help-text">Normalized coordinates: 0 is the top/left edge and 1 is the bottom/right edge of the video.</p>{message && <p className={message.startsWith('Calibration') ? 'form-success' : 'form-error'}>{message}</p>}<button className="btn-primary" type="submit"><Save size={15} /> Save counting line</button></form></section>;
}
