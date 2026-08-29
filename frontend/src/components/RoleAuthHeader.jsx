import React, { useState } from 'react';
import { 
  Shield, 
  User, 
  Building2, 
  Activity, 
  CheckCircle, 
  LogOut, 
  KeyRound,
  Sparkles,
  ArrowLeftRight
} from 'lucide-react';
import { api } from '../services/api';

export default function RoleAuthHeader({ 
  currentRole, 
  onRoleChange, 
  userSession, 
  onLoginSuccess, 
  onLogout 
}) {
  const [isAuthModalOpen, setIsAuthModalOpen] = useState(false);
  const [targetRole, setTargetRole] = useState(null);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleLoginSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const res = await api.loginRole(targetRole, username, password);
      onLoginSuccess(res);
      onRoleChange(targetRole);
      setIsAuthModalOpen(false);
    } catch (err) {
      setError(err.message || 'Authentication failed. Please verify credentials.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <div className="role-auth-banner">
        {/* LEFT: ONLY THE CURRENT ACTIVE PORTAL */}
        <div className="current-portal-box">
          <span className="current-portal-lbl">Current Portal:</span>
          {currentRole === 'PUBLIC_USER' && (
            <div className="active-portal-badge public">
              <User size={14} className="text-cyan" />
              <span>Public Citizen Portal</span>
              <span className="portal-pill-tag">Public</span>
            </div>
          )}

          {currentRole === 'HOSPITAL_DISPATCH' && (
            <div className="active-portal-badge hospital">
              <Activity size={14} className="text-emerald" />
              <span>Hospital Emergency Dispatch</span>
              <span className="portal-pill-tag auth">Corridor Preemption</span>
            </div>
          )}

          {currentRole === 'GOVERNMENT_OFFICIAL' && (
            <div className="active-portal-badge government">
              <Building2 size={14} className="text-blue" />
              <span>Traffic Police &amp; Government Command</span>
              <span className="portal-pill-tag auth">State Control</span>
            </div>
          )}
        </div>

        {/* RIGHT: AUTH STATUS & GATEWAY SWITCHER */}
        <div className="portal-right-controls">
          {userSession && userSession.role !== 'PUBLIC_USER' ? (
            <div className="authenticated-user-pill">
              <Shield size={13} className="text-cyan" />
              <span><strong>{userSession.organization_name}</strong> ({userSession.username})</span>
              <button className="btn-logout" onClick={onLogout} title="Sign Out">
                <LogOut size={12} />
              </button>
            </div>
          ) : null}

          <button 
            type="button"
            className="btn-gateway-switch"
            onClick={() => onRoleChange('GATEWAY')}
            title="Return to Main Portal Gateway"
          >
            <Sparkles size={13} className="text-amber" />
            <span>Portal Gateway</span>
          </button>
        </div>
      </div>

      {/* Role Authentication Modal */}
      {isAuthModalOpen && (
        <div className="modal-backdrop">
          <div className="auth-modal-content">
            <div className="modal-header">
              <div className="title-with-icon">
                <KeyRound size={20} className="text-cyan" />
                <div>
                  <h3>
                    {targetRole === 'HOSPITAL_DISPATCH' 
                      ? '🏥 Hospital Emergency Dispatch Login' 
                      : '🏛️ Traffic Police & Government Command Login'}
                  </h3>
                  <p className="card-subtitle">Confidential portal restricted to authorized personnel</p>
                </div>
              </div>
              <button className="close-btn" onClick={() => setIsAuthModalOpen(false)}>×</button>
            </div>

            {error && <div className="error-banner">{error}</div>}

            <form onSubmit={handleLoginSubmit} className="auth-form">
              <div className="demo-credentials-note">
                <Sparkles size={14} className="text-amber" />
                <span>
                  <strong>Demo Credentials Pre-filled:</strong> {targetRole === 'HOSPITAL_DISPATCH' ? 'hospital_admin / hospital123' : 'traffic_command / police123'}
                </span>
              </div>

              <div className="form-group">
                <label className="form-label">Username / Official ID</label>
                <input 
                  type="text" 
                  className="form-input"
                  value={username} 
                  onChange={(e) => setUsername(e.target.value)} 
                  required 
                />
              </div>

              <div className="form-group">
                <label className="form-label">Security Access Password</label>
                <input 
                  type="password" 
                  className="form-input font-mono"
                  value={password} 
                  onChange={(e) => setPassword(e.target.value)} 
                  required 
                />
              </div>

              <div className="modal-actions">
                <button type="button" className="btn-secondary" onClick={() => setIsAuthModalOpen(false)}>Cancel</button>
                <button type="submit" className="btn-primary" disabled={loading}>
                  {loading ? 'Authenticating...' : 'Unlock Protected Portal'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </>
  );
}
