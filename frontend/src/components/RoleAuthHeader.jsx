import React, { useState } from 'react';
import { 
  Shield, 
  Lock, 
  Unlock, 
  User, 
  Building2, 
  Cross, 
  Activity, 
  CheckCircle, 
  AlertCircle, 
  LogOut, 
  KeyRound,
  Sparkles
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

  const handleRoleSelect = (roleKey) => {
    if (roleKey === 'PUBLIC_USER') {
      onRoleChange('PUBLIC_USER');
      return;
    }
    // If switching to Hospital or Government, check if already authenticated
    if (userSession && userSession.role === roleKey) {
      onRoleChange(roleKey);
      return;
    }

    // Open Auth Modal
    setTargetRole(roleKey);
    if (roleKey === 'HOSPITAL_DISPATCH') {
      setUsername('hospital_admin');
      setPassword('hospital123');
    } else {
      setUsername('traffic_command');
      setPassword('police123');
    }
    setError(null);
    setIsAuthModalOpen(true);
  };

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
        <div className="role-selector-group">
          <span className="role-select-label">Select Active Portal Profile:</span>

          <button 
            className={`role-tab-btn ${currentRole === 'PUBLIC_USER' ? 'active public' : ''}`}
            onClick={() => handleRoleSelect('PUBLIC_USER')}
          >
            <User size={15} />
            <span>Public Citizen</span>
            <span className="role-badge open">Public</span>
          </button>

          <button 
            className={`role-tab-btn ${currentRole === 'HOSPITAL_DISPATCH' ? 'active hospital' : ''}`}
            onClick={() => handleRoleSelect('HOSPITAL_DISPATCH')}
          >
            <Activity size={15} className="text-emerald" />
            <span>Hospital Emergency Dispatch</span>
            {userSession && userSession.role === 'HOSPITAL_DISPATCH' ? (
              <span className="role-badge auth"><CheckCircle size={10} /> Authenticated</span>
            ) : (
              <span className="role-badge locked"><Lock size={10} /> Protected</span>
            )}
          </button>

          <button 
            className={`role-tab-btn ${currentRole === 'GOVERNMENT_OFFICIAL' ? 'active gov' : ''}`}
            onClick={() => handleRoleSelect('GOVERNMENT_OFFICIAL')}
          >
            <Building2 size={15} className="text-blue" />
            <span>Traffic Police &amp; Government Command</span>
            {userSession && userSession.role === 'GOVERNMENT_OFFICIAL' ? (
              <span className="role-badge auth"><CheckCircle size={10} /> Authenticated</span>
            ) : (
              <span className="role-badge locked"><Lock size={10} /> Protected</span>
            )}
          </button>

          <button 
            className="role-tab-btn btn-gateway-switch"
            onClick={() => onRoleChange('GATEWAY')}
            title="Return to Main Portal Gateway"
          >
            <Sparkles size={14} className="text-amber" />
            <span>Portal Gateway</span>
          </button>
        </div>

        <div className="auth-status-chip">
          {userSession && userSession.role !== 'PUBLIC_USER' ? (
            <div className="authenticated-user-pill">
              <Shield size={14} className="text-cyan" />
              <span><strong>{userSession.organization_name}</strong> ({userSession.username})</span>
              <button className="btn-logout" onClick={onLogout} title="Sign Out">
                <LogOut size={13} />
              </button>
            </div>
          ) : (
            <div className="public-user-pill">
              <User size={14} />
              <span>Public Citizen Mode (Accident Reporting &amp; Live Map)</span>
            </div>
          )}
        </div>
      </div>

      {/* Role Authentication Modal */}
      {isAuthModalOpen && (
        <div className="modal-backdrop">
          <div className="auth-modal-content">
            <div className="modal-header">
              <div className="title-with-icon">
                <KeyRound size={22} className="text-cyan" />
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
                <label>Username / Official ID</label>
                <input 
                  type="text" 
                  value={username} 
                  onChange={(e) => setUsername(e.target.value)} 
                  required 
                />
              </div>

              <div className="form-group">
                <label>Security Access Password</label>
                <input 
                  type="password" 
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
