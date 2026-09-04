import React from 'react';
import { 
  Shield, 
  User, 
  Building2, 
  Activity, 
  CheckCircle, 
  LogOut, 
  Sparkles,
  Lock,
  Unlock
} from 'lucide-react';

export default function RoleAuthHeader({ 
  currentRole, 
  onRoleChange, 
  userSession, 
  onLogout 
}) {
  return (
    <div className="role-auth-banner">
      {/* PORTAL PROFILE SELECTOR TABS */}
      <div className="role-selector-group">
        <span className="role-select-label">Active Portal:</span>

        {/* 1. PUBLIC CITIZEN */}
        <button 
          type="button"
          className={`role-tab-btn ${currentRole === 'PUBLIC_USER' ? 'active public' : ''}`}
          onClick={() => onRoleChange('PUBLIC_USER')}
          title="Switch to Public Citizen Portal"
        >
          <User size={14} />
          <span>Public Citizen</span>
          <span className="role-badge open">
            <Unlock size={9} /> Open
          </span>
        </button>

        {/* 2. HOSPITAL EMERGENCY */}
        <button 
          type="button"
          className={`role-tab-btn ${currentRole === 'HOSPITAL_DISPATCH' ? 'active hospital' : ''}`}
          onClick={() => onRoleChange('HOSPITAL_DISPATCH')}
          title="Switch to Hospital Emergency Dispatch"
        >
          <Activity size={14} className="text-emerald" />
          <span>Emergency Dispatch</span>
          {userSession && userSession.role === 'HOSPITAL_DISPATCH' ? (
            <span className="role-badge auth">
              <CheckCircle size={10} /> Authenticated
            </span>
          ) : (
            <span className="role-badge locked">
              <Lock size={10} /> Protected
            </span>
          )}
        </button>

        {/* 3. GOVERNMENT / TRAFFIC POLICE */}
        <button 
          type="button"
          className={`role-tab-btn ${currentRole === 'GOVERNMENT_OFFICIAL' ? 'active gov' : ''}`}
          onClick={() => onRoleChange('GOVERNMENT_OFFICIAL')}
          title="Switch to Traffic Police & Government Command"
        >
          <Building2 size={14} className="text-blue" />
          <span>Traffic Police Command</span>
          {userSession && userSession.role === 'GOVERNMENT_OFFICIAL' ? (
            <span className="role-badge auth">
              <CheckCircle size={10} /> Authenticated
            </span>
          ) : (
            <span className="role-badge locked">
              <Lock size={10} /> Protected
            </span>
          )}
        </button>
      </div>

      {/* RIGHT: AUTH STATUS & GATEWAY SWITCHER */}
      <div className="portal-right-controls">
        {userSession && userSession.role !== 'PUBLIC_USER' ? (
          <div className="authenticated-user-pill">
            <Shield size={13} className="text-cyan" />
            <span><strong>{userSession.organization_name}</strong> ({userSession.username})</span>
            <button 
              type="button" 
              className="btn-logout" 
              onClick={onLogout} 
              title="Sign Out of Protected Portal"
            >
              <LogOut size={12} />
            </button>
          </div>
        ) : null}

        <button 
          type="button"
          className="btn-gateway-switch"
          onClick={() => onRoleChange('GATEWAY')}
          title="Return to Main Portal Gateway Landing Page"
        >
          <Sparkles size={13} className="text-amber" />
          <span>Portal Gateway</span>
        </button>
      </div>
    </div>
  );
}
