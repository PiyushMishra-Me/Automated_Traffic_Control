import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import RoleAuthHeader from './components/RoleAuthHeader';
import PortalLandingPage from './components/PortalLandingPage';
import PublicCitizenPortal from './components/PublicCitizenPortal';
import HospitalEmergencyPortal from './components/HospitalEmergencyPortal';
import GovernmentCommandPortal from './components/GovernmentCommandPortal';
import IncidentReportingModal from './components/IncidentReportingModal';
import AuthLoginModal from './components/AuthLoginModal';
import { api } from './services/api';

export default function App() {
  // Role & Gateway State: 'GATEWAY' | 'PUBLIC_USER' | 'HOSPITAL_DISPATCH' | 'GOVERNMENT_OFFICIAL'
  const [currentRole, setCurrentRole] = useState('GATEWAY');
  const [userSession, setUserSession] = useState(null);
  const [isAuthModalOpen, setIsAuthModalOpen] = useState(false);
  const [authTargetRole, setAuthTargetRole] = useState(null);

  // Main Traffic Data & Telemetry
  const [junctions, setJunctions] = useState([]);
  const [selectedJunction, setSelectedJunction] = useState('J-01');
  const [junctionState, setJunctionState] = useState(null);
  const [incidents, setIncidents] = useState([]);
  const [activeAmbulances, setActiveAmbulances] = useState([]);
  const [weatherData, setWeatherData] = useState(null);
  const [isReportModalOpen, setIsReportModalOpen] = useState(false);
  const [analyticsRefresh, setAnalyticsRefresh] = useState(0);

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

  const fetchAmbulances = async () => {
    try {
      const list = await api.listAmbulances();
      setActiveAmbulances(list);
    } catch (err) {
      console.error('Failed to load emergency missions', err);
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
    fetchAmbulances();
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
    fetchAmbulances();
    fetchWeather(selectedJunction);
    setAnalyticsRefresh((c) => c + 1);
  };

  const handleSelectPortal = (roleKey) => {
    if (roleKey === 'GATEWAY') {
      setCurrentRole('GATEWAY');
      return;
    }
    if (roleKey === 'PUBLIC_USER') {
      setCurrentRole('PUBLIC_USER');
      return;
    }
    // Protected roles: check if current session matches the requested role
    if (userSession && userSession.role === roleKey) {
      setCurrentRole(roleKey);
      return;
    }
    // Otherwise, trigger the dummy login authentication modal
    setAuthTargetRole(roleKey);
    setIsAuthModalOpen(true);
  };

  const handleLoginSuccess = (session, role) => {
    setUserSession(session);
    setCurrentRole(role);
    setIsAuthModalOpen(false);
  };

  const handleLogout = () => {
    setUserSession(null);
    setCurrentRole('GATEWAY');
  };

  return (
    <div className="app-container">
      <Navbar 
        activeIncidentCount={activeIncidentCount} 
        onOpenReportModal={() => setIsReportModalOpen(true)} 
      />

      {/* RENDER VIEW 0: FRONT LANDING PORTAL GATEWAY */}
      {currentRole === 'GATEWAY' && (
        <PortalLandingPage 
          onSelectPortal={handleSelectPortal}
        />
      )}

      {/* RENDER VIEWS 1, 2, 3: WITH ROLE BANNER */}
      {currentRole !== 'GATEWAY' && (
        <>
          <RoleAuthHeader 
            currentRole={currentRole}
            onRoleChange={handleSelectPortal}
            userSession={userSession}
            onLogout={handleLogout}
          />

          {/* VIEW 1: PUBLIC CITIZEN PORTAL */}
          {currentRole === 'PUBLIC_USER' && (
            <PublicCitizenPortal 
              junctions={junctions}
              selectedJunction={selectedJunction}
              onSelectJunction={setSelectedJunction}
              incidents={incidents}
              weatherData={weatherData}
              onOpenReportModal={() => setIsReportModalOpen(true)}
              onWeatherUpdated={(updated) => {
                setWeatherData(updated);
                handleRefreshAll();
              }}
            />
          )}

          {/* VIEW 2: EMERGENCY SERVICES PORTAL */}
          {currentRole === 'HOSPITAL_DISPATCH' && (
            <HospitalEmergencyPortal 
              userSession={userSession}
              onMissionUpdated={() => {
                fetchAmbulances();
                fetchJunctionState(selectedJunction);
              }}
            />
          )}

          {/* VIEW 3: GOVERNMENT & POLICE COMMAND CENTER */}
          {currentRole === 'GOVERNMENT_OFFICIAL' && (
            <GovernmentCommandPortal 
              junctions={junctions}
              selectedJunction={selectedJunction}
              onSelectJunction={setSelectedJunction}
              userSession={userSession}
              onOpenReportModal={() => setIsReportModalOpen(true)}
              weatherData={weatherData}
              onWeatherUpdated={(updated) => {
                setWeatherData(updated);
                handleRefreshAll();
              }}
              analyticsRefresh={analyticsRefresh}
              onRefreshAll={handleRefreshAll}
              onJobCompleted={() => {
                fetchJunctionState(selectedJunction);
                setAnalyticsRefresh(c => c + 1);
              }}
            />
          )}
        </>
      )}

      {/* MANDATORY LIVE ON-THE-SPOT CAMERA CAPTURE MODAL */}
      <IncidentReportingModal 
        isOpen={isReportModalOpen}
        onClose={() => setIsReportModalOpen(false)}
        junctions={junctions}
        currentJunction={selectedJunction}
        onIncidentReported={() => {
          fetchIncidents();
          fetchJunctionState(selectedJunction);
          setAnalyticsRefresh(c => c + 1);
        }}
      />

      {/* DUMMY LOGIN AUTHENTICATION POPUP MODAL */}
      <AuthLoginModal 
        isOpen={isAuthModalOpen}
        targetRole={authTargetRole}
        onClose={() => setIsAuthModalOpen(false)}
        onLoginSuccess={handleLoginSuccess}
      />
    </div>
  );
}
