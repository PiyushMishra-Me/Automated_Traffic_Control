const API_BASE = '/api';

export const api = {
  // Junctions
  async listJunctions(city = null) {
    const url = city ? `${API_BASE}/junctions?city=${encodeURIComponent(city)}` : `${API_BASE}/junctions`;
    const res = await fetch(url);
    if (!res.ok) throw new Error('Failed to fetch junctions');
    return res.json();
  },

  async createJunction(data) {
    const res = await fetch(`${API_BASE}/junctions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    if (!res.ok) throw new Error('Failed to create junction');
    return res.json();
  },

  async getJunctionState(junctionId) {
    const res = await fetch(`${API_BASE}/junctions/${junctionId}/state`);
    if (!res.ok) throw new Error('Failed to fetch junction state');
    return res.json();
  },

  async updateCountingLines(junctionId, customCountingLines) {
    const res = await fetch(`${API_BASE}/junctions/${junctionId}/counting-lines`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ custom_counting_lines: customCountingLines }),
    });
    if (!res.ok) throw new Error('Failed to save counting-line calibration');
    return res.json();
  },

  async getSignalRecommendation(junctionId) {
    const res = await fetch(`${API_BASE}/junctions/${junctionId}/signal-recommendation`);
    if (!res.ok) throw new Error('Failed to fetch signal recommendation');
    return res.json();
  },

  async simulateSignal(junctionId, horizonSeconds = 180, forcedRedApproaches = []) {
    const res = await fetch(`${API_BASE}/junctions/${junctionId}/signal-simulation`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        current_phase: 'ALL_RED',
        horizon_seconds: horizonSeconds,
        forced_red_approaches: forcedRedApproaches
      }),
    });
    if (!res.ok) throw new Error('Failed to run signal simulation');
    return res.json();
  },

  async simulateCorridor({ junctionIds = [], links = [], forcedRed = {}, horizonSeconds = 180 }) {
    const res = await fetch(`${API_BASE}/junctions/corridor-simulation`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        junction_ids: junctionIds,
        links: links,
        forced_red: forcedRed,
        horizon_seconds: horizonSeconds,
      }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Failed to run corridor simulation' }));
      throw new Error(err.detail || 'Failed to run corridor simulation');
    }
    return res.json();
  },

  // Manual Police Emergency Signal Light Overrides
  async setManualSignalOverride(junctionId, data) {
    const res = await fetch(`${API_BASE}/junctions/${junctionId}/signal-override`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Failed to set manual signal override' }));
      throw new Error(err.detail || 'Failed to set manual signal override');
    }
    return res.json();
  },

  async getManualSignalOverride(junctionId) {
    const res = await fetch(`${API_BASE}/junctions/${junctionId}/signal-override`);
    if (!res.ok) return { junction_id: junctionId, active: false };
    return res.json();
  },

  async clearManualSignalOverride(junctionId) {
    const res = await fetch(`${API_BASE}/junctions/${junctionId}/signal-override`, {
      method: 'DELETE',
    });
    if (!res.ok) throw new Error('Failed to clear manual signal override');
    return res.json();
  },

  async listActiveSignalOverrides() {
    const res = await fetch(`${API_BASE}/junctions/active-signal-overrides`);
    if (!res.ok) return {};
    return res.json();
  },

  // Videos & Multi-Approach Ingest
  async uploadVideo(junctionId, approach, file) {
    const formData = new FormData();
    formData.append('junction_id', junctionId);
    formData.append('approach', approach);
    formData.append('video', file);

    const res = await fetch(`${API_BASE}/videos/upload`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) throw new Error('Failed to upload video');
    return res.json();
  },

  async batchUploadVideos(junctionId, approachFiles) {
    const formData = new FormData();
    formData.append('junction_id', junctionId);
    if (approachFiles.NORTH) formData.append('north_video', approachFiles.NORTH);
    if (approachFiles.SOUTH) formData.append('south_video', approachFiles.SOUTH);
    if (approachFiles.EAST) formData.append('east_video', approachFiles.EAST);
    if (approachFiles.WEST) formData.append('west_video', approachFiles.WEST);

    const res = await fetch(`${API_BASE}/videos/batch-upload`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) throw new Error('Failed to start simultaneous batch upload');
    return res.json();
  },

  async getBatchJobStatus(jobIds = [], junctionId = null) {
    const params = new URLSearchParams();
    if (jobIds.length > 0) params.set('job_ids', jobIds.join(','));
    if (junctionId) params.set('junction_id', junctionId);
    const res = await fetch(`${API_BASE}/videos/batch-status?${params}`);
    if (!res.ok) throw new Error('Failed to fetch batch job status');
    return res.json();
  },

  async getJobStatus(jobId) {
    const res = await fetch(`${API_BASE}/videos/status/${jobId}`);
    if (!res.ok) throw new Error('Failed to fetch job status');
    return res.json();
  },

  // Live Video & Camera Streams
  async registerLiveStream(data) {
    const res = await fetch(`${API_BASE}/videos/live-stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    if (!res.ok) throw new Error('Failed to register live stream feed');
    return res.json();
  },

  async getJunctionLiveStreams(junctionId) {
    const res = await fetch(`${API_BASE}/videos/live-stream/${junctionId}`);
    if (!res.ok) return {};
    return res.json();
  },

  async deleteLiveStream(junctionId, approach) {
    const res = await fetch(`${API_BASE}/videos/live-stream/${junctionId}/${approach}`, {
      method: 'DELETE',
    });
    if (!res.ok) throw new Error('Failed to delete live stream');
    return res.json();
  },

  async getLatestApproachObservation(junctionId, approach) {
    const res = await fetch(`${API_BASE}/analytics/junction/${junctionId}/approach/${approach}`);
    if (!res.ok) return null;
    return res.json();
  },

  async getAnalyticsHistory(junctionId, approach = null, limit = 50) {
    const params = new URLSearchParams({ limit: String(limit) });
    if (approach) params.set('approach', approach);
    const res = await fetch(`${API_BASE}/analytics/junction/${junctionId}/history?${params}`);
    if (!res.ok) throw new Error('Failed to fetch analytics history');
    return res.json();
  },

  async getAnalyticsSummary(junctionId, approach = null) {
    const params = new URLSearchParams();
    if (approach) params.set('approach', approach);
    const suffix = params.toString() ? `?${params}` : '';
    const res = await fetch(`${API_BASE}/analytics/junction/${junctionId}/summary${suffix}`);
    if (!res.ok) throw new Error('Failed to fetch analytics summary');
    return res.json();
  },

  // Incidents & Diversions
  async listIncidents(junctionId = null, status = null) {
    const params = new URLSearchParams();
    if (junctionId) params.set('junction_id', junctionId);
    if (status) params.set('status', status);
    const suffix = params.toString() ? `?${params}` : '';
    const res = await fetch(`${API_BASE}/incidents${suffix}`);
    if (!res.ok) throw new Error('Failed to fetch incidents');
    return res.json();
  },

  async reportIncident(data) {
    const res = await fetch(`${API_BASE}/incidents`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    if (!res.ok) throw new Error('Failed to submit incident report');
    return res.json();
  },

  async updateIncidentStatus(incidentId, status) {
    const res = await fetch(`${API_BASE}/incidents/${incidentId}/status`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ status }),
    });
    if (!res.ok) throw new Error('Failed to update incident status');
    return res.json();
  },

  async getActiveDiversions(junctionId) {
    const res = await fetch(`${API_BASE}/incidents/junction/${junctionId}/active-diversions`);
    if (!res.ok) throw new Error('Failed to fetch active diversions');
    return res.json();
  },

  // Weather
  async getJunctionWeather(junctionId) {
    const res = await fetch(`${API_BASE}/weather/junction/${junctionId}`);
    if (!res.ok) throw new Error('Failed to fetch weather telemetry');
    return res.json();
  },

  async overrideJunctionWeather(junctionId, data) {
    const res = await fetch(`${API_BASE}/weather/junction/${junctionId}/override`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    if (!res.ok) throw new Error('Failed to override weather condition');
    return res.json();
  },

  async clearWeatherOverride(junctionId) {
    const res = await fetch(`${API_BASE}/weather/junction/${junctionId}/override`, {
      method: 'DELETE',
    });
    if (!res.ok) throw new Error('Failed to clear weather override');
    return res.json();
  },

  // Auth & Roles
  async getAuthProfiles() {
    const res = await fetch(`${API_BASE}/auth/profiles`);
    if (!res.ok) throw new Error('Failed to fetch auth profiles');
    return res.json();
  },

  async loginRole(role, username, password, organizationName = null) {
    const res = await fetch(`${API_BASE}/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        role,
        username,
        password,
        organization_name: organizationName,
      }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Authentication failed' }));
      throw new Error(err.detail || 'Authentication failed');
    }
    return res.json();
  },

  // Ambulance & Emergency Green Wave
  async registerAmbulance(data) {
    const res = await fetch(`${API_BASE}/ambulances/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Failed to register ambulance mission' }));
      throw new Error(err.detail || 'Failed to register ambulance mission');
    }
    return res.json();
  },

  async listAmbulances(status = null, hospitalName = null) {
    const params = new URLSearchParams();
    if (status) params.set('status', status);
    if (hospitalName) params.set('hospital_name', hospitalName);
    const suffix = params.toString() ? `?${params}` : '';
    const res = await fetch(`${API_BASE}/ambulances${suffix}`);
    if (!res.ok) throw new Error('Failed to fetch ambulance missions');
    return res.json();
  },

  async getAmbulanceMission(missionId) {
    const res = await fetch(`${API_BASE}/ambulances/${missionId}`);
    if (!res.ok) throw new Error('Failed to fetch ambulance mission');
    return res.json();
  },

  async updateAmbulanceStatus(missionId, newStatus) {
    const res = await fetch(`${API_BASE}/ambulances/${missionId}/status?new_status=${newStatus}`, {
      method: 'PATCH',
    });
    if (!res.ok) throw new Error('Failed to update ambulance status');
    return res.json();
  },

  async getJunctionPreemption(junctionId) {
    const res = await fetch(`${API_BASE}/ambulances/junction/${junctionId}/preemption`);
    if (!res.ok) throw new Error('Failed to fetch junction preemption status');
    return res.json();
  },

  // Public Navigation & Optimal Route Pathfinder
  async calculateOptimalRoute(payload) {
    const res = await fetch(`${API_BASE}/navigation/route`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Failed to calculate optimal route' }));
      throw new Error(err.detail || 'Failed to calculate optimal route');
    }
    return res.json();
  },

  async getCorridorStatuses() {
    const res = await fetch(`${API_BASE}/navigation/corridors`);
    if (!res.ok) throw new Error('Failed to fetch corridor statuses');
    return res.json();
  },
};
