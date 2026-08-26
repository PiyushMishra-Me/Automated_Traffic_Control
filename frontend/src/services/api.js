const API_BASE = '/api';

export const api = {
  // Junctions
  async listJunctions() {
    const res = await fetch(`${API_BASE}/junctions`);
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

  // Videos
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

  async getJobStatus(jobId) {
    const res = await fetch(`${API_BASE}/videos/status/${jobId}`);
    if (!res.ok) throw new Error('Failed to fetch job status');
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
};
