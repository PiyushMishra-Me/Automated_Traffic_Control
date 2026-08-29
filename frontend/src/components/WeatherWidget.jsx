import React, { useState } from 'react';
import { 
  CloudRain, 
  Sun, 
  Cloud, 
  CloudLightning, 
  CloudFog, 
  Snowflake, 
  Wind, 
  Eye, 
  Droplets, 
  Gauge, 
  ShieldCheck, 
  AlertOctagon,
  RefreshCw,
  Sliders
} from 'lucide-react';
import { api } from '../services/api';

export default function WeatherWidget({ 
  junctionId = 'J-01', 
  weatherData = null, 
  onWeatherUpdated 
}) {
  const [isSimulating, setIsSimulating] = useState(false);
  const [selectedScenario, setSelectedScenario] = useState('HEAVY_RAIN');
  const [loading, setLoading] = useState(false);

  const getWeatherIcon = (condition) => {
    switch (condition) {
      case 'HEAVY_RAIN':
      case 'LIGHT_RAIN':
        return <CloudRain className="weather-icon text-cyan" size={32} />;
      case 'THUNDERSTORM':
        return <CloudLightning className="weather-icon text-amber" size={32} />;
      case 'FOG':
        return <CloudFog className="weather-icon text-slate" size={32} />;
      case 'SNOW':
        return <Snowflake className="weather-icon text-cyan" size={32} />;
      case 'PARTLY_CLOUDY':
      case 'CLOUDY':
        return <Cloud className="weather-icon text-muted" size={32} />;
      default:
        return <Sun className="weather-icon text-yellow" size={32} />;
    }
  };

  const handleApplyScenario = async (scenario) => {
    setLoading(true);
    try {
      let payload = {};
      if (scenario === 'HEAVY_RAIN') {
        payload = { condition: 'HEAVY_RAIN', precipitation_mm: 14.5, visibility_km: 2.2, road_surface: 'WET' };
      } else if (scenario === 'THUNDERSTORM') {
        payload = { condition: 'THUNDERSTORM', precipitation_mm: 26.0, visibility_km: 1.1, road_surface: 'FLOODED' };
      } else if (scenario === 'FOG') {
        payload = { condition: 'FOG', precipitation_mm: 0.0, visibility_km: 0.6, road_surface: 'SLIPPERY' };
      } else if (scenario === 'CLEAR') {
        payload = { condition: 'CLEAR', precipitation_mm: 0.0, visibility_km: 10.0, road_surface: 'DRY' };
      }

      const updated = await api.overrideJunctionWeather(junctionId, payload);
      if (onWeatherUpdated) onWeatherUpdated(updated);
    } catch (err) {
      console.error('Failed to apply weather scenario', err);
    } finally {
      setLoading(false);
    }
  };

  const handleResetLive = async () => {
    setLoading(true);
    try {
      const live = await api.clearWeatherOverride(junctionId);
      if (onWeatherUpdated) onWeatherUpdated(live);
    } catch (err) {
      console.error('Failed to reset live weather', err);
    } finally {
      setLoading(false);
    }
  };

  if (!weatherData) {
    return (
      <div className="weather-card loading-state">
        <CloudRain className="spinning-icon" size={24} />
        <span>Loading Junction Meteorological Telemetry...</span>
      </div>
    );
  }

  const adj = weatherData.adjustments || {};
  const hasWeatherAdjustment = (adj.extra_yellow_seconds > 0 || adj.extra_all_red_seconds > 0);

  return (
    <div className="weather-card">
      <div className="weather-card-top">
        <div className="weather-main-info">
          {getWeatherIcon(weatherData.condition)}
          <div>
            <div className="temp-display">
              <span className="temp-value">{weatherData.temperature_c}°C</span>
              <span className="condition-badge">{weatherData.condition.replace('_', ' ')}</span>
              {!weatherData.is_live && <span className="sim-chip">SIMULATED SCENARIO</span>}
            </div>
            <div className="road-surface-tag">
              Road Surface: <strong>{weatherData.road_surface}</strong> • Friction Factor: {weatherData.braking_distance_factor}x
            </div>
          </div>
        </div>

        <div className="weather-quick-stats">
          <div className="stat-pill">
            <Droplets size={14} /> <span>Rain: <strong>{weatherData.precipitation_mm} mm/h</strong></span>
          </div>
          <div className="stat-pill">
            <Wind size={14} /> <span>Wind: <strong>{weatherData.wind_speed_kmh} km/h</strong></span>
          </div>
          <div className="stat-pill">
            <Eye size={14} /> <span>Visibility: <strong>{weatherData.visibility_km} km</strong></span>
          </div>
        </div>
      </div>

      {/* Safety & Adaptive Clearance Banner */}
      <div className={`weather-safety-banner ${hasWeatherAdjustment ? 'warning-mode' : 'normal-mode'}`}>
        <div className="safety-header">
          {hasWeatherAdjustment ? (
            <AlertOctagon size={18} className="text-amber" />
          ) : (
            <ShieldCheck size={18} className="text-green" />
          )}
          <span className="safety-title">
            {hasWeatherAdjustment ? 'Weather-Adaptive Signal Timing Active' : 'Optimal Environmental Traction'}
          </span>
          <span className="speed-limit-chip font-mono">
            Speed Advisory: <strong>{adj.speed_advisory_kmh || 50} km/h</strong>
          </span>
        </div>
        <p className="safety-description">{adj.safety_advisory}</p>

        {hasWeatherAdjustment && (
          <div className="safety-timing-chips">
            <span className="chip">+ {adj.extra_yellow_seconds}s Yellow Clearance</span>
            <span className="chip">+ {adj.extra_all_red_seconds}s All-Red Safety Interval</span>
            <span className="chip">-{adj.saturation_flow_reduction_pct}% Saturation Flow</span>
          </div>
        )}
      </div>

      {/* Interactive Scenario Controls */}
      <div className="weather-simulator-drawer">
        <div className="drawer-header">
          <Sliders size={14} />
          <span>Simulate Weather Scenarios for {junctionId}:</span>
        </div>
        <div className="scenario-btn-group">
          <button 
            className="scenario-btn" 
            onClick={() => handleApplyScenario('HEAVY_RAIN')}
            disabled={loading}
          >
            🌧️ Heavy Rain
          </button>
          <button 
            className="scenario-btn" 
            onClick={() => handleApplyScenario('THUNDERSTORM')}
            disabled={loading}
          >
            ⛈️ Storm / Flood
          </button>
          <button 
            className="scenario-btn" 
            onClick={() => handleApplyScenario('FOG')}
            disabled={loading}
          >
            🌫️ Dense Fog
          </button>
          <button 
            className="scenario-btn" 
            onClick={() => handleApplyScenario('CLEAR')}
            disabled={loading}
          >
            ☀️ Clear Skies
          </button>
          {!weatherData.is_live && (
            <button 
              className="scenario-btn btn-reset-live" 
              onClick={handleResetLive}
              disabled={loading}
            >
              <RefreshCw size={12} /> Fetch Live Open-Meteo
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
