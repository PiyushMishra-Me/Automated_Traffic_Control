import React, { useState, useEffect, useRef } from 'react';
import { 
  MapPin, 
  AlertTriangle, 
  CloudRain, 
  Navigation, 
  Layers, 
  ZoomIn, 
  ZoomOut, 
  RotateCcw, 
  ShieldAlert, 
  Compass, 
  ArrowRight, 
  Activity, 
  Radio, 
  Building2, 
  Sparkles, 
  Navigation2,
  CheckCircle2,
  Maximize2
} from 'lucide-react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

// Metropolitan City Center Presets
export const CITY_PRESETS = {
  DELHI: {
    name: 'New Delhi / NCR',
    center: [28.6250, 77.2100],
    zoom: 12,
    icon: '🏛️',
    description: 'Connaught Place, ITO, AIIMS & Airport Corridors'
  },
  MUMBAI: {
    name: 'Mumbai Metropolitan',
    center: [19.0400, 72.8700],
    zoom: 12,
    icon: '🌊',
    description: 'BKC, Dadar, Marine Drive, Andheri & Vashi Link'
  },
  HYDERABAD: {
    name: 'Hyderabad Cyber Hub',
    center: [17.4200, 78.4100],
    zoom: 12,
    icon: '💎',
    description: 'Hitec City, Gachibowli, Jubilee Hills & Charminar'
  },
  BENGALURU: {
    name: 'Bengaluru Tech Grid',
    center: [12.9350, 77.6300],
    zoom: 12,
    icon: '🌳',
    description: 'Silk Board, E-City, Koramangala, Indiranagar & MG Road'
  }
};

export default function LiveJunctionMap({ 
  junctions = [], 
  selectedJunction = 'J-01', 
  onSelectJunction, 
  incidents = [],
  activeAmbulances = [],
  weatherData = null,
  navigationRoute = null,
  onOpenReportModal,
  selectedCity = 'DELHI',
  onSelectCity,
  onSetNavOrigin,
  onSetNavDestination
}) {
  const [mapMode, setMapMode] = useState('LEAFLET'); // 'LEAFLET' | 'TACTICAL'
  const [currentCity, setCurrentCity] = useState(selectedCity || 'DELHI');
  const [activeLayers, setActiveLayers] = useState({
    traffic: true,
    corridors: true,
    diversions: true,
    ambulances: true,
    weather: true
  });

  const mapContainerRef = useRef(null);
  const leafletMapRef = useRef(null);
  const markersGroupRef = useRef(null);
  const corridorsGroupRef = useRef(null);
  const routeGroupRef = useRef(null);

  // Sync city prop if changed from parent
  useEffect(() => {
    if (selectedCity && selectedCity !== currentCity) {
      setCurrentCity(selectedCity);
    }
  }, [selectedCity]);

  // Initialize Leaflet Map
  useEffect(() => {
    if (mapMode !== 'LEAFLET' || !mapContainerRef.current) return;

    if (!leafletMapRef.current) {
      const cityCfg = CITY_PRESETS[currentCity] || CITY_PRESETS.DELHI;
      const map = L.map(mapContainerRef.current, {
        center: cityCfg.center,
        zoom: cityCfg.zoom,
        zoomControl: false,
        attributionControl: false
      });

      // Add CartoDB Voyager tile layer (clean, high-contrast, modern street map)
      L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png', {
        maxZoom: 19,
        subdomains: 'abcd',
      }).addTo(map);

      // Create Layer Groups
      corridorsGroupRef.current = L.layerGroup().addTo(map);
      routeGroupRef.current = L.layerGroup().addTo(map);
      markersGroupRef.current = L.layerGroup().addTo(map);

      leafletMapRef.current = map;
    }

    return () => {
      // Map cleanup handled gracefully
    };
  }, [mapMode]);

  // Handle City Change: Pan/FlyTo City Center
  const handleCityChange = (cityKey) => {
    setCurrentCity(cityKey);
    if (onSelectCity) onSelectCity(cityKey);

    const cityCfg = CITY_PRESETS[cityKey];
    if (leafletMapRef.current && cityCfg) {
      leafletMapRef.current.flyTo(cityCfg.center, cityCfg.zoom, {
        duration: 1.5,
        easeLinearity: 0.25
      });
    }

    // If selected junction not in this city, select first junction of this city
    const cityJunctions = junctions.filter(j => (j.city || 'DELHI').toUpperCase() === cityKey.toUpperCase());
    if (cityJunctions.length > 0 && !cityJunctions.some(j => j.junction_id === selectedJunction)) {
      if (onSelectJunction) onSelectJunction(cityJunctions[0].junction_id);
    }
  };

  // Render Markers, Corridors & Navigation Path on Leaflet Map
  useEffect(() => {
    if (mapMode !== 'LEAFLET' || !leafletMapRef.current) return;

    const map = leafletMapRef.current;
    if (markersGroupRef.current) markersGroupRef.current.clearLayers();
    if (corridorsGroupRef.current) corridorsGroupRef.current.clearLayers();
    if (routeGroupRef.current) routeGroupRef.current.clearLayers();

    const jMap = new Map();
    junctions.forEach(j => jMap.set(j.junction_id, j));

    // 1. Draw Corridors between connected junctions
    if (activeLayers.corridors && corridorsGroupRef.current) {
      const drawnPairs = new Set();
      junctions.forEach(j => {
        const lat1 = j.latitude || 28.6139;
        const lng1 = j.longitude || 77.2090;

        (j.connected_junctions || []).forEach(neighborId => {
          const neighbor = jMap.get(neighborId);
          if (neighbor) {
            const pairKey = [j.junction_id, neighborId].sort().join('-');
            if (!drawnPairs.has(pairKey)) {
              drawnPairs.add(pairKey);
              const lat2 = neighbor.latitude || 28.6139;
              const lng2 = neighbor.longitude || 77.2090;

              const corridorLine = L.polyline([[lat1, lng1], [lat2, lng2]], {
                color: '#64748b',
                weight: 3.5,
                opacity: 0.6,
                dashArray: '6, 6'
              });
              corridorLine.bindTooltip(`${j.name.split(' ')[0]} ↔ ${neighbor.name.split(' ')[0]}`, { sticky: true });
              corridorsGroupRef.current.addLayer(corridorLine);
            }
          }
        });
      });
    }

    // 2. Draw Active Public Navigation Shortest Path Route
    if (navigationRoute && routeGroupRef.current) {
      let jIds = [];
      if (Array.isArray(navigationRoute)) {
        jIds = navigationRoute;
      } else if (Array.isArray(navigationRoute.optimal_route_junctions)) {
        jIds = navigationRoute.optimal_route_junctions;
      } else if (Array.isArray(navigationRoute.steps)) {
        const collected = [];
        navigationRoute.steps.forEach(step => {
          if (step.from_junction_id && !collected.includes(step.from_junction_id)) collected.push(step.from_junction_id);
          if (step.to_junction_id && !collected.includes(step.to_junction_id)) collected.push(step.to_junction_id);
        });
        jIds = collected;
      }

      const routeCoords = [];
      jIds.forEach(id => {
        const jNode = jMap.get(id);
        if (jNode && jNode.latitude && jNode.longitude) {
          routeCoords.push([jNode.latitude, jNode.longitude]);
        }
      });

      if (routeCoords.length >= 2) {
        // Glowing background line
        const glowLine = L.polyline(routeCoords, {
          color: '#38bdf8',
          weight: 10,
          opacity: 0.5
        });
        routeGroupRef.current.addLayer(glowLine);

        // Active solid route line
        const activeRouteLine = L.polyline(routeCoords, {
          color: '#0284c7',
          weight: 5,
          opacity: 0.95
        });
        const transitMin = navigationRoute.estimated_travel_time_formatted || 
          (navigationRoute.total_travel_time_seconds ? `${Math.round(navigationRoute.total_travel_time_seconds / 60)} mins` : 'Optimal');
        const distKm = navigationRoute.total_distance_km ? `${navigationRoute.total_distance_km} km` : '';

        activeRouteLine.bindPopup(`
          <div style="font-family: system-ui; padding: 4px;">
            <div style="font-weight: 700; color: #0284c7; margin-bottom: 4px;">⚡ Optimal Navigation Path</div>
            <div>Estimated Transit: <b>${transitMin}</b></div>
            ${distKm ? `<div>Distance: <b>${distKm}</b></div>` : ''}
            <div style="color: #10b981; font-size: 11px; margin-top: 4px;">✓ Real-time traffic & weather optimized</div>
          </div>
        `);
        routeGroupRef.current.addLayer(activeRouteLine);

        // Auto-fit map to route
        map.fitBounds(activeRouteLine.getBounds(), { padding: [40, 40], maxZoom: 15 });
      }
    }

    // 3. Draw Junction Markers
    if (markersGroupRef.current) {
      junctions.forEach(j => {
        const lat = j.latitude || 28.6139;
        const lng = j.longitude || 77.2090;
        const isSelected = j.junction_id === selectedJunction;
        const hasIncident = incidents.some(inc => inc.junction_id === j.junction_id && inc.status === 'ACTIVE');
        const hasAmbulance = activeAmbulances.some(amb => 
          amb.current_junction_id === j.junction_id || 
          (amb.route_corridor && amb.route_corridor.some(n => n.junction_id === j.junction_id))
        );

        // Custom HTML Marker Pin
        const markerHtml = `
          <div class="custom-leaflet-marker ${isSelected ? 'marker-selected' : ''}" style="
            position: relative;
            width: ${isSelected ? '44px' : '36px'};
            height: ${isSelected ? '44px' : '36px'};
            border-radius: 50%;
            background: ${hasIncident ? '#ef4444' : hasAmbulance ? '#f59e0b' : isSelected ? '#0284c7' : '#ffffff'};
            border: 3px solid ${isSelected ? '#ffffff' : hasIncident ? '#f87171' : hasAmbulance ? '#fde047' : '#0284c7'};
            box-shadow: 0 4px 14px rgba(0,0,0,0.25);
            display: flex;
            align-items: center;
            justify-content: center;
            color: ${isSelected || hasIncident || hasAmbulance ? '#ffffff' : '#0284c7'};
            font-weight: 800;
            font-size: ${isSelected ? '13px' : '11px'};
            cursor: pointer;
            transition: all 0.2s ease;
          ">
            ${hasIncident ? '⚠️' : hasAmbulance ? '🚑' : j.junction_id.replace('J-', '').replace('DEL-', '').replace('BOM-', '').replace('HYD-', '').replace('BLR-', '')}
            ${isSelected ? '<div style="position: absolute; top: -6px; right: -6px; width: 14px; height: 14px; background: #10b981; border: 2px solid #ffffff; border-radius: 50%;"></div>' : ''}
          </div>
        `;

        const icon = L.divIcon({
          html: markerHtml,
          className: 'leaflet-custom-div-icon',
          iconSize: [40, 40],
          iconAnchor: [20, 20]
        });

        const marker = L.marker([lat, lng], { icon });

        // Popup with details and quick action buttons
        const popupContent = `
          <div style="font-family: system-ui, -apple-system, sans-serif; min-width: 220px; padding: 4px;">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px;">
              <span style="font-weight: 800; font-size: 14px; color: #0f172a;">${j.name}</span>
              <span style="background: #e0f2fe; color: #0284c7; font-size: 10px; font-weight: 700; padding: 2px 6px; border-radius: 4px;">${j.junction_id}</span>
            </div>
            <div style="font-size: 12px; color: #64748b; margin-bottom: 8px;">📍 ${j.location || 'Urban Arterial'}</div>
            
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 6px; font-size: 11px; margin-bottom: 10px;">
              <div style="color: #475569;"><b>GPS:</b> ${lat.toFixed(4)}° N, ${lng.toFixed(4)}° E</div>
              <div style="color: #475569;"><b>City:</b> ${j.city || 'DELHI'}</div>
            </div>

            <div style="display: flex; gap: 4px; flex-direction: column;">
              <button id="btn-select-${j.junction_id}" style="
                background: #0284c7; color: white; border: none; padding: 6px 10px; border-radius: 6px; font-size: 11px; font-weight: 700; cursor: pointer; display: flex; align-items: center; justify-content: center; gap: 4px;
              ">
                🎯 Select & Inspect Junction
              </button>
              <div style="display: flex; gap: 4px; margin-top: 2px;">
                <button id="btn-nav-origin-${j.junction_id}" style="
                  flex: 1; background: #f0fdf4; color: #16a34a; border: 1px solid #bbf7d0; padding: 4px; border-radius: 4px; font-size: 10px; font-weight: 700; cursor: pointer;
                ">
                  🚩 Route Start
                </button>
                <button id="btn-nav-dest-${j.junction_id}" style="
                  flex: 1; background: #eff6ff; color: #2563eb; border: 1px solid #bfdbfe; padding: 4px; border-radius: 4px; font-size: 10px; font-weight: 700; cursor: pointer;
                ">
                  🏁 Destination
                </button>
              </div>
            </div>
          </div>
        `;

        marker.bindPopup(popupContent);

        marker.on('popupopen', () => {
          const btnSelect = document.getElementById(`btn-select-${j.junction_id}`);
          if (btnSelect) {
            btnSelect.onclick = () => {
              if (onSelectJunction) onSelectJunction(j.junction_id);
            };
          }
          const btnOrigin = document.getElementById(`btn-nav-origin-${j.junction_id}`);
          if (btnOrigin) {
            btnOrigin.onclick = () => {
              if (onSetNavOrigin) onSetNavOrigin(j.junction_id);
            };
          }
          const btnDest = document.getElementById(`btn-nav-dest-${j.junction_id}`);
          if (btnDest) {
            btnDest.onclick = () => {
              if (onSetNavDestination) onSetNavDestination(j.junction_id);
            };
          }
        });

        marker.on('click', () => {
          if (onSelectJunction) onSelectJunction(j.junction_id);
        });

        markersGroupRef.current.addLayer(marker);
      });
    }
  }, [junctions, selectedJunction, incidents, activeAmbulances, navigationRoute, activeLayers, mapMode]);

  const selectedNode = junctions.find(j => j.junction_id === selectedJunction) || junctions[0];

  return (
    <div className="live-map-wrapper" style={{ position: 'relative', borderRadius: '12px', overflow: 'hidden', border: '1px solid #e2e8f0', background: '#ffffff', boxShadow: '0 4px 20px rgba(0,0,0,0.06)' }}>
      
      {/* Top Map Toolbar with Real-World City Selector */}
      <div style={{
        padding: '0.85rem 1.25rem',
        background: '#ffffff',
        borderBottom: '1px solid #e2e8f0',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: '0.75rem'
      }}>
        {/* City Selector Pills */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
          <span style={{ fontSize: '0.82rem', fontWeight: 800, color: '#475569', display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
            <Building2 size={15} color="#0284c7" /> Metropolitan Grid:
          </span>
          {Object.entries(CITY_PRESETS).map(([key, city]) => (
            <button
              key={key}
              onClick={() => handleCityChange(key)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.35rem',
                padding: '0.35rem 0.75rem',
                borderRadius: '8px',
                fontSize: '0.8rem',
                fontWeight: currentCity === key ? 800 : 600,
                background: currentCity === key ? '#0284c7' : '#f1f5f9',
                color: currentCity === key ? '#ffffff' : '#334155',
                border: `1px solid ${currentCity === key ? '#0284c7' : '#cbd5e1'}`,
                cursor: 'pointer',
                transition: 'all 0.15s ease'
              }}
            >
              <span>{city.icon}</span> {city.name.split(' ')[0]}
            </button>
          ))}
        </div>

        {/* Map View & Layer Controls */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <button
            onClick={() => {
              const cityCfg = CITY_PRESETS[currentCity] || CITY_PRESETS.DELHI;
              if (leafletMapRef.current) leafletMapRef.current.flyTo(cityCfg.center, cityCfg.zoom);
            }}
            title="Reset Map View"
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.3rem',
              padding: '0.35rem 0.65rem',
              borderRadius: '6px',
              background: '#f8fafc',
              border: '1px solid #cbd5e1',
              color: '#475569',
              fontSize: '0.78rem',
              fontWeight: 700,
              cursor: 'pointer'
            }}
          >
            <RotateCcw size={13} /> Reset View
          </button>
          <span style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '0.35rem',
            padding: '0.35rem 0.65rem',
            background: '#ecfeff',
            color: '#0891b2',
            border: '1px solid #a5f3fc',
            borderRadius: '6px',
            fontSize: '0.78rem',
            fontWeight: 700
          }}>
            <Navigation2 size={13} /> Real Street Map (OSM)
          </span>
        </div>
      </div>

      {/* Real-World Leaflet Map Container */}
      <div 
        ref={mapContainerRef} 
        style={{ 
          width: '100%', 
          height: '520px', 
          background: '#f8fafc',
          position: 'relative' 
        }} 
      />

      {/* Floating Selected Junction Quick HUD */}
      {selectedNode && (
        <div style={{
          position: 'absolute',
          bottom: '16px',
          left: '16px',
          background: 'rgba(255, 255, 255, 0.95)',
          backdropFilter: 'blur(8px)',
          border: '1px solid #e2e8f0',
          borderRadius: '10px',
          padding: '0.75rem 1rem',
          boxShadow: '0 8px 24px rgba(0,0,0,0.12)',
          maxWidth: '360px',
          zIndex: 1000
        }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '4px' }}>
            <div style={{ fontSize: '0.88rem', fontWeight: 800, color: '#0f172a' }}>{selectedNode.name}</div>
            <span style={{ fontSize: '0.75rem', fontWeight: 800, color: '#0284c7', background: '#e0f2fe', padding: '2px 6px', borderRadius: '4px' }}>{selectedNode.junction_id}</span>
          </div>
          <div style={{ fontSize: '0.78rem', color: '#64748b', marginBottom: '6px' }}>
            📍 {selectedNode.location || 'Urban Arterial Intersection'}
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', fontSize: '0.75rem', color: '#475569' }}>
            <span>🌐 <b>GPS:</b> {selectedNode.latitude}° N, {selectedNode.longitude}° E</span>
            <span>🏛️ <b>City:</b> {selectedNode.city || currentCity}</span>
          </div>
        </div>
      )}

      {/* Floating Active Route Banner (if navigation active) */}
      {navigationRoute && (
        <div style={{
          position: 'absolute',
          top: '64px',
          left: '16px',
          background: 'rgba(255, 255, 255, 0.95)',
          backdropFilter: 'blur(8px)',
          border: '1.5px solid #0284c7',
          borderRadius: '10px',
          padding: '0.65rem 1rem',
          boxShadow: '0 8px 24px rgba(2, 132, 199, 0.15)',
          zIndex: 1000,
          display: 'flex',
          alignItems: 'center',
          gap: '0.75rem'
        }}>
          <div style={{ width: '10px', height: '10px', borderRadius: '50%', background: '#0284c7', animation: 'pulse 1.5s infinite' }} />
          <div>
            <div style={{ fontSize: '0.82rem', fontWeight: 800, color: '#0284c7' }}>⚡ Active Citizen Shortest Path</div>
            <div style={{ fontSize: '0.75rem', color: '#475569' }}>
              Transit: <b>{Math.round((navigationRoute.total_travel_time_seconds || 180) / 60)} mins</b> ({navigationRoute.total_distance_km || 4.2} km) • {navigationRoute.route_summary || 'Fastest Route'}
            </div>
          </div>
        </div>
      )}

    </div>
  );
}
