'use client';
import React, { useRef, useState } from 'react';
import { useEffect } from 'react';
import { Loader } from '@googlemaps/js-api-loader';

export const Maps = ({ address, crimeData }) => {
  const mapRef = useRef(null);
  const mapInstanceRef = useRef(null);
  const markersRef = useRef([]);
  const crimeCirclesRef = useRef([]);
  const [addressHistory, setAddressHistory] = useState([]);

  useEffect(() => {
    console.log('Address received:', address);
    if (address && address.lat && address.lng) {
      setAddressHistory((prev) => {
        const exists = prev.some(
          (item) => item.lat === address.lat && item.lng === address.lng
        );
        if (!exists) {
          return [...prev, address];
        }
        return prev;
      });
    }
  }, [address]);

  const addMarker = async (location, map) => {
    try {
      if (!map || !location || !location.lat || !location.lng) return;

      console.log('Adding marker at:', location);

      const { AdvancedMarkerElement } = await google.maps.importLibrary(
        'marker'
      );

      const markerContent = document.createElement('div');
      const markerIndex = markersRef.current.length + 1;
      markerContent.innerHTML = `
        <div style="background-color: #e74c3c; color: white; border-radius: 50%; width: 30px; height: 30px; display: flex; justify-content: center; align-items: center; font-weight: bold; box-shadow: 0 2px 6px rgba(0,0,0,0.3);">
          ${markerIndex}
        </div>
      `;

      const marker = new AdvancedMarkerElement({
        map: map,
        position: { lat: location.lat, lng: location.lng },
        content: markerContent,
        title: location.title || `Alert Location ${markerIndex}`,
      });

      markersRef.current.push(marker);
    } catch (error) {
      console.error('Error adding marker:', error);
    }
  };

  useEffect(() => {
    console.log('Initializing map...');

    const initMap = async () => {
      const loader = new Loader({
        apiKey: 'AIzaSyCb4aE_i6ANsLCSaGLt3NuUqkDdkcbkV7o',
        version: 'weekly',
      });

      await loader.load();

      const { Map } = await google.maps.importLibrary('maps');

      const indiaCenter = { lat: 23.5937, lng: 80.9629 };

      const mapOptions = {
        center: indiaCenter,
        zoom: 5,
        mapId: 'India Monitoring Map',
        minZoom: 3,
        restriction: {
          latLngBounds: { north: 37.5, south: 6.5, west: 68.0, east: 98.0 },
          strictBounds: false,
        },
        mapTypeId: google.maps.MapTypeId.TERRAIN,
      };

      const map = new Map(mapRef.current, mapOptions);
      mapInstanceRef.current = map;

      console.log('Map initialized successfully.');
    };

    initMap();
  }, []);

  useEffect(() => {
    console.log('Address history updated:', addressHistory);
    if (!mapInstanceRef.current) return;

    const latestAddress = addressHistory[addressHistory.length - 1];
    if (latestAddress && latestAddress.lat && latestAddress.lng) {
      addMarker(latestAddress, mapInstanceRef.current);
    }
  }, [addressHistory]);

  useEffect(() => {
    console.log('Processing crime data:', crimeData);
    const map = mapInstanceRef.current;
    if (!map || !crimeData || !crimeData.length) return;

    // Clear existing circles
    crimeCirclesRef.current.forEach((circle) => {
      if (circle) circle.setMap(null);
    });
    crimeCirclesRef.current = [];

    // Calculate max count for normalization
    const maxCount = Math.max(...crimeData.map((city) => city.count || 0));
    console.log('Max crime count:', maxCount);
    if (!maxCount) return;

    const geocodeCity = async (city) => {
      if (!city || !city.city) return null;

      try {
        const geocodeUrl = `https://maps.googleapis.com/maps/api/geocode/json?address=${encodeURIComponent(
          city.city + ', India'
        )}&key=AIzaSyCb4aE_i6ANsLCSaGLt3NuUqkDdkcbkV7o`;

        const response = await fetch(geocodeUrl);
        const data = await response.json();

        if (data.results && data.results.length > 0) {
          const { lat, lng } = data.results[0].geometry.location;
          console.log(`Geocoded ${city.city}:`, { lat, lng });
          return { lat, lng };
        }
        console.warn(`Failed to geocode ${city.city}`);
        return null;
      } catch (error) {
        console.error(`Error geocoding ${city.city}:`, error);
        return null;
      }
    };

    const processCities = async () => {
      for (const city of crimeData) {
        if (!city || !city.count) continue;

        const location = await geocodeCity(city);
        if (!location) continue;

        const normalizedCount = city.count / maxCount;
        console.log(
          `City: ${city.city}, Count: ${city.count}, Normalized: ${normalizedCount}`
        );

        const getColor = (value) => {
          const r = Math.min(255, Math.round(255 * value * 2));
          const g = Math.min(255, Math.round(255 * (1 - value)));
          const b = 0;
          return `rgb(${r}, ${g}, ${b})`;
        };

        const color = getColor(normalizedCount);
        const opacity = 0.35 + normalizedCount * 0.4;
        const radius = 20000 + normalizedCount * 60000;

        try {
          console.log(`Drawing circle for ${city.city} at`, location);
          const cityCircle = new google.maps.Circle({
            strokeColor: color,
            strokeOpacity: 0.8,
            strokeWeight: 2,
            fillColor: color,
            fillOpacity: opacity,
            map: map,
            center: { lat: location.lat, lng: location.lng },
            radius: radius,
            clickable: true,
          });

          const infoContent = `
            <div style="padding: 10px; max-width: 200px;">
              <h3 style="margin: 0 0 5px; font-size: 16px;">${city.city}</h3>
              <p style="margin: 0; font-size: 14px;">Crime Count: <strong>${city.count}</strong></p>
            </div>
          `;

          const infoWindow = new google.maps.InfoWindow({
            content: infoContent,
          });

          google.maps.event.addListener(cityCircle, 'click', function () {
            infoWindow.setPosition({ lat: location.lat, lng: location.lng });
            infoWindow.open(map);
          });

          crimeCirclesRef.current.push(cityCircle);
        } catch (error) {
          console.error(`Error creating circle for ${city.city}:`, error);
        }
      }
    };

    processCities();
  }, [crimeData]);

  return (
    <div
      style={{ height: '100%', width: '100%', position: 'relative' }}
      ref={mapRef}
    >
      <div
        style={{
          position: 'absolute',
          bottom: '10px',
          left: '10px',
          backgroundColor: 'rgba(255,255,255,0.8)',
          padding: '5px',
          borderRadius: '3px',
          fontSize: '12px',
          fontWeight: 'bold',
        }}
      >
        India Monitoring System
      </div>
    </div>
  );
};
