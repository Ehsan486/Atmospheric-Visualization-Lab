import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";

const R = 2.2;
const MAP_W = 6.6;
const MAP_H = 3.3;
const DEG = Math.PI / 180;
const WIND_CELL_DEG = 8;

type Mode = "all" | "primitives" | "flows" | "storm" | "wind";
type PrimitiveType = "circle" | "triangle" | "square";

type Primitive = {
  id: string;
  type: PrimitiveType;
  lat: number;
  lon: number;
  radiusDeg: number;
  color: string;
};

type Cyclone = {
  id: string;
  lat: number;
  lon: number;
  radiusDeg: number;
  color: string;
  phase: number;
};

type StormPoint = {
  id: string;
  name: string;
  time: string;
  lat: number;
  lon: number;
  wind?: number;
  pressure?: number;
};

type StormTrack = {
  id: string;
  name: string;
  color: string;
  points: StormPoint[];
};

type WindVector = {
  lat: number;
  lon: number;
  u: number;
  v: number;
  speed: number;
};

type WindIndex = {
  vectors: WindVector[];
  bins: Map<string, WindVector[]>;
};

type WindPath = {
  id: string;
  color: string;
  points: { lat: number; lon: number }[];
};

type Offset = {
  lat: number;
  lon: number;
};

const primitives: Primitive[] = [
  { id: "circle", type: "circle", lat: 10, lon: 160, radiusDeg: 14, color: "#f97316" },
  { id: "triangle", type: "triangle", lat: 55, lon: -30, radiusDeg: 13, color: "#38bdf8" },
  { id: "square", type: "square", lat: -35, lon: 70, radiusDeg: 12, color: "#a78bfa" },
];

const cyclones: Cyclone[] = [
  { id: "northPacific", lat: 48, lon: -165, radiusDeg: 24, color: "#bef264", phase: 0 },
  { id: "indian", lat: -24, lon: 74, radiusDeg: 20, color: "#fb7185", phase: 1.2 },
  { id: "atlantic", lat: 32, lon: -58, radiusDeg: 18, color: "#fde047", phase: 2.1 },
];

const sampleStormCsv = `id,name,time,lat,lon,wind,pressure
AL012024,Sample Atlantic Storm,2024-08-01T00:00:00Z,12.0,-42.0,35,1006
AL012024,Sample Atlantic Storm,2024-08-01T06:00:00Z,12.6,-44.8,40,1002
AL012024,Sample Atlantic Storm,2024-08-01T12:00:00Z,13.4,-47.2,45,998
AL012024,Sample Atlantic Storm,2024-08-01T18:00:00Z,14.6,-49.9,55,992
AL012024,Sample Atlantic Storm,2024-08-02T00:00:00Z,16.2,-52.2,65,985
AL012024,Sample Atlantic Storm,2024-08-02T06:00:00Z,18.4,-54.1,75,976
AL012024,Sample Atlantic Storm,2024-08-02T12:00:00Z,21.0,-55.4,90,965
AL012024,Sample Atlantic Storm,2024-08-02T18:00:00Z,24.2,-55.6,105,950
AL012024,Sample Atlantic Storm,2024-08-03T00:00:00Z,28.1,-54.2,95,960
AL012024,Sample Atlantic Storm,2024-08-03T06:00:00Z,32.4,-51.6,80,972
AL012024,Sample Atlantic Storm,2024-08-03T12:00:00Z,37.8,-47.9,65,984`;

const sampleWindCsv = `lat,lon,u,v
72,-150,12,4
66,-130,20,8
58,-112,28,4
50,-95,20,-4
42,-70,12,-8
35,-45,8,-12
30,-20,18,4
45,0,28,8
55,30,34,5
60,70,24,-2
50,110,18,-8
38,145,10,-12
25,160,-8,-6
10,140,-14,2
-5,115,-20,4
-20,95,-16,8
-35,70,-8,14
-45,35,16,20
-55,0,36,8
-58,-40,44,-2
-52,-80,38,-10
-45,-120,30,-12
-35,-155,22,-4
-20,-130,8,12
-5,-95,-8,18
12,-70,-14,4
22,-55,-18,-10
5,20,8,-12
-15,30,18,-4
-30,45,24,6
-42,85,16,18
-48,130,30,10
-50,170,42,0`;

const stormColors = ["#fb7185", "#f97316", "#facc15", "#4ade80", "#38bdf8", "#a78bfa", "#f472b6"];

function wrapLon(lon: number) {
  let v = ((((lon + 180) % 360) + 360) % 360) - 180;
  if (Object.is(v, -180)) v = 180;
  return v;
}

function normalizeLatLon(lat: number, lon: number) {
  let a = lat;
  let o = lon;

  while (a > 90) {
    a = 180 - a;
    o += 180;
  }

  while (a < -90) {
    a = -180 - a;
    o += 180;
  }

  return { lat: a, lon: wrapLon(o) };
}

function shifted(lat: number, lon: number, offset: Offset) {
  return normalizeLatLon(lat + offset.lat, lon + offset.lon);
}

function latLonToVector3(lat: number, lon: number, radius = R) {
  const phi = (90 - lat) * DEG;
  const theta = (lon + 180) * DEG;
  const x = -radius * Math.sin(phi) * Math.cos(theta);
  const y = radius * Math.cos(phi);
  const z = radius * Math.sin(phi) * Math.sin(theta);
  return new THREE.Vector3(x, y, z);
}

function equirect(lat: number, lon: number) {
  const maxMercatorLat = 85.05112878;
  const clampedLat = Math.max(-maxMercatorLat, Math.min(maxMercatorLat, lat));
  const phi = clampedLat * DEG;
  const mercatorY = Math.log(Math.tan(Math.PI / 4 + phi / 2));
  const x = (lon / 180) * (MAP_W / 2);
  const y = (mercatorY / Math.PI) * (MAP_H / 2);
  return new THREE.Vector3(x, y, 0);
}

function parseNumber(value: unknown) {
  if (value === undefined || value === null || value === "") return undefined;
  const n = Number(String(value).trim());
  return Number.isFinite(n) ? n : undefined;
}

function parseCsvLine(line: string) {
  const values: string[] = [];
  let current = "";
  let quoted = false;

  for (let i = 0; i < line.length; i++) {
    const char = line[i];
    const next = line[i + 1];

    if (char === '"' && quoted && next === '"') {
      current += '"';
      i += 1;
    } else if (char === '"') {
      quoted = !quoted;
    } else if (char === "," && !quoted) {
      values.push(current.trim());
      current = "";
    } else {
      current += char;
    }
  }

  values.push(current.trim());
  return values;
}

function moveFromCenter(centerLat: number, centerLon: number, angle: number, distanceDeg: number) {
  // Geodesic construction on the sphere.
  // The previous version used linear latitude and longitude offsets, which is only a local approximation.
  // This version moves from the center by an angular distance along a bearing, so circles remain valid spherical circles even near poles.
  const angularDistance = distanceDeg * DEG;
  const lat1 = centerLat * DEG;
  const lon1 = centerLon * DEG;

  // In the rest of the code, angle = 0 points east and angle = pi/2 points north.
  // The geodesic formula uses bearing = 0 for north and pi/2 for east.
  const bearing = Math.PI / 2 - angle;

  const sinLat1 = Math.sin(lat1);
  const cosLat1 = Math.cos(lat1);
  const sinD = Math.sin(angularDistance);
  const cosD = Math.cos(angularDistance);
  const sinLat2 = sinLat1 * cosD + cosLat1 * sinD * Math.cos(bearing);
  const lat2 = Math.asin(Math.max(-1, Math.min(1, sinLat2)));
  const y = Math.sin(bearing) * sinD * cosLat1;
  const x = cosD - sinLat1 * Math.sin(lat2);
  const lon2 = lon1 + Math.atan2(y, x);

  return normalizeLatLon(lat2 / DEG, lon2 / DEG);
}

function makePrimitiveRing(primitive: Primitive, offset: Offset, segments = 96) {
  const center = shifted(primitive.lat, primitive.lon, offset);
  const points: { lat: number; lon: number }[] = [];
  const sides = primitive.type === "circle" ? segments : primitive.type === "triangle" ? 3 : 4;
  const total = primitive.type === "circle" ? segments + 1 : sides + 1;
  const rotation = primitive.type === "square" ? Math.PI / 4 : Math.PI / 2;

  for (let i = 0; i < total; i++) {
    const t = primitive.type === "circle" ? (i / segments) * Math.PI * 2 : ((i % sides) / sides) * Math.PI * 2;
    points.push(moveFromCenter(center.lat, center.lon, t + rotation, primitive.radiusDeg));
  }

  return points;
}

function parseStormCsv(text: string) {
  const lines = text.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
  if (lines.length < 2) throw new Error("CSV needs a header row and at least one data row.");

  const headers = parseCsvLine(lines[0]).map((header) => header.toLowerCase());
  const required = ["id", "lat", "lon"];

  for (const key of required) {
    if (!headers.includes(key)) throw new Error(`CSV missing required column: ${key}`);
  }

  const points: StormPoint[] = [];

  for (let i = 1; i < lines.length; i++) {
    const row = parseCsvLine(lines[i]);
    const record: Record<string, string> = {};

    headers.forEach((header, index) => {
      record[header] = row[index] ?? "";
    });

    const lat = parseNumber(record.lat);
    const lon = parseNumber(record.lon);

    if (lat === undefined || lon === undefined) throw new Error(`Row ${i + 1} has invalid latitude or longitude.`);

    const id = record.id || "storm";
    const name = record.name || id;
    const time = record.time || record.datetime || record.date || String(i);
    const wind = parseNumber(record.wind ?? record.wind_kt ?? record.wind_kts ?? record.max_wind);
    const pressure = parseNumber(record.pressure ?? record.pressure_mb ?? record.min_pressure);
    const normalized = normalizeLatLon(lat, lon);

    points.push({ id, name, time, lat: normalized.lat, lon: normalized.lon, wind, pressure });
  }

  return buildStormTracks(points);
}

function parseStormJson(text: string) {
  const raw = JSON.parse(text);
  let records: unknown[] = [];

  if (Array.isArray(raw)) {
    records = raw;
  } else if (raw && typeof raw === "object" && Array.isArray((raw as { points?: unknown[] }).points)) {
    records = (raw as { points: unknown[] }).points;
  } else if (raw && typeof raw === "object" && Array.isArray((raw as { features?: unknown[] }).features)) {
    const features = (raw as { features: unknown[] }).features;
    records = features.flatMap((feature) => {
      const f = feature as { properties?: Record<string, unknown>; geometry?: { coordinates?: unknown } };
      const props = f.properties ?? {};
      const coords = f.geometry?.coordinates;

      if (!Array.isArray(coords)) return [];

      if (coords.length >= 2 && typeof coords[0] === "number" && typeof coords[1] === "number") {
        return [{ ...props, lon: coords[0], lat: coords[1] }];
      }

      return coords
        .filter((c): c is [number, number] => Array.isArray(c) && typeof c[0] === "number" && typeof c[1] === "number")
        .map((c, index) => ({ ...props, lon: c[0], lat: c[1], time: props.time ?? props.datetime ?? String(index) }));
    });
  } else {
    throw new Error("JSON must be an array, an object with a points array, or simple GeoJSON features.");
  }

  const points = records.map((record, index) => {
    if (!record || typeof record !== "object") throw new Error(`JSON record ${index + 1} is not an object.`);
    const r = record as Record<string, unknown>;
    const lat = parseNumber(r.lat ?? r.latitude);
    const lon = parseNumber(r.lon ?? r.lng ?? r.longitude);

    if (lat === undefined || lon === undefined) throw new Error(`JSON record ${index + 1} has invalid latitude or longitude.`);

    const id = String(r.id ?? r.storm_id ?? r.stormId ?? "storm");
    const name = String(r.name ?? r.storm_name ?? r.stormName ?? id);
    const time = String(r.time ?? r.datetime ?? r.date ?? index);
    const wind = parseNumber(r.wind ?? r.wind_kt ?? r.wind_kts ?? r.max_wind);
    const pressure = parseNumber(r.pressure ?? r.pressure_mb ?? r.min_pressure);
    const normalized = normalizeLatLon(lat, lon);

    return { id, name, time, lat: normalized.lat, lon: normalized.lon, wind, pressure };
  });

  return buildStormTracks(points);
}

function buildStormTracks(points: StormPoint[]) {
  const grouped = new Map<string, StormPoint[]>();

  points.forEach((point) => {
    const list = grouped.get(point.id) ?? [];
    list.push(point);
    grouped.set(point.id, list);
  });

  return Array.from(grouped.entries()).map(([id, list], index) => {
    const sorted = [...list].sort((a, b) => String(a.time).localeCompare(String(b.time)));
    return {
      id,
      name: sorted[0]?.name ?? id,
      color: stormColors[index % stormColors.length],
      points: sorted,
    };
  });
}

function parseWindCsv(text: string) {
  const lines = text.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
  if (lines.length < 2) throw new Error("Wind CSV needs a header row and at least one data row.");

  const headers = parseCsvLine(lines[0]).map((header) => header.toLowerCase());
  const findHeader = (...names: string[]) => names.find((name) => headers.includes(name));
  const latKey = findHeader("lat", "latitude");
  const lonKey = findHeader("lon", "lng", "long", "longitude");
  const uKey = findHeader("u", "uwind", "u_wind", "eastward", "eastward_wind");
  const vKey = findHeader("v", "vwind", "v_wind", "northward", "northward_wind");

  if (!latKey || !lonKey || !uKey || !vKey) throw new Error("Wind CSV requires lat, lon, u, and v columns.");

  const vectors: WindVector[] = [];

  for (let i = 1; i < lines.length; i++) {
    const row = parseCsvLine(lines[i]);
    const record: Record<string, string> = {};

    headers.forEach((header, index) => {
      record[header] = row[index] ?? "";
    });

    const lat = parseNumber(record[latKey]);
    const lon = parseNumber(record[lonKey]);
    const u = parseNumber(record[uKey]);
    const v = parseNumber(record[vKey]);

    if (lat === undefined || lon === undefined || u === undefined || v === undefined) {
      throw new Error(`Wind row ${i + 1} has invalid lat, lon, u, or v.`);
    }

    const normalized = normalizeLatLon(lat, lon);
    vectors.push({ lat: normalized.lat, lon: normalized.lon, u, v, speed: Math.hypot(u, v) });
  }

  return vectors;
}

function windCellKey(lat: number, lon: number) {
  const latIndex = Math.floor((lat + 90) / WIND_CELL_DEG);
  const lonIndex = Math.floor((wrapLon(lon) + 180) / WIND_CELL_DEG);
  return `${latIndex}:${lonIndex}`;
}

function buildWindIndex(vectors: WindVector[]): WindIndex {
  const bins = new Map<string, WindVector[]>();

  vectors.forEach((vector) => {
    const key = windCellKey(vector.lat, vector.lon);
    const list = bins.get(key) ?? [];
    list.push(vector);
    bins.set(key, list);
  });

  return { vectors, bins };
}

function sampleImportedWind(lat: number, lon: number, index: WindIndex | null) {
  if (!index || index.vectors.length === 0) return null;

  const latIndex = Math.floor((lat + 90) / WIND_CELL_DEG);
  const lonBins = Math.ceil(360 / WIND_CELL_DEG);
  const lonIndex = Math.floor((wrapLon(lon) + 180) / WIND_CELL_DEG);
  let best: WindVector | null = null;
  let bestD2 = Infinity;

  for (let di = -1; di <= 1; di++) {
    for (let dj = -1; dj <= 1; dj++) {
      const li = latIndex + di;
      const lj = (lonIndex + dj + lonBins) % lonBins;
      const bucket = index.bins.get(`${li}:${lj}`) ?? [];

      for (const vector of bucket) {
        const dLat = vector.lat - lat;
        const dLon = wrapLon(vector.lon - lon) * Math.max(0.25, Math.cos(lat * DEG));
        const d2 = dLat * dLat + dLon * dLon;
        if (d2 < bestD2) {
          bestD2 = d2;
          best = vector;
        }
      }
    }
  }

  if (!best) return null;
  if (bestD2 > 18 * 18) return null;
  return { u: best.u, v: best.v, speed: best.speed };
}

function vortexContribution(lat: number, lon: number, centerLat: number, centerLon: number, strength: number, radius: number, clockwise: boolean) {
  const dx = wrapLon(lon - centerLon) * Math.max(0.22, Math.cos(centerLat * DEG));
  const dy = lat - centerLat;
  const d2 = dx * dx + dy * dy + 9;
  const falloff = Math.exp(-d2 / (radius * radius));
  const sign = clockwise ? -1 : 1;
  const scale = (strength * falloff) / Math.sqrt(d2);

  return {
    u: sign * -dy * scale,
    v: sign * dx * scale,
  };
}

function proceduralWind(lat: number, lon: number) {
  let u = 18 * Math.cos(lat * DEG) + 10 * Math.sin((lon + lat * 1.7) * DEG);
  let v = 6 * Math.sin((lon * 1.4 - lat) * DEG);

  const vortices = [
    { lat: 42, lon: -150, strength: 90, radius: 32, clockwise: false },
    { lat: 50, lon: -35, strength: 75, radius: 26, clockwise: false },
    { lat: 62, lon: 65, strength: 80, radius: 34, clockwise: false },
    { lat: 28, lon: 135, strength: 65, radius: 24, clockwise: false },
    { lat: -35, lon: -70, strength: 90, radius: 30, clockwise: true },
    { lat: -45, lon: 40, strength: 110, radius: 36, clockwise: true },
    { lat: -52, lon: 120, strength: 125, radius: 40, clockwise: true },
    { lat: -58, lon: -150, strength: 105, radius: 38, clockwise: true },
  ];

  vortices.forEach((vortex) => {
    const c = vortexContribution(lat, lon, vortex.lat, vortex.lon, vortex.strength, vortex.radius, vortex.clockwise);
    u += c.u;
    v += c.v;
  });

  if (Math.abs(lat) > 35) {
    u += 22 * Math.sign(lat) * Math.sin((lon * 0.65 + lat) * DEG);
  }

  return { u, v, speed: Math.hypot(u, v) };
}

function sampleWind(lat: number, lon: number, index: WindIndex | null) {
  return sampleImportedWind(lat, lon, index) ?? proceduralWind(lat, lon);
}

function makeProceduralSeeds() {
  const seeds: { lat: number; lon: number }[] = [];
  const rows = 34;
  const cols = 48;

  for (let r = 0; r < rows; r++) {
    const lat = -82 + (164 * r) / (rows - 1);
    const rowShift = r % 2 === 0 ? 0 : 3.75;
    for (let c = 0; c < cols; c++) {
      const lon = wrapLon(-180 + (360 * c) / cols + rowShift);
      if ((r + c) % 3 !== 0) seeds.push({ lat, lon });
    }
  }

  return seeds;
}

const proceduralSeeds = makeProceduralSeeds();

function makeWindSeeds(index: WindIndex | null) {
  if (!index || index.vectors.length === 0) return proceduralSeeds;
  const maxSeeds = 650;
  const stride = Math.max(1, Math.floor(index.vectors.length / maxSeeds));
  return index.vectors.filter((_, i) => i % stride === 0).map((v) => ({ lat: v.lat, lon: v.lon }));
}

function speedColor(speed: number) {
  if (speed > 70) return "#fb923c";
  if (speed > 50) return "#fde047";
  if (speed > 32) return "#bef264";
  if (speed > 18) return "#5eead4";
  return "#38bdf8";
}

function traceWindPath(seed: { lat: number; lon: number }, index: WindIndex | null) {
  const points: { lat: number; lon: number }[] = [];
  let current = normalizeLatLon(seed.lat, seed.lon);
  let speedSum = 0;
  let count = 0;

  for (let i = 0; i < 26; i++) {
    points.push(current);
    const wind = sampleWind(current.lat, current.lon, index);
    const magnitude = Math.max(0.001, Math.hypot(wind.u, wind.v));
    const step = Math.min(2.8, 0.7 + magnitude / 38);
    const safeCosine = Math.max(0.2, Math.abs(Math.cos(current.lat * DEG)));
    const dLon = (wind.u / magnitude) * step / safeCosine;
    const dLat = (wind.v / magnitude) * step;
    current = normalizeLatLon(current.lat + dLat, current.lon + dLon);
    speedSum += wind.speed;
    count += 1;
  }

  return { points, speed: speedSum / Math.max(1, count) };
}

function buildWindPaths(index: WindIndex | null) {
  const seeds = makeWindSeeds(index);
  return seeds.map((seed, i) => {
    const traced = traceWindPath(seed, index);
    return {
      id: `wind-${i}`,
      color: speedColor(traced.speed),
      points: traced.points,
    };
  });
}

function splitAtSeam(points: THREE.Vector3[]) {
  const groups: THREE.Vector3[][] = [];
  let current: THREE.Vector3[] = [];

  for (let i = 0; i < points.length; i++) {
    const point = points[i];
    if (i > 0) {
      const previous = points[i - 1];
      if (Math.abs(point.x - previous.x) > MAP_W * 0.55) {
        if (current.length > 1) groups.push(current);
        current = [];
      }
    }
    current.push(point);
  }

  if (current.length > 1) groups.push(current);
  return groups;
}

function windRadius(wind?: number) {
  if (!wind) return 0.035;
  return Math.min(0.12, 0.035 + wind / 1400);
}

function runProjectionSelfTests() {
  console.assert(wrapLon(190) === -170, "wrapLon should wrap 190 degrees to negative 170 degrees");
  console.assert(wrapLon(-190) === 170, "wrapLon should wrap negative 190 degrees to 170 degrees");

  const north = normalizeLatLon(100, 20);
  console.assert(north.lat === 80 && north.lon === -160, "north pole crossing should reflect latitude and rotate longitude");

  const circle = makePrimitiveRing(primitives[0], { lat: 0, lon: 0 });
  console.assert(circle.length === 97, "circle should be closed and sampled with 97 points when segments is 96");

  const parsedStorm = parseStormCsv(sampleStormCsv);
  console.assert(parsedStorm.length === 1 && parsedStorm[0].points.length === 11, "sample storm CSV should parse into one track with eleven points");

  const parsedWind = parseWindCsv(sampleWindCsv);
  console.assert(parsedWind.length > 10, "sample wind CSV should parse into several vector records");
}

function Line3D({ points, color, width = 2, opacity = 0.92 }: { points: THREE.Vector3[]; color: string; width?: number; opacity?: number }) {
  const geometry = useMemo(() => new THREE.BufferGeometry().setFromPoints(points), [points]);

  useEffect(() => {
    return () => geometry.dispose();
  }, [geometry]);

  if (points.length < 2) return null;

  return (
    <line geometry={geometry}>
      <lineBasicMaterial attach="material" color={color} linewidth={width} transparent opacity={opacity} />
    </line>
  );
}

function Dot3D({ position, color, radius = 0.05 }: { position: THREE.Vector3; color: string; radius?: number }) {
  return (
    <mesh position={position}>
      <sphereGeometry args={[radius, 16, 16]} />
      <meshBasicMaterial color={color} />
    </mesh>
  );
}

function Dot2D({ position, color, radius = 0.04 }: { position: THREE.Vector3; color: string; radius?: number }) {
  return (
    <mesh position={position}>
      <circleGeometry args={[radius, 20]} />
      <meshBasicMaterial color={color} />
    </mesh>
  );
}

function GlobeGrid() {
  const lines = useMemo(() => {
    const result: { points: THREE.Vector3[]; key: string }[] = [];

    for (let lat = -60; lat <= 60; lat += 30) {
      const points: THREE.Vector3[] = [];
      for (let lon = -180; lon <= 180; lon += 3) points.push(latLonToVector3(lat, lon, R + 0.006));
      result.push({ key: `lat-${lat}`, points });
    }

    for (let lon = -180; lon < 180; lon += 30) {
      const points: THREE.Vector3[] = [];
      for (let lat = -90; lat <= 90; lat += 3) points.push(latLonToVector3(lat, lon, R + 0.006));
      result.push({ key: `lon-${lon}`, points });
    }

    return result;
  }, []);

  return <>{lines.map((line) => <Line3D key={line.key} points={line.points} color="#ffffff" width={1} opacity={0.18} />)}</>;
}

function ContinentLines({ spherical }: { spherical: boolean }) {
  const lines = useMemo(() => {
    const boxes = [
      { name: "northAmerica", lat0: 10, lat1: 72, lon0: -168, lon1: -52 },
      { name: "southAmerica", lat0: -56, lat1: 13, lon0: -82, lon1: -35 },
      { name: "eurasia", lat0: 34, lat1: 76, lon0: -12, lon1: 165 },
      { name: "africa", lat0: -35, lat1: 34, lon0: -18, lon1: 52 },
      { name: "australia", lat0: -44, lat1: -10, lon0: 112, lon1: 154 },
      { name: "greenland", lat0: 58, lat1: 84, lon0: -74, lon1: -12 },
      { name: "antarctica", lat0: -88, lat1: -64, lon0: -180, lon1: 180 },
    ];

    return boxes.map((box) => {
      const points: THREE.Vector3[] = [];
      const project = (lat: number, lon: number) => (spherical ? latLonToVector3(lat, lon, R + 0.014) : equirect(lat, lon));
      const step = spherical ? 3 : 2;

      for (let lon = box.lon0; lon <= box.lon1; lon += step) points.push(project(box.lat0, lon));
      for (let lat = box.lat0; lat <= box.lat1; lat += step) points.push(project(lat, box.lon1));
      for (let lon = box.lon1; lon >= box.lon0; lon -= step) points.push(project(box.lat1, lon));
      for (let lat = box.lat1; lat >= box.lat0; lat -= step) points.push(project(lat, box.lon0));

      return { key: box.name, points };
    });
  }, [spherical]);

  return <>{lines.map((line) => <Line3D key={line.key} points={line.points} color="#c7d2fe" width={1} opacity={0.72} />)}</>;
}

function PrimitiveOnGlobe({ primitive, offset }: { primitive: Primitive; offset: Offset }) {
  const points = useMemo(() => makePrimitiveRing(primitive, offset).map((point) => latLonToVector3(point.lat, point.lon, R + 0.045)), [primitive, offset]);
  return <Line3D points={points} color={primitive.color} width={3} />;
}

function PrimitiveOnMap({ primitive, offset }: { primitive: Primitive; offset: Offset }) {
  const groups = useMemo(() => {
    const points = makePrimitiveRing(primitive, offset).map((point) => equirect(point.lat, point.lon));
    return splitAtSeam(points);
  }, [primitive, offset]);

  return <>{groups.map((group, index) => <Line3D key={`${primitive.id}-${index}`} points={group} color={primitive.color} width={3} />)}</>;
}

function CycloneOnGlobe({ cyclone, offset, time }: { cyclone: Cyclone; offset: Offset; time: number }) {
  return (
    <>
      {[0, 1, 2, 3].map((arm) => {
        const center = shifted(cyclone.lat, cyclone.lon, offset);
        const points: THREE.Vector3[] = [];

        for (let i = 0; i < 72; i++) {
          const k = i / 71;
          const distance = cyclone.radiusDeg * k;
          const angle = arm * Math.PI * 0.5 + k * Math.PI * 3.7 + time * 0.7 + cyclone.phase;
          const point = moveFromCenter(center.lat, center.lon, angle, distance);
          points.push(latLonToVector3(point.lat, point.lon, R + 0.07));
        }

        return <Line3D key={`${cyclone.id}-${arm}`} points={points} color={cyclone.color} width={2} />;
      })}
    </>
  );
}

function CycloneOnMap({ cyclone, offset, time }: { cyclone: Cyclone; offset: Offset; time: number }) {
  return (
    <>
      {[0, 1, 2, 3].map((arm) => {
        const center = shifted(cyclone.lat, cyclone.lon, offset);
        const points: THREE.Vector3[] = [];

        for (let i = 0; i < 72; i++) {
          const k = i / 71;
          const distance = cyclone.radiusDeg * k;
          const angle = arm * Math.PI * 0.5 + k * Math.PI * 3.7 + time * 0.7 + cyclone.phase;
          const point = moveFromCenter(center.lat, center.lon, angle, distance);
          points.push(equirect(point.lat, point.lon));
        }

        return splitAtSeam(points).map((group, index) => <Line3D key={`${cyclone.id}-${arm}-${index}`} points={group} color={cyclone.color} width={2} />);
      })}
    </>
  );
}

function StormTrackOnGlobe({ track, offset, time }: { track: StormTrack; offset: Offset; time: number }) {
  const shiftedPoints = useMemo(() => track.points.map((point) => shifted(point.lat, point.lon, offset)), [track.points, offset]);
  const linePoints = useMemo(() => shiftedPoints.map((point) => latLonToVector3(point.lat, point.lon, R + 0.11)), [shiftedPoints]);
  const activeIndex = track.points.length > 0 ? Math.floor((time * 1.2) % track.points.length) : 0;
  const activePoint = shiftedPoints[activeIndex];
  const activeSource = track.points[activeIndex];

  return (
    <>
      <Line3D points={linePoints} color={track.color} width={3} opacity={0.95} />
      {shiftedPoints.map((point, index) => <Dot3D key={`${track.id}-dot-${index}`} position={latLonToVector3(point.lat, point.lon, R + 0.14)} color={track.color} radius={windRadius(track.points[index].wind)} />)}
      {activePoint && <Dot3D position={latLonToVector3(activePoint.lat, activePoint.lon, R + 0.2)} color="#ffffff" radius={windRadius(activeSource?.wind) + 0.035} />}
    </>
  );
}

function StormTrackOnMap({ track, offset, time }: { track: StormTrack; offset: Offset; time: number }) {
  const shiftedPoints = useMemo(() => track.points.map((point) => shifted(point.lat, point.lon, offset)), [track.points, offset]);
  const mapPoints = useMemo(() => shiftedPoints.map((point) => equirect(point.lat, point.lon)), [shiftedPoints]);
  const groups = useMemo(() => splitAtSeam(mapPoints), [mapPoints]);
  const activeIndex = track.points.length > 0 ? Math.floor((time * 1.2) % track.points.length) : 0;
  const activePoint = shiftedPoints[activeIndex];
  const activeSource = track.points[activeIndex];

  return (
    <>
      {groups.map((group, index) => <Line3D key={`${track.id}-map-line-${index}`} points={group} color={track.color} width={3} opacity={0.95} />)}
      {shiftedPoints.map((point, index) => <Dot2D key={`${track.id}-map-dot-${index}`} position={equirect(point.lat, point.lon)} color={track.color} radius={windRadius(track.points[index].wind)} />)}
      {activePoint && <Dot2D position={equirect(activePoint.lat, activePoint.lon)} color="#ffffff" radius={windRadius(activeSource?.wind) + 0.035} />}
    </>
  );
}

function WindTextureOnGlobe({ paths, offset }: { paths: WindPath[]; offset: Offset }) {
  return (
    <>
      {paths.map((path) => {
        const points = path.points.map((point) => {
          const p = shifted(point.lat, point.lon, offset);
          return latLonToVector3(p.lat, p.lon, R + 0.075);
        });
        return <Line3D key={`globe-${path.id}`} points={points} color={path.color} width={1} opacity={0.68} />;
      })}
    </>
  );
}

function WindTextureOnMap({ paths, offset }: { paths: WindPath[]; offset: Offset }) {
  return (
    <>
      {paths.map((path) => {
        const points = path.points.map((point) => {
          const p = shifted(point.lat, point.lon, offset);
          return equirect(p.lat, p.lon);
        });
        return splitAtSeam(points).map((group, index) => <Line3D key={`map-${path.id}-${index}`} points={group} color={path.color} width={1} opacity={0.72} />);
      })}
    </>
  );
}

function GlobeScene({ mode, offset, paused, stormTracks, windPaths }: { mode: Mode; offset: Offset; paused: boolean; stormTracks: StormTrack[]; windPaths: WindPath[] }) {
  const [time, setTime] = useState(0);

  useFrame((_, delta) => {
    if (!paused) setTime((current) => current + delta);
  });

  return (
    <>
      <ambientLight intensity={1.0} />
      <directionalLight position={[4, 5, 3]} intensity={1.2} />
      <mesh>
        <sphereGeometry args={[R, 96, 96]} />
        <meshStandardMaterial color="#071b3c" roughness={0.85} metalness={0.0} transparent opacity={0.98} />
      </mesh>
      <mesh>
        <sphereGeometry args={[R + 0.012, 96, 96]} />
        <meshBasicMaterial color="#14b8a6" transparent opacity={0.11} wireframe />
      </mesh>
      <GlobeGrid />
      <ContinentLines spherical />
      {(mode === "all" || mode === "wind") && <WindTextureOnGlobe paths={windPaths} offset={offset} />}
      {(mode === "all" || mode === "primitives") && primitives.map((primitive) => <PrimitiveOnGlobe key={primitive.id} primitive={primitive} offset={offset} />)}
      {(mode === "all" || mode === "flows") && cyclones.map((cyclone) => <CycloneOnGlobe key={cyclone.id} cyclone={cyclone} offset={offset} time={time} />)}
      {(mode === "all" || mode === "storm") && stormTracks.map((track) => <StormTrackOnGlobe key={track.id} track={track} offset={offset} time={time} />)}
      <OrbitControls enablePan={false} enableZoom rotateSpeed={0.75} />
    </>
  );
}

function MapGrid() {
  const lines = useMemo(() => {
    const result: { key: string; points: THREE.Vector3[] }[] = [];

    for (let lat = -60; lat <= 60; lat += 30) result.push({ key: `mlat-${lat}`, points: [equirect(lat, -180), equirect(lat, 180)] });
    for (let lon = -180; lon <= 180; lon += 30) result.push({ key: `mlon-${lon}`, points: [equirect(-90, lon), equirect(90, lon)] });

    return result;
  }, []);

  return <>{lines.map((line) => <Line3D key={line.key} points={line.points} color="#64748b" width={1} opacity={0.46} />)}</>;
}

function MapScene({ mode, offset, paused, stormTracks, windPaths }: { mode: Mode; offset: Offset; paused: boolean; stormTracks: StormTrack[]; windPaths: WindPath[] }) {
  const [time, setTime] = useState(0);

  useFrame((_, delta) => {
    if (!paused) setTime((current) => current + delta);
  });

  return (
    <>
      <ambientLight intensity={1} />
      <group position={[0, 0, 0]}>
        <mesh position={[0, 0, -0.02]}>
          <planeGeometry args={[MAP_W, MAP_H]} />
          <meshBasicMaterial color="#061428" transparent opacity={0.98} />
        </mesh>
        {(mode === "all" || mode === "wind") && <WindTextureOnMap paths={windPaths} offset={offset} />}
        <MapGrid />
        <ContinentLines spherical={false} />
        <Line3D points={[equirect(-90, -180), equirect(90, -180)]} color="#ef4444" width={2} />
        <Line3D points={[equirect(-90, 180), equirect(90, 180)]} color="#ef4444" width={2} />
        {(mode === "all" || mode === "primitives") && primitives.map((primitive) => <PrimitiveOnMap key={primitive.id} primitive={primitive} offset={offset} />)}
        {(mode === "all" || mode === "flows") && cyclones.map((cyclone) => <CycloneOnMap key={cyclone.id} cyclone={cyclone} offset={offset} time={time} />)}
        {(mode === "all" || mode === "storm") && stormTracks.map((track) => <StormTrackOnMap key={track.id} track={track} offset={offset} time={time} />)}
      </group>
      <OrbitControls enableRotate={false} enablePan enableZoom zoomSpeed={0.7} />
    </>
  );
}

function ButtonLike({ active, children, onClick }: { active?: boolean; children: React.ReactNode; onClick: () => void }) {
  return (
    <button className={active ? "button active" : "button"} onClick={onClick} type="button">
      {children}
    </button>
  );
}

function Label({ children }: { children: React.ReactNode }) {
  return <span className="label">{children}</span>;
}

export default function App() {
  const [mode, setMode] = useState<Mode>("wind");
  const [offset, setOffset] = useState<Offset>({ lat: 0, lon: 0 });
  const [paused, setPaused] = useState(false);
  const [controlsOpen, setControlsOpen] = useState(false);
  const [stormTracks, setStormTracks] = useState<StormTrack[]>(() => parseStormCsv(sampleStormCsv));
  const [windVectors, setWindVectors] = useState<WindVector[]>([]);
  const [dataMessage, setDataMessage] = useState("Loaded sample storm track. You can import your own CSV or JSON file.");
  const [windMessage, setWindMessage] = useState("Using procedural global wind texture. Import a vector CSV to use actual wind data.");
  const step = useRef(10);

  const windIndex = useMemo(() => buildWindIndex(windVectors), [windVectors]);
  const activeWindIndex = windVectors.length > 0 ? windIndex : null;
  const windPaths = useMemo(() => buildWindPaths(activeWindIndex), [activeWindIndex]);

  const nudge = useCallback((dLat: number, dLon: number) => {
    setOffset((current) => ({ lat: current.lat + dLat, lon: current.lon + dLon }));
  }, []);

  useEffect(() => {
    runProjectionSelfTests();
  }, []);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "ArrowLeft") nudge(0, -step.current);
      if (event.key === "ArrowRight") nudge(0, step.current);
      if (event.key === "ArrowUp") nudge(step.current, 0);
      if (event.key === "ArrowDown") nudge(-step.current, 0);
      if (event.key.toLowerCase() === "r") setOffset({ lat: 0, lon: 0 });
      if (event.key.toLowerCase() === "p") setPaused((value) => !value);
      if (event.key === "1") setMode("wind");
      if (event.key === "2") setMode("storm");
      if (event.key === "3") setMode("flows");
      if (event.key === "4") setMode("primitives");
      if (event.key === "5") setMode("all");
    };

    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [nudge]);

  const loadStormText = useCallback((text: string, fileName: string) => {
    try {
      const tracks = fileName.toLowerCase().endsWith(".json") || text.trim().startsWith("{") || text.trim().startsWith("[") ? parseStormJson(text) : parseStormCsv(text);
      const pointCount = tracks.reduce((sum, track) => sum + track.points.length, 0);
      setStormTracks(tracks);
      setMode("storm");
      setDataMessage(`Loaded ${tracks.length} storm track(s) with ${pointCount} point(s).`);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown parsing error.";
      setDataMessage(`Could not load storm data: ${message}`);
    }
  }, []);

  const loadWindText = useCallback((text: string) => {
    try {
      const vectors = parseWindCsv(text);
      setWindVectors(vectors);
      setMode("wind");
      setWindMessage(`Loaded ${vectors.length} wind vector(s). Streamlines now use imported data where available.`);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown parsing error.";
      setWindMessage(`Could not load wind vectors: ${message}`);
    }
  }, []);

  const onStormFileChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => loadStormText(String(reader.result ?? ""), file.name);
    reader.onerror = () => setDataMessage("Could not read the selected storm file.");
    reader.readAsText(file);
  }, [loadStormText]);

  const onWindFileChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => loadWindText(String(reader.result ?? ""));
    reader.onerror = () => setWindMessage("Could not read the selected wind file.");
    reader.readAsText(file);
  }, [loadWindText]);

  const copySampleCsv = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(sampleStormCsv);
      setDataMessage("Sample storm CSV copied to clipboard.");
    } catch {
      setDataMessage("Clipboard copy failed. You can copy the sample from the source code.");
    }
  }, []);

  const copySampleWindCsv = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(sampleWindCsv);
      setWindMessage("Sample wind CSV copied to clipboard.");
    } catch {
      setWindMessage("Clipboard copy failed. You can copy the sample from the source code.");
    }
  }, []);

  const stormSummary = useMemo(() => {
    const pointCount = stormTracks.reduce((sum, track) => sum + track.points.length, 0);
    if (stormTracks.length === 0) return "no storm data";
    return `${stormTracks.length} storm track(s), ${pointCount} point(s)`;
  }, [stormTracks]);

  const windSummary = useMemo(() => windVectors.length > 0 ? `${windVectors.length} imported wind vector(s)` : "procedural wind field", [windVectors]);

  const seamNote = useMemo(() => {
    const lon = wrapLon(offset.lon);
    const nearSeam = Math.abs(Math.abs(lon) - 180) < 28 || Math.abs(lon) < 28;
    const nearPole = Math.abs(offset.lat) > 65;

    if (nearPole) return "Pole crossing active: latitude reflection flips longitude by 180 degrees.";
    if (nearSeam) return "Seam behavior visible: 2D features split at the date line while the globe stays continuous.";
    return "Use arrow keys to translate data. Move east or west to cross the red seam lines.";
  }, [offset]);

  return (
    <div className="app">
      <style>{`
        :root {
          --uc-red: #d6001c;
          --uc-red-dark: #8b0013;
          --uc-gold: #ffcc00;
          --uc-black: #111111;
          --uc-panel: rgba(255, 255, 255, 0.96);
          --uc-muted: #5b5b5b;
          --uc-map: #07111f;
        }

        * { box-sizing: border-box; }

        html,
        body,
        #root {
          width: 100%;
          height: 100%;
          margin: 0;
          padding: 0;
          max-width: none;
          text-align: left;
        }

        body {
          margin: 0;
          background: #f5f2ed;
          color: var(--uc-black);
          font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          overflow: hidden;
          display: block;
          place-items: initial;
        }

        .app {
          position: fixed;
          inset: 0;
          width: 100%;
          height: 100%;
          padding: 0;
          background:
            radial-gradient(circle at 12% 0%, rgba(214, 0, 28, 0.20) 0, transparent 26%),
            radial-gradient(circle at 92% 8%, rgba(255, 204, 0, 0.18) 0, transparent 24%),
            linear-gradient(135deg, #fffaf0 0%, #f3eee6 44%, #ece7df 100%);
        }

        .container {
          width: 100%;
          height: 100%;
          max-width: none;
          margin: 0;
          display: grid;
          grid-template-rows: 78px minmax(0, 1fr);
          gap: 0;
        }

        .topBar {
          position: relative;
          z-index: 20;
          display: grid;
          grid-template-columns: minmax(0, 1fr) auto;
          align-items: center;
          gap: 18px;
          padding: 12px 18px 12px 26px;
          background: linear-gradient(90deg, #ffffff 0%, #fff7d7 62%, #ffffff 100%);
          border-bottom: 3px solid #1d1d1d;
          box-shadow:
            0 5px 0 rgba(214, 0, 28, 0.52),
            0 12px 25px rgba(17, 17, 17, 0.18);
        }

        .topBar::before {
          content: '';
          position: absolute;
          left: 0;
          top: 0;
          bottom: 0;
          width: 10px;
          background: linear-gradient(180deg, var(--uc-red), var(--uc-gold));
        }

        .brandLine {
          display: flex;
          align-items: center;
          gap: 14px;
          min-width: 0;
        }

        .brandMark {
          display: inline-grid;
          place-items: center;
          width: 42px;
          height: 42px;
          border: 2px solid #1d1d1d;
          border-radius: 0;
          background: linear-gradient(180deg, #ef2339 0%, var(--uc-red) 50%, var(--uc-red-dark) 100%);
          color: white;
          box-shadow:
            4px 4px 0 #57000c,
            inset 0 1px 0 rgba(255, 255, 255, 0.38);
          font-weight: 950;
          letter-spacing: -0.04em;
          flex: 0 0 auto;
        }

        .titleBlock {
          min-width: 0;
        }

        .brandKicker {
          color: var(--uc-red-dark);
          font-size: 12px;
          font-weight: 900;
          letter-spacing: 0.12em;
          text-transform: uppercase;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
        }

        .title {
          margin: 2px 0 0;
          color: var(--uc-black);
          font-size: clamp(18px, 2.2vw, 26px);
          line-height: 1.05;
          letter-spacing: -0.035em;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
        }

        .views {
          display: grid;
          grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
          width: 100%;
          height: calc(100vh - 78px);
          gap: 0;
          padding: 16px;
        }

        .viewCard {
          min-width: 0;
          min-height: 0;
          display: grid;
          grid-template-rows: 42px minmax(0, 1fr);
          overflow: hidden;
          background: #ffffff;
          border: 3px solid #1d1d1d;
          border-radius: 0;
          box-shadow:
            8px 8px 0 rgba(214, 0, 28, 0.55),
            15px 15px 0 rgba(17, 17, 17, 0.28),
            0 22px 55px rgba(17, 17, 17, 0.16),
            inset 0 1px 0 rgba(255, 255, 255, 0.92);
        }

        .viewCard:first-child {
          margin-right: 10px;
        }

        .viewCard:last-child {
          margin-left: 10px;
        }

        .viewTitle {
          position: relative;
          padding: 12px 16px 12px 30px;
          border-bottom: 3px solid #1d1d1d;
          background: linear-gradient(90deg, #ffffff 0%, #fff7d7 100%);
          color: var(--uc-black);
          font-size: 14px;
          font-weight: 900;
          letter-spacing: 0.01em;
        }

        .viewTitle::before {
          content: '';
          position: absolute;
          left: 14px;
          top: 8px;
          width: 7px;
          height: 24px;
          border-radius: 0;
          background: var(--uc-red);
          box-shadow: 3px 3px 0 rgba(17, 17, 17, 0.25);
        }

        .canvasBox {
          min-height: 0;
          height: 100%;
          background: var(--uc-map);
        }

        .settingsButton {
          z-index: 45;
        }

        .button, .fileButton {
          position: relative;
          border: 2px solid #1d1d1d;
          background: linear-gradient(180deg, #ffffff 0%, #f1efe9 48%, #d7d2c8 100%);
          color: var(--uc-black);
          padding: 11px 15px;
          border-radius: 0;
          cursor: pointer;
          font-weight: 850;
          letter-spacing: 0.01em;
          transition: transform 90ms ease, background 120ms ease, box-shadow 120ms ease;
          display: inline-flex;
          align-items: center;
          justify-content: center;
          font-size: 14px;
          box-shadow:
            0 5px 0 #8f887a,
            0 10px 18px rgba(17, 17, 17, 0.18),
            inset 0 1px 0 rgba(255, 255, 255, 0.95);
        }

        .button:hover, .fileButton:hover {
          transform: translateY(-2px);
          background: linear-gradient(180deg, #ffffff 0%, #fff6cf 45%, #e6d88a 100%);
          box-shadow:
            0 7px 0 #8f887a,
            0 14px 22px rgba(214, 0, 28, 0.18),
            inset 0 1px 0 rgba(255, 255, 255, 0.98);
        }

        .button:active, .fileButton:active {
          transform: translateY(4px);
          box-shadow:
            0 1px 0 #8f887a,
            0 4px 8px rgba(17, 17, 17, 0.20),
            inset 0 2px 4px rgba(17, 17, 17, 0.18);
        }

        .button.active {
          background: linear-gradient(180deg, #ef2339 0%, var(--uc-red) 48%, var(--uc-red-dark) 100%);
          border-color: #1d1d1d;
          color: white;
          box-shadow:
            0 5px 0 #57000c,
            0 12px 22px rgba(214, 0, 28, 0.28),
            inset 0 1px 0 rgba(255, 255, 255, 0.35);
        }

        .fileButton {
          background: linear-gradient(180deg, #333333 0%, #151515 54%, #000000 100%);
          color: white;
          box-shadow:
            0 5px 0 #65000e,
            0 10px 18px rgba(17, 17, 17, 0.22),
            inset 0 1px 0 rgba(255, 255, 255, 0.22);
        }

        .drawerBackdrop {
          position: fixed;
          inset: 0;
          z-index: 35;
          background: rgba(17, 17, 17, 0.44);
          opacity: 0;
          pointer-events: none;
          transition: opacity 180ms ease;
        }

        .drawerBackdrop.open {
          opacity: 1;
          pointer-events: auto;
        }

        .settingsDrawer {
          position: fixed;
          z-index: 40;
          top: 0;
          right: 0;
          width: min(470px, 94vw);
          height: 100vh;
          padding: 20px;
          overflow-y: auto;
          background: linear-gradient(180deg, #ffffff 0%, #fff7d7 100%);
          border-left: 4px solid #1d1d1d;
          box-shadow:
            -8px 0 0 rgba(214, 0, 28, 0.72),
            -18px 0 45px rgba(17, 17, 17, 0.34),
            inset 1px 0 0 rgba(255, 255, 255, 0.9);
          transform: translateX(calc(100% + 40px));
          transition: transform 220ms ease;
        }

        .settingsDrawer.open {
          transform: translateX(0);
        }

        .drawerHeader {
          display: flex;
          align-items: flex-start;
          justify-content: space-between;
          gap: 12px;
          padding-bottom: 16px;
          border-bottom: 3px solid #1d1d1d;
          margin-bottom: 16px;
        }

        .drawerTitle {
          margin: 0;
          font-size: 24px;
          letter-spacing: -0.035em;
        }

        .drawerSubtitle, .text {
          color: var(--uc-muted);
          line-height: 1.6;
          font-size: 14px;
        }

        .statusGrid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 9px;
          margin-bottom: 16px;
        }

        .label {
          background: linear-gradient(180deg, #2b2b2b 0%, #111111 100%);
          border: 2px solid #1d1d1d;
          color: #fff7d6;
          border-radius: 0;
          padding: 10px 12px;
          font-size: 12px;
          font-weight: 750;
          min-height: 42px;
          display: inline-flex;
          align-items: center;
          box-shadow:
            4px 4px 0 rgba(214, 0, 28, 0.58),
            inset 0 1px 0 rgba(255, 255, 255, 0.18);
        }

        .controlSection {
          background: linear-gradient(180deg, #ffffff 0%, #f3efe6 100%);
          border: 2px solid #1d1d1d;
          border-radius: 0;
          padding: 14px;
          margin-bottom: 14px;
          box-shadow:
            5px 5px 0 rgba(143, 136, 122, 0.85),
            10px 10px 18px rgba(17, 17, 17, 0.14),
            inset 0 1px 0 rgba(255, 255, 255, 0.9);
        }

        .controlSection h2 {
          margin: 0 0 8px;
          font-size: 18px;
          letter-spacing: -0.02em;
        }

        .buttons {
          display: flex;
          flex-wrap: wrap;
          gap: 9px;
          margin: 14px 0;
        }

        .moveGrid {
          display: grid;
          grid-template-columns: repeat(4, minmax(0, 1fr));
          gap: 9px;
          margin-bottom: 12px;
        }

        .note {
          background: #ffffff;
          border: 2px solid #1d1d1d;
          border-left: 8px solid var(--uc-gold);
          border-radius: 0;
          padding: 12px;
          color: var(--uc-muted);
          line-height: 1.6;
          font-size: 14px;
          box-shadow: 4px 4px 0 rgba(17, 17, 17, 0.18);
        }

        .format {
          background: linear-gradient(180deg, #202020 0%, #0c0c0c 100%);
          color: #fff7d6;
          border: 2px solid #1d1d1d;
          border-radius: 0;
          padding: 12px;
          overflow-x: auto;
          font-size: 12px;
          line-height: 1.5;
          box-shadow:
            5px 5px 0 rgba(214, 0, 28, 0.45),
            inset 0 1px 0 rgba(255, 255, 255, 0.08);
        }

        input[type='file'] { display: none; }

        @media (max-width: 980px) {
          body { overflow: auto; }
          .app { min-height: 100vh; height: auto; }
          .container { height: auto; min-height: 100vh; grid-template-rows: auto auto; }
          .topBar { grid-template-columns: 1fr; }
          .views { grid-template-columns: 1fr; height: auto; }
          .viewCard { height: 48vh; margin: 0 0 18px 0 !important; }
        }

        @media (max-width: 640px) {
          .brandKicker { white-space: normal; }
          .title { white-space: normal; }
          .statusGrid { grid-template-columns: 1fr; }
          .moveGrid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        }
      `}</style>

      <main className="container">
        <header className="topBar">
          <div className="brandLine">
            <span className="brandMark">FS</span>
            <div className="titleBlock">
              <div className="brandKicker">Prof. Faramarz Samavati · University of Calgary · Department of Computer Science</div>
              <h1 className="title">Globe and Mercator Map Atmospheric Visualization</h1>
            </div>
          </div>
          <button className="button settingsButton" type="button" onClick={() => setControlsOpen(true)}>Open controls</button>
        </header>

        <section className="views">
          <div className="viewCard">
            <div className="viewTitle">Globe view: spherical continuity</div>
            <div className="canvasBox">
              <Canvas camera={{ position: [0, 0, 6.2], fov: 45 }} gl={{ antialias: true, alpha: true }}>
                <color attach="background" args={["#020617"]} />
                <GlobeScene mode={mode} offset={offset} paused={paused} stormTracks={stormTracks} windPaths={windPaths} />
              </Canvas>
            </div>
          </div>

          <div className="viewCard">
            <div className="viewTitle">Flat map view: Mercator projection and seam behavior</div>
            <div className="canvasBox">
              <Canvas orthographic camera={{ position: [0, 0, 8], zoom: 92 }} gl={{ antialias: true, alpha: true }}>
                <color attach="background" args={["#020617"]} />
                <MapScene mode={mode} offset={offset} paused={paused} stormTracks={stormTracks} windPaths={windPaths} />
              </Canvas>
            </div>
          </div>
        </section>

        <div className={controlsOpen ? "drawerBackdrop open" : "drawerBackdrop"} onClick={() => setControlsOpen(false)} />

        <aside className={controlsOpen ? "settingsDrawer open" : "settingsDrawer"} aria-hidden={!controlsOpen}>
          <div className="drawerHeader">
            <div>
              <h2 className="drawerTitle">Controls and data</h2>
              <div className="drawerSubtitle">Open only when changes are needed. The two visualizations remain the main workspace.</div>
            </div>
            <button className="button" type="button" onClick={() => setControlsOpen(false)}>Close</button>
          </div>

          <div className="statusGrid">
            <Label>longitude offset {wrapLon(offset.lon).toFixed(0)}°</Label>
            <Label>latitude offset {offset.lat.toFixed(0)}°</Label>
            <Label>{paused ? "animation paused" : "animation running"}</Label>
            <Label>{windSummary}</Label>
            <Label>{stormSummary}</Label>
          </div>

          <section className="controlSection">
            <h2>Layer selection</h2>
            <p className="text">{seamNote}</p>
            <div className="buttons">
              <ButtonLike active={mode === "wind"} onClick={() => setMode("wind")}>Wind texture</ButtonLike>
              <ButtonLike active={mode === "storm"} onClick={() => setMode("storm")}>Storm data</ButtonLike>
              <ButtonLike active={mode === "flows"} onClick={() => setMode("flows")}>Cyclone model</ButtonLike>
              <ButtonLike active={mode === "primitives"} onClick={() => setMode("primitives")}>Geometric primitives</ButtonLike>
              <ButtonLike active={mode === "all"} onClick={() => setMode("all")}>All layers</ButtonLike>
              <ButtonLike onClick={() => setPaused((value) => !value)}>{paused ? "Resume" : "Pause"}</ButtonLike>
              <ButtonLike onClick={() => setOffset({ lat: 0, lon: 0 })}>Reset</ButtonLike>
            </div>
          </section>

          <section className="controlSection">
            <h2>Shift data</h2>
            <div className="moveGrid">
              <ButtonLike onClick={() => nudge(0, -10)}>West</ButtonLike>
              <ButtonLike onClick={() => nudge(0, 10)}>East</ButtonLike>
              <ButtonLike onClick={() => nudge(10, 0)}>North</ButtonLike>
              <ButtonLike onClick={() => nudge(-10, 0)}>South</ButtonLike>
            </div>
            <div className="note">Keyboard: arrows shift all data, 1 wind, 2 storm, 3 cyclone, 4 primitives, 5 all, P pause, R reset.</div>
          </section>

          <section className="controlSection">
            <h2>Load actual wind vectors</h2>
            <p className="text">Import CSV with lat, lon, u, and v. The app traces dense streamlines from the vector field and colors them by speed.</p>
            <div className="buttons">
              <label className="fileButton">Choose wind CSV<input type="file" accept=".csv,text/csv" onChange={onWindFileChange} /></label>
              <ButtonLike onClick={() => loadWindText(sampleWindCsv)}>Load sample vectors</ButtonLike>
              <ButtonLike onClick={() => setWindVectors([])}>Use procedural wind</ButtonLike>
              <ButtonLike onClick={copySampleWindCsv}>Copy wind CSV</ButtonLike>
            </div>
            <div className="note">{windMessage}</div>
          </section>

          <section className="controlSection">
            <h2>Load actual storm track</h2>
            <p className="text">Import CSV or JSON with id, lat, lon, and optional name, time, wind, and pressure.</p>
            <div className="buttons">
              <label className="fileButton">Choose storm CSV or JSON<input type="file" accept=".csv,.json,application/json,text/csv" onChange={onStormFileChange} /></label>
              <ButtonLike onClick={() => loadStormText(sampleStormCsv, "sample.csv")}>Reload sample storm</ButtonLike>
              <ButtonLike onClick={copySampleCsv}>Copy storm CSV</ButtonLike>
            </div>
            <div className="note">{dataMessage}</div>
          </section>

          <section className="controlSection">
            <h2>Input formats</h2>
            <p className="text">Wind vectors use eastward and northward components. Positive u moves east. Positive v moves north.</p>
            <pre className="format">{`lat,lon,u,v
45,-80,25,5
50,-60,20,-10
35,-40,10,-20`}</pre>
            <p className="text">Storm tracks use one row per observation. Multiple storms can be stored in the same file by using different id values.</p>
            <pre className="format">{`id,name,time,lat,lon,wind,pressure
AL012024,Example Storm,2024-08-01T00:00:00Z,12.0,-42.0,35,1006
AL012024,Example Storm,2024-08-01T06:00:00Z,12.6,-44.8,40,1002`}</pre>
          </section>
        </aside>
      </main>
    </div>
  );
}
