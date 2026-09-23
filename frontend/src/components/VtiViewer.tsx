/**
 * VtiViewer — the simulated tumour evolution rendered in-browser.
 *
 * Consumes the base64 VTI the twin endpoint returns (the same file ParaView
 * opens), decodes the Float32 point scalars (drug_A, drug_B, sensitive,
 * resistant, plus the derived tissue_class / diffusion_mm2_per_day /
 * proliferative / vessels maps) and the tumour mask, and renders them as an
 * instanced voxel field with OrbitControls. Field selection, isovalue, and
 * the cutaway re-threshold the instanced set client-side — no extra engine
 * calls.
 *
 * Axis convention: the exporter writes arrays Fortran-order (axis 0 fastest)
 * and declares the extent as (0 sx-1)(0 sy-1)(0 sz-1) for engine shape
 * (sx, sy, sz) — VTK x/y/z are the array axes in order.
 */
import React, { useMemo, useState, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';

interface VtiData {
  dims: [number, number, number]; // array shape (axis0, axis1, axis2)
  fields: Record<string, Float32Array>;
  mask: Uint8Array;
}

function decodeBase64(b64: string): Uint8Array {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

function parseVtiFromBytes(bytes: Uint8Array): VtiData {
  let text = '';
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    text += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  const doc = new DOMParser().parseFromString(text, 'text/xml');
  const img = doc.querySelector('ImageData');
  if (!img) throw new Error('not a VTK ImageData file');
  const extent = (img.getAttribute('WholeExtent') || '').split(/\s+/).map(Number);
  if (extent.length !== 6) throw new Error('malformed WholeExtent');
  // Exporter writes engine arrays (axis0 = x, NIfTI dim order) with extent
  // (0 sx-1)(0 sy-1)(0 sz-1): VTK x/y/z are the array axes in order.
  const dims: [number, number, number] = [extent[1] + 1, extent[3] + 1, extent[5] + 1];
  const fields: Record<string, Float32Array> = {};
  let mask = new Uint8Array(0);
  doc.querySelectorAll('DataArray').forEach(el => {
    const name = el.getAttribute('Name') || '';
    const type = el.getAttribute('type');
    const raw = decodeBase64((el.textContent || '').trim());
    if (type === 'Float32') fields[name] = new Float32Array(raw.buffer.slice(0));
    else if (type === 'UInt8' && name === 'tumour_mask') mask = new Uint8Array(raw);
  });
  if (!fields['sensitive'] || !fields['resistant']) {
    throw new Error('VTI missing sensitive/resistant fields');
  }
  return { dims, fields, mask };
}

const FIELD_COLORS: Record<string, string> = {
  sensitive: '#06b6d4',
  resistant: '#f43f5e',
  drug_A: '#3b82f6',
  drug_B: '#8b5cf6',
  tissue_class: '#f59e0b',
  diffusion_mm2_per_day: '#34d399',
  proliferative: '#eab308',
  vessels: '#fb923c',
};

/** Renderable fields in chip order; tissue/diffusion maps only exist in
 * VTIs exported with include_tissue=True. */
const FIELDS = [
  'sensitive', 'resistant', 'drug_A', 'drug_B',
  'tissue_class', 'diffusion_mm2_per_day', 'proliferative', 'vessels',
];

function ramp(t: number, hex: string): THREE.Color {
  // low -> dim, high -> saturated field color
  const c = new THREE.Color(hex);
  return new THREE.Color(0.05, 0.07, 0.10).lerp(c, 0.25 + 0.75 * t);
}

interface Props { vtiBase64: string | null }

export default function VtiViewer({ vtiBase64 }: Props) {
  const [field, setField] = useState<string>('sensitive');
  const [iso, setIso] = useState(0.0);
  const [cut, setCut] = useState(0);      // 0..1 fraction of axis0 hidden from the front
  const [overlayR, setOverlayR] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const data = useMemo(() => {
    if (!vtiBase64) return null;
    try { setError(null); return parseVtiFromBytes(decodeBase64(vtiBase64)); }
    catch (e: any) { setError(e?.message || 'VTI parse failed'); return null; }
  }, [vtiBase64]);

  const [tumourCount, maxVal, maxR] = useMemo(() => {
    if (!data) return [0, 1, 1];
    const f = data.fields[field];
    let n = 0, mx = 0;
    for (let i = 0; i < data.mask.length; i++) {
      if (data.mask[i]) { n++; if (f[i] > mx) mx = f[i]; }
    }
    const r = data.fields['resistant'];
    let mr = 0;
    for (let i = 0; i < r.length; i++) if (r[i] > mr) mr = r[i];
    return [n, mx || 1, mr || 1];
  }, [data, field]);

  useEffect(() => { setIso(0); }, [field]);

  if (error) {
    return <div className="seq-empty">3D viewer: {error}</div>;
  }
  if (!data) return <div className="seq-empty">No VTI payload — run a plan first.</div>;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8, height: '100%' }}>
      <div className="vti-controls">
        <div className="vti-ctrl-group">
          {FIELDS.filter(f => data.fields[f]).map(f => (
            <button key={f} className={`chip ${field === f ? 'active' : ''}`} onClick={() => setField(f)}>
              <span className="chip-dot" style={{ background: FIELD_COLORS[f] }} />{f}
            </button>
          ))}
        </div>
        <label className="vti-slider">
          isovalue ≥ <b>{(iso * maxVal).toFixed(2)}</b>
          <input type="range" min={0} max={1} step={0.01} value={iso}
            onChange={e => setIso(parseFloat(e.target.value))} disabled={field === 'sensitive' && iso === 0} />
        </label>
        <label className="vti-slider">
          cutaway x <b>{(cut * 100).toFixed(0)}%</b>
          <input type="range" min={0} max={0.9} step={0.05} value={cut}
            onChange={e => setCut(parseFloat(e.target.value))} />
        </label>
        <label className="vti-check">
          <input type="checkbox" checked={overlayR} onChange={e => setOverlayR(e.target.checked)} />
          resistant overlay
        </label>
      </div>
      <div style={{ flex: 1, minHeight: 380 }}>
        <Canvas camera={{ position: [1.6, 1.1, 1.6], fov: 45 }} dpr={[1, 2]}>
          <ambientLight intensity={0.7} />
          <directionalLight position={[5, 8, 5]} intensity={1.1} />
          <VoxelField data={data} field={field} iso={iso} cut={cut} overlayR={overlayR} maxVal={maxVal} maxR={maxR} />
          <OrbitControls enableDamping dampingFactor={0.08} />
        </Canvas>
      </div>
      <div className="vti-footer mono">
        {tumourCount.toLocaleString()} tumour voxels · final-day fields on the uploaded grid · same file ParaView opens
      </div>
    </div>
  );
}

const MAX_INSTANCES = 150_000;

function VoxelField({ data, field, iso, cut, overlayR, maxVal, maxR }: {
  data: VtiData;
  field: string;
  iso: number;
  cut: number;
  overlayR: boolean;
  maxVal: number;
  maxR: number;
}) {
  const { geom, count, overlay } = useMemo(() => {
    const [s0, s1, s2] = data.dims;
    const f = data.fields[field];
    const r = data.fields['resistant'];
    const cutIdx = Math.floor(cut * s0);
    const main: number[] = [];
    const over: number[] = [];
    for (let a = 0; a < s0; a++) {
      if (a < cutIdx) continue;
      for (let b = 0; b < s1; b++) {
        const base = a + b * s0;
        for (let c = 0; c < s2; c++) {
          const idx = base + c * s0 * s1;
          if (!data.mask[idx]) continue;
          if (f[idx] >= iso * maxVal) main.push(idx);
          else if (overlayR && r[idx] >= 0.05 * maxR) over.push(idx);
        }
      }
    }
    const stride = Math.max(1, Math.ceil(main.length / MAX_INSTANCES));
    const overStride = Math.max(1, Math.ceil(over.length / MAX_INSTANCES));
    const scale = 60 / Math.max(s0, s1, s2);
    const positions = new Float32Array(Math.floor(main.length / stride) * 3);
    const colors = new Float32Array(Math.floor(main.length / stride) * 3);
    const col = new THREE.Color();
    const hex = FIELD_COLORS[field] || '#06b6d4';
    let w = 0;
    for (let i = 0; i < main.length; i += stride) {
      const idx = main[i];
      const a = idx % s0;
      const b = Math.floor(idx / s0) % s1;
      const c = Math.floor(idx / (s0 * s1));
      positions[w * 3] = (a - s0 / 2) * scale;
      positions[w * 3 + 1] = (b - s1 / 2) * scale;
      positions[w * 3 + 2] = (c - s2 / 2) * scale;
      col.copy(ramp(f[idx] / maxVal, hex));
      colors[w * 3] = col.r; colors[w * 3 + 1] = col.g; colors[w * 3 + 2] = col.b;
      w++;
    }
    const overPos = new Float32Array(Math.floor(over.length / overStride) * 3);
    w = 0;
    for (let i = 0; i < over.length; i += overStride) {
      const idx = over[i];
      const a = idx % s0;
      const b = Math.floor(idx / s0) % s1;
      const c = Math.floor(idx / (s0 * s1));
      overPos[w * 3] = (a - s0 / 2) * scale;
      overPos[w * 3 + 1] = (b - s1 / 2) * scale;
      overPos[w * 3 + 2] = (c - s2 / 2) * scale;
      w++;
    }
    return {
      geom: { positions, colors, voxel: scale * 0.9 },
      count: Math.floor(main.length / stride),
      overlay: { positions: overPos, count: Math.floor(over.length / overStride) },
    };
  }, [data, field, iso, cut, overlayR, maxVal, maxR]);

  return (
    <group>
      {count > 0 && (
        <Instances positions={geom.positions} colors={geom.colors} count={count} size={geom.voxel} />
      )}
      {overlay.count > 0 && (
        <Instances positions={overlay.positions} colors={null} count={overlay.count} size={geom.voxel * 1.15} color="#f43f5e" />
      )}
    </group>
  );
}

function Instances({ positions, colors, count, size, color }: {
  positions: Float32Array;
  colors: Float32Array | null;
  count: number;
  size: number;
  color?: string;
}) {
  const ref = useMemo(() => {
    const g = new THREE.BoxGeometry(size, size, size);
    const m = new THREE.InstancedMesh(g, new THREE.MeshBasicMaterial({}), count);
    const mat = new THREE.Object3D();
    for (let i = 0; i < count; i++) {
      mat.position.set(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
      mat.updateMatrix();
      m.setMatrixAt(i, mat.matrix);
      if (colors) m.setColorAt(i, new THREE.Color(colors[i * 3], colors[i * 3 + 1], colors[i * 3 + 2]));
    }
    if (!colors && color) {
      (m.material as THREE.MeshBasicMaterial).color = new THREE.Color(color);
    }
    m.instanceMatrix.needsUpdate = true;
    if (m.instanceColor) m.instanceColor.needsUpdate = true;
    return m;
  }, [positions, colors, count, size, color]);
  return <primitive object={ref} />;
}
