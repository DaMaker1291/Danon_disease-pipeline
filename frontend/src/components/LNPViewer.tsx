import React, { useRef, useMemo, useState } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Environment, Line, ContactShadows } from '@react-three/drei';
import { EffectComposer, Bloom, N8AO, ToneMapping, Vignette } from '@react-three/postprocessing';
import * as THREE from 'three';
import ErrorBoundary from './ErrorBoundary';
import { useArtifact, artifactStamp } from '../utils/artifacts';
import type { LNPArtifactData } from '../types/artifacts';

/**
 * Lipid composition and particle size come from `artifacts/lnp.json`, which
 * `danon.lnp_generator.DanonLNPGenerator` produced by scoring real candidates.
 * The previous version of this file hardcoded a composition (DLin-MC3-DMA at
 * 50%, 38.5% cholesterol, …) that no module in this repository generates.
 */
interface LipidBand { name: string; color: string; frac: number }

interface Lipid {
  dir: THREE.Vector3;
  color: THREE.Color;
  headOuter: THREE.Vector3;
  tailInner: THREE.Vector3;
  phase: number;
}

function fibonacciSphere(n: number): THREE.Vector3[] {
  const pts: THREE.Vector3[] = [];
  const golden = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < n; i++) {
    const y = 1 - (i / (n - 1)) * 2;
    const r = Math.sqrt(1 - y * y);
    const theta = golden * i;
    pts.push(new THREE.Vector3(Math.cos(theta) * r, y, Math.sin(theta) * r));
  }
  return pts;
}

function LipidBilayer({ radius, n2p, bands }: { radius: number; n2p: number; bands: LipidBand[] }) {
  const groupRef = useRef<THREE.Group>(null!);
  const nLipids = 320;
  const compactness = 0.85 + (n2p / 12) * 0.3;

  const lipids: Lipid[] = useMemo(() => {
    const dirs = fibonacciSphere(nLipids);
    let cum = 0;
    const scaled = bands.map(t => { const b = { ...t, from: cum, to: cum + t.frac }; cum += t.frac; return b; });
    return dirs.map((dir, i) => {
      const f = i / nLipids;
      const band = scaled.find(b => f >= b.from && f <= b.to) || scaled[0];
      return {
        dir,
        color: new THREE.Color(band.color),
        headOuter: dir.clone().multiplyScalar((radius + 0.32) * compactness),
        tailInner: dir.clone().multiplyScalar((radius - 0.18) * compactness),
        phase: Math.random() * Math.PI * 2,
      };
    });
  }, [radius, compactness, bands]);

  useFrame((_, dt) => {
    groupRef.current.rotation.y += dt * 0.09;
    groupRef.current.rotation.x = Math.sin(Date.now() * 0.0002) * 0.1;
  });

  return (
    <group ref={groupRef}>
      {lipids.map((lp, i) => {
        const mid = lp.headOuter.clone().add(lp.tailInner).multiplyScalar(0.5);
        const seg = lp.headOuter.clone().sub(lp.tailInner);
        const len = seg.length();
        const quat = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), seg.clone().normalize());
        return (
          <group key={i}>
            {/* head group */}
            <mesh position={lp.headOuter}>
              <sphereGeometry args={[0.11, 12, 12]} />
              <meshPhysicalMaterial color={lp.color} roughness={0.25} metalness={0.15} clearcoat={0.4} emissive={lp.color} emissiveIntensity={0.12} />
            </mesh>
            {/* lipid tail */}
            <mesh position={mid} quaternion={quat}>
              <cylinderGeometry args={[0.025, 0.04, len, 6]} />
              <meshStandardMaterial color={lp.color} roughness={0.6} transparent opacity={0.55} />
            </mesh>
          </group>
        );
      })}
    </group>
  );
}

function MRNACore({ radius }: { radius: number }) {
  const ref = useRef<THREE.Group>(null!);
  const points = useMemo(() => {
    const pts: [number, number, number][] = [];
    const N = 160;
    for (let i = 0; i < N; i++) {
      const t = i / N;
      // random-walk-ish coiled strand within the core
      const a = t * Math.PI * 14;
      const r = radius * (0.15 + 0.6 * Math.abs(Math.sin(t * Math.PI * 3)));
      pts.push([
        r * Math.cos(a),
        (t - 0.5) * radius * 1.2,
        r * Math.sin(a),
      ]);
    }
    return pts;
  }, [radius]);

  useFrame((_, dt) => { ref.current.rotation.y += dt * 0.25; ref.current.rotation.z += dt * 0.05; });

  return (
    <group ref={ref}>
      <Line points={points} color="#fbbf24" lineWidth={3} transparent opacity={0.85} />
      {points.filter((_, i) => i % 6 === 0).map((p, i) => (
        <mesh key={i} position={p}>
          <sphereGeometry args={[0.05, 8, 8]} />
          <meshStandardMaterial color="#f59e0b" emissive="#f59e0b" emissiveIntensity={0.6} />
        </mesh>
      ))}
    </group>
  );
}

function Scene({ n2p, bands, radius }: { n2p: number; bands: LipidBand[]; radius: number }) {
  return (
    <>
      <color attach="background" args={['#05080f']} />
      <fog attach="fog" args={['#05080f', 7, 16]} />
      <ambientLight intensity={0.3} />
      <directionalLight position={[5, 7, 5]} intensity={1.4} castShadow shadow-mapSize={[2048, 2048]} />
      <directionalLight position={[-5, -2, -5]} intensity={0.5} color="#3b82f6" />
      <pointLight position={[0, 0, 0]} intensity={1.0} color="#fbbf24" distance={4} />
      <spotLight position={[4, 6, -4]} angle={0.5} penumbra={0.8} intensity={1.0} color="#a78bfa" />
      <Environment preset="warehouse" />

      {/* translucent aqueous core boundary */}
      <mesh>
        <sphereGeometry args={[radius - 0.2, 48, 48]} />
        <meshPhysicalMaterial color="#1e3a5f" roughness={0.1} transmission={0.9} thickness={1} transparent opacity={0.18} ior={1.33} />
      </mesh>

      <MRNACore radius={radius - 0.35} />
      <LipidBilayer radius={radius} n2p={n2p} bands={bands} />

      <ContactShadows position={[0, -2.6, 0]} opacity={0.4} scale={10} blur={2.5} far={4} />
      <OrbitControls enablePan={false} minDistance={3} maxDistance={10} enableDamping autoRotate autoRotateSpeed={0.35} />
      <EffectComposer enableNormalPass>
        <N8AO aoRadius={1.0} intensity={1.6} />
        <Bloom luminanceThreshold={0.5} intensity={0.7} mipmapBlur />
        <ToneMapping />
        <Vignette offset={0.15} darkness={0.7} />
      </EffectComposer>
    </>
  );
}

export default function LNPViewer() {
  const [n2p, setN2p] = useState(6);
  const { artifact, loading, error } = useArtifact<LNPArtifactData>('lnp');

  const bands: LipidBand[] = (artifact?.data.composition ?? []).map(c => ({
    name: c.name, color: c.color, frac: c.frac,
  }));
  const best = artifact?.data.best;
  const summary = artifact?.data.summary;

  if (loading) return <div className="panel-title">Lipid Nanoparticle — loading formulation artifact…</div>;
  if (error || !artifact || !best || !summary) {
    return <div className="panel-title" style={{ color: 'var(--warning)' }}>LNP artifact unavailable: {error}</div>;
  }

  // The sphere scales with the PREDICTED particle size from the artifact, so the
  // visual and the number cannot disagree about which formulation is shown.
  const radius = Math.max(0.9, Math.min(2.4, best.particle_size_nm / 60));

  return (
    <>
      <div className="panel-title" style={{ marginBottom: 6 }}>
        Lipid Nanoparticle — best of {summary.n_candidates} scored candidates ({best.particle_size_nm.toFixed(1)} nm predicted)
      </div>
      <div className="canvas-container" style={{ minHeight: 400 }}>
        <ErrorBoundary label="LNPViewer">
          <Canvas shadows dpr={[1, 2]} camera={{ position: [4, 2.5, 4], fov: 40 }} gl={{ antialias: true, toneMapping: THREE.ACESFilmicToneMapping }}>
            <Scene n2p={n2p} bands={bands} radius={radius} />
          </Canvas>
        </ErrorBoundary>
      </div>
      <div style={{ marginTop: 10, display: 'flex', gap: 20, alignItems: 'flex-end', flexWrap: 'wrap' }}>
        <div className="input-group" style={{ flex: 1, minWidth: 160 }}>
          <label>N/P Ratio: {n2p} <span style={{ color: 'var(--text3)' }}>(visual only — compaction of the drawn core)</span></label>
          <input type="range" min={2} max={12} step={1} value={n2p} onChange={e => setN2p(+e.target.value)} />
        </div>
      </div>

      <div style={{ marginTop: 10, display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 12 }}>
        <div className="stat-chip"><span>ionizable lipid</span><strong>{best.ionizable_lipid}</strong></div>
        <div className="stat-chip"><span>helper lipid</span><strong>{best.helper_lipid}</strong></div>
        <div className="stat-chip"><span>PEG lipid</span><strong>{best.peg_lipid}</strong></div>
        <div className="stat-chip"><span>pKa</span><strong>{best.pka.toFixed(1)}</strong></div>
        <div className="stat-chip ok"><span>fitness</span><strong>{best.fitness.toFixed(3)}</strong></div>
        <div className="stat-chip"><span>size spread (mean ± sd)</span><strong>{summary.mean_size_nm.toFixed(1)} ± {summary.sd_size_nm.toFixed(1)} nm</strong></div>
      </div>

      <div style={{ marginTop: 8, display: 'flex', gap: 16, fontSize: 10, color: '#64748b', flexWrap: 'wrap' }}>
        {bands.map(t => (
          <span key={t.name}><span style={{ color: t.color }}>■</span> {t.name} ({(t.frac * 100).toFixed(1)} mol%)</span>
        ))}
        <span style={{ color: '#fbbf24' }}>■ mRNA payload (schematic)</span>
      </div>

      <div style={{ marginTop: 6, color: 'var(--text3)', fontSize: 10.5, lineHeight: 1.6 }}>
        {artifactStamp(artifact)}
        <br />Predicted size range across the cohort: {summary.size_range_nm[0].toFixed(1)}–{summary.size_range_nm[1].toFixed(1)} nm ·
        ionizable lipids seen: {artifact.data.ionizable_lipids_seen.join(', ')}
      </div>
    </>
  );
}
