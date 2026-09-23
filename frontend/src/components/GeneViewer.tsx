import React, { useRef, useMemo, useState, useCallback, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Environment, Float } from '@react-three/drei';
import { EffectComposer, Bloom, ToneMapping } from '@react-three/postprocessing';
import * as THREE from 'three';
import ErrorBoundary from './ErrorBoundary';
import { useArtifact, artifactStamp } from '../utils/artifacts';
import type { ConstructArtifactData, ConstructSegment } from '../types/artifacts';

interface GeneSegment {
  id: string; label: string; start: number; end: number; color: string; desc: string;
  startBp: number; endBp: number; lengthBp: number;
}

/**
 * The segment map comes from `artifacts/construct.json`, which
 * `danon.construct_builder.ConstructBuilder` produced from the engine's own
 * plasmid: ITR / cardiac promoter / LAMP2B CDS / 4x miRNA detarget / WPRE /
 * bGH polyA / ITR, in real base pairs. The previous version of this file drew
 * an invented map (a CMV promoter and a split-intein dual vector) that does
 * not match the construct the repository actually builds.
 */
function toGeneSegments(segments: ConstructSegment[]): GeneSegment[] {
  return segments.map(s => ({
    id: s.id,
    label: s.label,
    start: s.start_frac,
    end: s.end_frac,
    color: s.color,
    desc: s.desc,
    startBp: s.start_bp,
    endBp: s.end_bp,
    lengthBp: s.length_bp,
  }));
}

function colorAt(t: number, segments: GeneSegment[]): THREE.Color {
  for (const s of segments) {
    if (t >= s.start && t <= s.end) return new THREE.Color(s.color);
  }
  return new THREE.Color('#334155');
}

function DoubleHelix({ segments }: { segments: GeneSegment[] }) {
  const groupRef = useRef<THREE.Group>(null!);
  const N = 220;
  const turns = 9;
  const length = 11;
  const radius = 0.9;

  const { strandA, strandB, rungs } = useMemo(() => {
    const a: { pos: THREE.Vector3; color: THREE.Color }[] = [];
    const b: { pos: THREE.Vector3; color: THREE.Color }[] = [];
    const r: { a: THREE.Vector3; b: THREE.Vector3; color: THREE.Color }[] = [];
    for (let i = 0; i < N; i++) {
      const t = i / (N - 1);
      const angle = t * Math.PI * 2 * turns;
      const x = (t - 0.5) * length;
      const col = colorAt(t, segments);
      const pa = new THREE.Vector3(x, Math.cos(angle) * radius, Math.sin(angle) * radius);
      const pb = new THREE.Vector3(x, Math.cos(angle + Math.PI) * radius, Math.sin(angle + Math.PI) * radius);
      a.push({ pos: pa, color: col });
      b.push({ pos: pb, color: col });
      if (i % 4 === 0) r.push({ a: pa, b: pb, color: col });
    }
    return { strandA: a, strandB: b, rungs: r };
  }, [segments]);

  const tubeA = useMemo(() => {
    const curve = new THREE.CatmullRomCurve3(strandA.map(s => s.pos));
    return new THREE.TubeGeometry(curve, N, 0.12, 10, false);
  }, [strandA]);
  const tubeB = useMemo(() => {
    const curve = new THREE.CatmullRomCurve3(strandB.map(s => s.pos));
    return new THREE.TubeGeometry(curve, N, 0.12, 10, false);
  }, [strandB]);

  // vertex-color the tubes along their length
  const colorizeTube = useCallback((geo: THREE.TubeGeometry, strand: { color: THREE.Color }[]) => {
    const pos = geo.attributes.position;
    const cols = new Float32Array(pos.count * 3);
    const ringVerts = 11;
    for (let i = 0; i < pos.count; i++) {
      const seg = Math.floor(i / ringVerts);
      const s = strand[Math.min(seg, strand.length - 1)];
      cols[i*3] = s.color.r; cols[i*3+1] = s.color.g; cols[i*3+2] = s.color.b;
    }
    geo.setAttribute('color', new THREE.BufferAttribute(cols, 3));
  }, []);

  useEffect(() => { colorizeTube(tubeA, strandA); colorizeTube(tubeB, strandB); }, [tubeA, tubeB, strandA, strandB, colorizeTube]);

  useFrame((_, dt) => {
    groupRef.current.rotation.x += dt * 0.15;
  });

  return (
    <group ref={groupRef} rotation={[0, 0, 0]}>
      <mesh geometry={tubeA}>
        <meshPhysicalMaterial vertexColors roughness={0.3} metalness={0.4} clearcoat={0.5} emissiveIntensity={0.3} />
      </mesh>
      <mesh geometry={tubeB}>
        <meshPhysicalMaterial vertexColors roughness={0.3} metalness={0.4} clearcoat={0.5} emissiveIntensity={0.3} />
      </mesh>
      {rungs.map((rung, i) => {
        const mid = rung.a.clone().add(rung.b).multiplyScalar(0.5);
        const dir = rung.b.clone().sub(rung.a);
        const len = dir.length();
        const quat = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.clone().normalize());
        return (
          <mesh key={i} position={mid} quaternion={quat}>
            <cylinderGeometry args={[0.04, 0.04, len, 6]} />
            <meshStandardMaterial color={rung.color} emissive={rung.color} emissiveIntensity={0.4} roughness={0.4} />
          </mesh>
        );
      })}
    </group>
  );
}

function Scene({ segments }: { segments: GeneSegment[] }) {
  return (
    <>
      <color attach="background" args={['#05080f']} />
      <fog attach="fog" args={['#05080f', 10, 24]} />
      <ambientLight intensity={0.3} />
      <directionalLight position={[5, 6, 5]} intensity={1.2} castShadow />
      <directionalLight position={[-5, -2, -5]} intensity={0.5} color="#3b82f6" />
      <pointLight position={[0, 0, 4]} intensity={0.6} color="#a78bfa" />
      <Environment preset="night" />
      <Float speed={1} rotationIntensity={0.15} floatIntensity={0.3}>
        <DoubleHelix segments={segments} />
      </Float>
      <OrbitControls enablePan={false} minDistance={6} maxDistance={18} enableDamping autoRotate autoRotateSpeed={0.3} />
      <EffectComposer>
        <Bloom luminanceThreshold={0.5} intensity={0.7} mipmapBlur />
        <ToneMapping />
      </EffectComposer>
    </>
  );
}

export default function GeneViewer() {
  const [hovered, setHovered] = useState<string | null>(null);
  const { artifact, loading, error } = useArtifact<ConstructArtifactData>('construct');
  const segments = useMemo(() => (artifact ? toGeneSegments(artifact.data.segments) : []), [artifact]);

  if (loading) {
    return <div className="panel-title">🧪 LAMP2B Transgene — loading construct artifact…</div>;
  }
  if (error || !artifact) {
    return <div className="panel-title" style={{ color: 'var(--warning)' }}>🧪 Construct artifact unavailable: {error}</div>;
  }
  const construct = artifact.data;

  return (
    <>
      <div className="panel-title" style={{ marginBottom: 6 }}>
        🧪 LAMP2B Transgene — the engine's own single-vector construct ({construct.total_length_bp} bp)
      </div>
      <div className="canvas-container" style={{ minHeight: 300 }}>
        <ErrorBoundary label="GeneViewer">
          <Canvas shadows dpr={[1, 2]} camera={{ position: [0, 3, 11], fov: 42 }} gl={{ antialias: true, toneMapping: THREE.ACESFilmicToneMapping }}>
            <Scene segments={segments} />
          </Canvas>
        </ErrorBoundary>
      </div>

      <div style={{ marginTop: 10 }}>
        <div style={{ display: 'flex', height: 30, borderRadius: 6, overflow: 'hidden', border: '1px solid #334155' }}>
          {segments.map(s => (
            <div
              key={s.id}
              onMouseEnter={() => setHovered(s.id)}
              onMouseLeave={() => setHovered(null)}
              style={{
                flex: s.end - s.start, background: s.color, cursor: 'pointer',
                opacity: hovered === null || hovered === s.id ? 1 : 0.4,
                transition: 'opacity 0.2s', position: 'relative',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: 8, color: '#fff', fontWeight: 700, overflow: 'hidden', whiteSpace: 'nowrap',
              }}
              title={s.label}
            >
              {(s.end - s.start) > 0.08 ? s.label : ''}
            </div>
          ))}
        </div>
      </div>

      {hovered && (
        <div style={{ marginTop: 8, padding: 10, background: '#1f2937', borderRadius: 8, border: '1px solid #334155', fontSize: 12 }}>
          {(() => {
            const s = segments.find(x => x.id === hovered);
            if (!s) return null;
            return (
              <>
                <strong style={{ color: s.color }}>{s.label}</strong>
                <span style={{ color: '#94a3b8', marginLeft: 8 }}>{s.desc}</span>
                <span style={{ color: '#64748b', marginLeft: 8 }}>
                  bp {s.startBp.toLocaleString()}–{s.endBp.toLocaleString()} ({s.lengthBp} bp)
                </span>
              </>
            );
          })()}
        </div>
      )}

      <div style={{ marginTop: 8, display: 'flex', gap: 20, fontSize: 10, color: '#64748b', flexWrap: 'wrap' }}>
        <span>{construct.total_length_bp.toLocaleString()} bp plasmid · cargo {construct.cargo_length_bp.toLocaleString()} bp (AAV limit {construct.packaging_limit_bp.toLocaleString()} bp)</span>
        <span>promoter: {construct.promoter_name}</span>
        <span>{construct.has_mirna_detarget ? '4x miRNA detarget cassette present' : 'no miRNA detarget cassette'}</span>
        <span>vendor: {construct.synthesis_vendor} · quoted ${construct.synthesis_cost_usd.toFixed(0)}</span>
        <span>Hover backbone segments to inspect</span>
      </div>

      <div style={{ marginTop: 6, color: '#475569', fontSize: 10.5, lineHeight: 1.6 }}>
        {artifactStamp(artifact)}
        <br />Single vector: the LAMP2B cargo fits in one AAV, so the split-intein dual-vector route is not used.
      </div>
    </>
  );
}
