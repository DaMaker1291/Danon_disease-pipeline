import React, { useRef, useMemo, useState, useCallback } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Environment, Text, Float } from '@react-three/drei';
import { EffectComposer, Bloom, ToneMapping } from '@react-three/postprocessing';
import * as THREE from 'three';
import ErrorBoundary from './ErrorBoundary';

interface GeneSegment {
  id: string; label: string; start: number; end: number; color: string; desc: string;
}

const SEGMENTS: GeneSegment[] = [
  { id: 'enh1', label: 'cTnT Enhancer', start: 0.0, end: 0.09, color: '#8b5cf6', desc: 'Cardiac-specific enhancer' },
  { id: 'smar', label: 'SMAR Insulator', start: 0.09, end: 0.14, color: '#a78bfa', desc: 'Prevents hepatic silencing (<0.01% leakage)' },
  { id: 'prom', label: 'CMV Promoter', start: 0.14, end: 0.23, color: '#3b82f6', desc: 'Strong constitutive promoter' },
  { id: 'sp', label: 'Signal Peptide', start: 0.23, end: 0.26, color: '#f59e0b', desc: 'ER targeting (MCFRLFVPLL...)' },
  { id: 'lum', label: 'Lumenal Domain', start: 0.26, end: 0.54, color: '#22c55e', desc: 'LAMP2B lumenal domain (1–356 aa)' },
  { id: 'intN', label: 'Intein-N', start: 0.44, end: 0.51, color: '#ef4444', desc: 'Npu DnaE N-intein (102 aa) — split @ aa200' },
  { id: 'intC', label: 'Intein-C', start: 0.51, end: 0.55, color: '#dc2626', desc: 'Npu DnaE C-intein (36 aa)' },
  { id: 'tm', label: 'Transmembrane', start: 0.55, end: 0.60, color: '#06b6d4', desc: 'TM domain (357–380 aa)' },
  { id: 'cyt', label: 'Cytoplasmic Tail', start: 0.60, end: 0.66, color: '#f97316', desc: 'GYQTI tail (381–410 aa)' },
  { id: 'wpre', label: 'WPRE', start: 0.66, end: 0.88, color: '#ec4899', desc: 'Post-transcriptional regulatory element' },
  { id: 'polyA', label: 'bGH polyA', start: 0.88, end: 1.0, color: '#64748b', desc: 'Polyadenylation signal' },
];

function colorAt(t: number): THREE.Color {
  for (const s of SEGMENTS) {
    if (t >= s.start && t <= s.end) return new THREE.Color(s.color);
  }
  return new THREE.Color('#334155');
}

function DoubleHelix({ onHover }: { onHover: (id: string | null) => void }) {
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
      const col = colorAt(t);
      const pa = new THREE.Vector3(x, Math.cos(angle) * radius, Math.sin(angle) * radius);
      const pb = new THREE.Vector3(x, Math.cos(angle + Math.PI) * radius, Math.sin(angle + Math.PI) * radius);
      a.push({ pos: pa, color: col });
      b.push({ pos: pb, color: col });
      if (i % 4 === 0) r.push({ a: pa, b: pb, color: col });
    }
    return { strandA: a, strandB: b, rungs: r };
  }, []);

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

  useMemo(() => { colorizeTube(tubeA, strandA); colorizeTube(tubeB, strandB); }, [tubeA, tubeB, strandA, strandB, colorizeTube]);

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

function Scene({ onHover }: { onHover: (id: string | null) => void }) {
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
        <DoubleHelix onHover={onHover} />
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

  return (
    <>
      <div className="panel-title" style={{ marginBottom: 6 }}>
        🧪 LAMP2B Transgene — 3D DNA Double Helix & Split-Intein Dual-Vector Map
      </div>
      <div className="canvas-container" style={{ minHeight: 300 }}>
        <ErrorBoundary label="GeneViewer">
          <Canvas shadows dpr={[1, 2]} camera={{ position: [0, 3, 11], fov: 42 }} gl={{ antialias: true, toneMapping: THREE.ACESFilmicToneMapping }}>
            <Scene onHover={setHovered} />
          </Canvas>
        </ErrorBoundary>
      </div>

      <div style={{ marginTop: 10 }}>
        <div style={{ display: 'flex', height: 30, borderRadius: 6, overflow: 'hidden', border: '1px solid #334155' }}>
          {SEGMENTS.map(s => (
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
            const s = SEGMENTS.find(x => x.id === hovered);
            if (!s) return null;
            return <><strong style={{ color: s.color }}>{s.label}</strong><span style={{ color: '#94a3b8', marginLeft: 8 }}>{s.desc}</span></>;
          })()}
        </div>
      )}

      <div style={{ marginTop: 8, display: 'flex', gap: 20, fontSize: 10, color: '#64748b', flexWrap: 'wrap' }}>
        <span>~2.0 kb construct (AAV9 limit 4.7 kb)</span>
        <span>Npu DnaE split intein (138 aa)</span>
        <span>Hover backbone segments to inspect</span>
      </div>
    </>
  );
}
