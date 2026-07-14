import React, { useRef, useMemo, useCallback, useState } from 'react';
import { Canvas, useFrame, ThreeEvent } from '@react-three/fiber';
import { OrbitControls, Environment, Sparkles, Html, ContactShadows } from '@react-three/drei';
import { EffectComposer, Bloom, N8AO, ToneMapping, Vignette } from '@react-three/postprocessing';
import * as THREE from 'three';
import ErrorBoundary from './ErrorBoundary';
import {
  buildCapsidShell,
  buildProtrusionGeometry,
  computeSpikeDescriptors,
  chargeToColor,
  AA_CHARGE,
  type SpikeDescriptor,
} from '../utils/capsidGeometry';
import type { PipelineResult } from '../types';

interface Props {
  result: PipelineResult | null;
  selectedSpike: number | null;
  onSelectSpike: (i: number | null) => void;
}

function emissiveShader(shader: any) {
  shader.vertexShader = 'attribute float aEmissive;\nvarying float vEmissive;\n' +
    shader.vertexShader.replace('#include <begin_vertex>', '#include <begin_vertex>\n  vEmissive = aEmissive;');
  shader.fragmentShader = 'varying float vEmissive;\n' +
    shader.fragmentShader.replace(
      '#include <emissivemap_fragment>',
      '#include <emissivemap_fragment>\n  totalEmissiveRadiance += vColor.rgb * vEmissive * 1.6;'
    );
}

const R = 2.0;
const icosa = new THREE.IcosahedronGeometry(R * 1.005, 1);

function IcosaEdges() {
  const ref = useRef<THREE.LineSegments>(null!);
  const geo = useMemo(() => {
    const eg = new THREE.EdgesGeometry(icosa, 1);
    return eg;
  }, []);
  return (
    <lineSegments geometry={geo}>
      <lineBasicMaterial color="#7dd3fc" transparent opacity={0.35} />
    </lineSegments>
  );
}

function SpikeLabels({ spikes }: { spikes: SpikeDescriptor[] }) {
  return (
    <>
      {spikes.map(s => (
        <Html key={`lbl-${s.index}`} center distanceFactor={10} position={s.position} zIndexRange={[10, 0]}>
          <div className="spike-num">{s.index + 1}</div>
        </Html>
      ))}
    </>
  );
}

function Assembly({ result, selectedSpike, onSelectSpike }: Props) {
  const groupRef = useRef<THREE.Group>(null!);
  const [hover, setHover] = useState<number | null>(null);

  const netIV = result?.regions?.VR_IV?.netCharge ?? 0;
  const netVIII = result?.regions?.VR_VIII?.netCharge ?? 0;

  const { geometry: shellGeo, radius } = useMemo(
    () => buildCapsidShell(6, netIV, netVIII),
    [netIV, netVIII]
  );
  const { apex, shoulder } = useMemo(() => buildProtrusionGeometry(), []);
  const spikes = useMemo<SpikeDescriptor[]>(
    () => computeSpikeDescriptors(radius, netIV, netVIII),
    [radius, netIV, netVIII]
  );

  useFrame((_, dt) => {
    groupRef.current.rotation.y += dt * 0.14;
    groupRef.current.rotation.x = Math.sin(Date.now() * 0.0001) * 0.12;
  });

  const handleClick = useCallback((e: ThreeEvent<MouseEvent>, i: number) => {
    e.stopPropagation();
    onSelectSpike(selectedSpike === i ? null : i);
  }, [onSelectSpike, selectedSpike]);

  return (
    <group ref={groupRef}>
      <mesh geometry={shellGeo} castShadow receiveShadow>
        <meshPhysicalMaterial
          vertexColors
          roughness={0.4}
          metalness={0.5}
          clearcoat={0.5}
          clearcoatRoughness={0.35}
          reflectivity={0.55}
          envMapIntensity={1.0}
          onBeforeCompile={emissiveShader}
        />
      </mesh>

      <IcosaEdges />

      {spikes.map((s) => {
        const active = selectedSpike === s.index || hover === s.index;
        const emis = active ? 1.4 : 0.25;
        const scale = active ? 1.18 : 1.0;
        return (
          <group
            key={s.index}
            position={s.position}
            quaternion={s.quaternion}
            scale={scale}
            onClick={(e) => handleClick(e, s.index)}
            onPointerOver={(e) => { e.stopPropagation(); setHover(s.index); document.body.style.cursor = 'pointer'; }}
            onPointerOut={() => { setHover(null); document.body.style.cursor = 'auto'; }}
          >
            <mesh geometry={shoulder} castShadow>
              <meshPhysicalMaterial
                color={s.colorIV}
                emissive={s.colorIV}
                emissiveIntensity={emis * 0.5}
                roughness={0.35}
                metalness={0.4}
                clearcoat={0.6}
              />
            </mesh>
            <mesh geometry={apex} position={[0, 0.18, 0]} castShadow>
              <meshPhysicalMaterial
                color={s.colorVIII}
                emissive={s.colorVIII}
                emissiveIntensity={emis}
                roughness={0.3}
                metalness={0.45}
                clearcoat={0.7}
              />
            </mesh>
            {active && (
              <Html center distanceFactor={9} position={[0, 1.1, 0]}>
                <div className="spike-tag">
                  Protrusion #{s.index + 1}
                  <span>VR-IV {netIV >= 0 ? '+' : ''}{netIV.toFixed(1)} · VR-VIII {netVIII >= 0 ? '+' : ''}{netVIII.toFixed(1)}</span>
                </div>
              </Html>
            )}
          </group>
        );
      })}

      {result && <OrbitingIons netIV={netIV} netVIII={netVIII} />}
      {result && <SpikeLabels spikes={spikes} />}
    </group>
  );
}

function OrbitingIons({ netIV, netVIII }: { netIV: number; netVIII: number }) {
  const count = 220;
  const ref = useRef<THREE.Points>(null!);
  const { positions, initialY, colors, speeds } = useMemo(() => {
    const p = new Float32Array(count * 3);
    const iy = new Float32Array(count);
    const col = new Float32Array(count * 3);
    const s = new Float32Array(count);
    const bias = (netIV + netVIII) / 2;
    for (let i = 0; i < count; i++) {
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      const r = 2.8 + Math.random() * 1.7;
      p[i * 3] = r * Math.sin(phi) * Math.cos(theta);
      p[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
      p[i * 3 + 2] = r * Math.cos(phi);
      iy[i] = p[i * 3 + 1];
      s[i] = 0.2 + Math.random() * 0.6;
      const c = chargeToColor(-bias * (0.5 + Math.random()), 1.4);
      col[i * 3] = c.r; col[i * 3 + 1] = c.g; col[i * 3 + 2] = c.b;
    }
    return { positions: p, initialY: iy, colors: col, speeds: s };
  }, [netIV, netVIII]);

  useFrame(() => {
    ref.current.rotation.y -= 0.0015;
    const arr = ref.current.geometry.attributes.position.array as Float32Array;
    for (let i = 0; i < count; i++) {
      const t = Date.now() * 0.0004 * speeds[i] + i;
      arr[i * 3 + 1] = initialY[i] + Math.sin(t) * 0.15;
    }
    ref.current.geometry.attributes.position.needsUpdate = true;
  });

  return (
    <points ref={ref}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
        <bufferAttribute attach="attributes-color" args={[colors, 3]} />
      </bufferGeometry>
      <pointsMaterial size={0.05} vertexColors transparent opacity={0.8} sizeAttenuation depthWrite={false} blending={THREE.AdditiveBlending} />
    </points>
  );
}

function Scene(props: Props) {
  return (
    <>
      <color attach="background" args={['#05080f']} />
      <fog attach="fog" args={['#05080f', 9, 20]} />
      <ambientLight intensity={0.28} />
      <directionalLight position={[6, 8, 5]} intensity={1.6} castShadow shadow-mapSize={[2048, 2048]} />
      <directionalLight position={[-6, -3, -6]} intensity={0.6} color="#2563eb" />
      <pointLight position={[0, 4, 0]} intensity={0.8} color="#60a5fa" distance={14} />
      <spotLight position={[-4, 6, 4]} angle={0.5} penumbra={0.8} intensity={1.2} color="#a78bfa" castShadow />
      <Environment preset="night" />

      <Assembly {...props} />
      <Sparkles count={70} scale={[10, 10, 10]} size={1.3} speed={0.25} color="#93c5fd" opacity={0.35} />

      <ContactShadows position={[0, -3.6, 0]} opacity={0.5} scale={16} blur={2.4} far={5} />
      <OrbitControls enablePan={false} minDistance={3.5} maxDistance={14} enableDamping dampingFactor={0.05} />

      <EffectComposer enableNormalPass>
        <N8AO aoRadius={1.2} intensity={2.0} distanceFalloff={1.0} />
        <Bloom luminanceThreshold={0.5} luminanceSmoothing={0.3} intensity={0.95} mipmapBlur />
        <ToneMapping />
        <Vignette eskil={false} offset={0.15} darkness={0.75} />
      </EffectComposer>
    </>
  );
}

export default function CapsidViewer(props: Props) {
  return (
    <>
      <div className="panel-title" style={{ marginBottom: 6 }}>
        🧬 AAV9 Capsid — Icosahedral Assembly · 20 Three-Fold Protrusions · PDB 3J1S Electrostatics
      </div>
      <div className="canvas-container" style={{ position: 'relative' }}>
        <ErrorBoundary label="CapsidViewer">
          <Canvas
            shadows
            dpr={[1, 2]}
            camera={{ position: [5.5, 3, 5.5], fov: 38 }}
            gl={{ antialias: true, toneMapping: THREE.ACESFilmicToneMapping, toneMappingExposure: 1.1 }}
          >
            <Scene {...props} />
          </Canvas>
        </ErrorBoundary>
        {!props.result && (
          <div className="canvas-overlay">
            <div className="overlay-card">
              <span className="overlay-icon">🧬</span>
              <strong>No engineered capsid yet</strong>
              <p>Set target constraints and run the optimization pipeline. The backend NSGA-II + PDB&nbsp;3J1S Poisson-Boltzmann modules will generate the capsid and paint these protrusions.</p>
            </div>
          </div>
        )}
      </div>
      <div className="capsid-legend">
        <span><i style={{ background: '#1d4ed8' }} /> Positive patch (R/K)</span>
        <span><i style={{ background: '#dc2626' }} /> Negative patch (D/E)</span>
        <span><i style={{ background: '#7c8ba1' }} /> Neutral surface</span>
        <span className="hint">Click a protrusion → highlights VR-IV / VR-VIII substitutions →</span>
      </div>
    </>
  );
}
