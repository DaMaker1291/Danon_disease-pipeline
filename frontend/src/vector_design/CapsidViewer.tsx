import React, { useRef, useMemo, useCallback, useState, useEffect } from 'react';
import { Canvas, useFrame, ThreeEvent } from '@react-three/fiber';
import { OrbitControls, Environment, Sparkles, Html, ContactShadows } from '@react-three/drei';
import { EffectComposer, Bloom, N8AO, ToneMapping, Vignette } from '@react-three/postprocessing';
import * as THREE from 'three';
import ErrorBoundary from '../components/ErrorBoundary';
import {
  buildBackbone,
  buildProtrusionGeometry,
  buildProtrusions,
  decodeStructure,
  MUTATION_COLOR,
  type DecodedStructure,
  type ProtrusionPlacement,
} from './capsidGeometry';
import { useArtifact } from '../utils/artifacts';
import type { PipelineResult } from '../types';
import type { StructureArtifactData } from '../types/artifacts';

interface Props {
  result: PipelineResult | null;
  selectedSpike: number | null;
  onSelectSpike: (i: number | null) => void;
}

/* ------------------------------------------------------------------ scene -- */

function AxisFrame({ structure }: { structure: DecodedStructure }) {
  // The capsid's own symmetry elements, drawn from the deposited operators.
  const geometries = useMemo(() => {
    const line = (axes: THREE.Vector3[], color: string, opacity: number) => {
      const pts: number[] = [];
      const r = structure.outerRadiusScene * 1.08;
      for (const a of axes) {
        pts.push(a.x * r, a.y * r, a.z * r, -a.x * r, -a.y * r, -a.z * r);
      }
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
      return (
        <lineSegments key={color} geometry={g}>
          <lineBasicMaterial color={color} transparent opacity={opacity} />
        </lineSegments>
      );
    };
    return [
      line(structure.threeFoldLines, '#7dd3fc', 0.5),
      line(structure.fiveFoldLines, '#a78bfa', 0.4),
      line(structure.twoFoldLines, '#64748b', 0.25),
    ];
  }, [structure]);
  return <>{geometries}</>;
}

function SpikeLabels({ protrusions }: { protrusions: ProtrusionPlacement[] }) {
  return (
    <>
      {protrusions.map((p) => (
        <Html
          key={`lbl-${p.index}`}
          center
          distanceFactor={10}
          position={p.position}
          zIndexRange={[10, 0]}
        >
          <div className="spike-num">{p.index + 1}</div>
        </Html>
      ))}
    </>
  );
}

function Assembly({ result, selectedSpike, onSelectSpike, structure }: Props & { structure: DecodedStructure }) {
  const groupRef = useRef<THREE.Group>(null!);
  const [hover, setHover] = useState<number | null>(null);

  const mutations = result?.mutations ?? [];
  const epitopeStored = useMemo(() => {
    const ep = result?.regions?.VR_IV?.residueProfiles ?? [];
    const ep8 = result?.regions?.VR_VIII?.residueProfiles ?? [];
    return [...ep, ...ep8].filter((r) => r.isEpitope).map((r) => r.position);
  }, [result]);

  const backbone = useMemo(
    () => buildBackbone(structure, mutations, epitopeStored),
    [structure, mutations, epitopeStored],
  );
  const { apex } = useMemo(() => buildProtrusionGeometry(), []);
  const protrusions = useMemo(() => buildProtrusions(structure), [structure]);

  useFrame((_, dt) => {
    groupRef.current.rotation.y += dt * 0.14;
    groupRef.current.rotation.x = Math.sin(Date.now() * 0.0001) * 0.12;
  });

  const handleClick = useCallback((e: ThreeEvent<MouseEvent>, i: number) => {
    e.stopPropagation();
    onSelectSpike(selectedSpike === i ? null : i);
  }, [onSelectSpike, selectedSpike]);

  const active = (i: number) => selectedSpike === i || hover === i;

  return (
    <group ref={groupRef}>
      {/* the deposited C-alpha backbone: 60 subunits x 518 residues */}
      <points geometry={backbone.geometry}>
        <pointsMaterial
          size={0.032}
          vertexColors
          transparent
          opacity={0.95}
          sizeAttenuation
          depthWrite={false}
          blending={THREE.AdditiveBlending}
        />
      </points>

      <AxisFrame structure={structure} />

      {protrusions.map((p) => {
        const on = active(p.index);
        const colour = on ? MUTATION_COLOR : new THREE.Color('#334155');
        return (
          <group
            key={p.index}
            position={p.position}
            quaternion={p.quaternion}
            scale={on ? 1.22 : 1.0}
            onClick={(e) => handleClick(e, p.index)}
            onPointerOver={(e) => { e.stopPropagation(); setHover(p.index); document.body.style.cursor = 'pointer'; }}
            onPointerOut={() => { setHover(null); document.body.style.cursor = 'auto'; }}
          >
            <mesh geometry={apex} castShadow>
              <meshPhysicalMaterial
                color={colour}
                emissive={colour}
                emissiveIntensity={on ? 1.2 : 0.2}
                roughness={0.32}
                metalness={0.42}
                clearcoat={0.6}
              />
            </mesh>
            {on && (
              <Html center distanceFactor={9} position={[0, 0.9, 0]}>
                <div className="spike-tag">
                  Three-fold operator #{p.index + 1}
                  <span>
                    {result
                      ? `VR-IV ${result.regions?.VR_IV?.netCharge >= 0 ? '+' : ''}${(result.regions?.VR_IV?.netCharge ?? 0).toFixed(2)} · VR-VIII ${result.regions?.VR_VIII?.netCharge >= 0 ? '+' : ''}${(result.regions?.VR_VIII?.netCharge ?? 0).toFixed(2)}`
                      : 'engine charge profile loads with the pipeline run'}
                  </span>
                </div>
              </Html>
            )}
          </group>
        );
      })}

      {/* mutated positions, drawn at their real coordinates on every subunit */}
      {backbone.mutatedSceneIndices.map((idx) => {
        const r = structure.residues[idx];
        const c = new THREE.Vector3(r.x, r.y, r.z);
        return structure.operators.map((m, s) => {
          const p = c.clone().applyMatrix4(m);
          return (
            <mesh key={`mut-${idx}-${s}`} position={p} scale={0.05}>
              <sphereGeometry args={[1, 10, 10]} />
              <meshBasicMaterial color={MUTATION_COLOR} />
            </mesh>
          );
        });
      })}
    </group>
  );
}

function Scene(props: Props & { structure: DecodedStructure }) {
  return (
    <>
      <color attach="background" args={['#05080f']} />
      <fog attach="fog" args={['#05080f', 9, 20]} />
      <ambientLight intensity={0.3} />
      <directionalLight position={[6, 8, 5]} intensity={1.5} castShadow shadow-mapSize={[2048, 2048]} />
      <directionalLight position={[-6, -3, -6]} intensity={0.6} color="#2563eb" />
      <pointLight position={[0, 4, 0]} intensity={0.8} color="#60a5fa" distance={14} />
      <spotLight position={[-4, 6, 4]} angle={0.5} penumbra={0.8} intensity={1.2} color="#a78bfa" castShadow />
      <Environment preset="night" />

      <Assembly {...props} />
      <Sparkles count={70} scale={[10, 10, 10]} size={1.3} speed={0.25} color="#93c5fd" opacity={0.3} />

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

/* ------------------------------------------------------------------ shell -- */

function MissingStructure() {
  return (
    <div className="canvas-overlay">
      <div className="overlay-card">
        
        <strong>No structure artifact</strong>
        <p>
          This view renders the deposited AAV9 capsid (PDB&nbsp;3UX1). It has no
          coordinates loaded, so it draws nothing rather than an idealised shell.
          Generate it with <code>write_app_artifacts()</code>.
        </p>
      </div>
    </div>
  );
}

export default function CapsidViewer(props: Props) {
  const { artifact, error, loading } = useArtifact<StructureArtifactData>('structure', 'vector_design');
  const [decoded, setDecoded] = useState<DecodedStructure | null>(null);

  useEffect(() => {
    if (!artifact) return;
    try {
      setDecoded(decodeStructure(artifact.data));
    } catch {
      setDecoded(null);
    }
  }, [artifact]);

  const g = artifact?.data.geometry;

  return (
    <>
      <div className="panel-title" style={{ marginBottom: 6 }}>
        AAV9 Capsid — {g ? `${g.subunits}-subunit T=1 assembly` : 'T=1 assembly'} ·{' '}
        PDB&nbsp;{artifact?.data.structure_id ?? '3UX1'} deposited coordinates ·{' '}
        {g ? `${g.protrusions} three-fold protrusions on ${g.three_fold_axis_lines} axis lines` : 'symmetry from REMARK 350'}
      </div>
      <div className="canvas-container" style={{ position: 'relative' }}>
        <ErrorBoundary label="CapsidViewer">
          <Canvas
            shadows
            dpr={[1, 2]}
            camera={{ position: [5.5, 3, 5.5], fov: 38 }}
            gl={{ antialias: true, toneMapping: THREE.ACESFilmicToneMapping, toneMappingExposure: 1.1 }}
          >
            {decoded && <Scene {...props} structure={decoded} />}
          </Canvas>
        </ErrorBoundary>
        {!decoded && !loading && <MissingStructure />}
        {error && (
          <div className="canvas-overlay" style={{ top: 'auto', bottom: 8, padding: 0 }}>
            <div className="overlay-card" style={{ fontSize: 11 }}>{error}</div>
          </div>
        )}
      </div>
      <div className="capsid-legend">
        <span><i style={{ background: '#8fa3bd' }} /> Backbone Cα, shaded by burial (contact number)</span>
        <span><i style={{ background: '#fbbf24' }} /> Engineered substitution</span>
        <span><i style={{ background: '#f472b6' }} /> Wild-type epitope residue</span>
        <span className="hint">
          {g
            ? `outer radius ${g.outer_radius_a.toFixed(1)} Å · ${g.ca_atoms.toLocaleString()} Cα atoms · sha256 ${artifact?.data.sha256?.slice(0, 12) ?? '—'}`
            : 'structure artifact not loaded'}
        </span>
      </div>
    </>
  );
}
