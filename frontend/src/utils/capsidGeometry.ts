import * as THREE from 'three';
import type { StructureArtifactData, StructureResidue, SymmetryOperator } from '../types/artifacts';

/**
 * Capsid geometry, built from the deposited structure.
 *
 * Everything in this file used to be synthetic: an idealised icosphere whose
 * "20 protrusions" sat on hand-computed dodecahedron vertices, sized in
 * arbitrary scene units. It now consumes the `structure` artifact, which is
 * PDB 3UX1 (AAV9) read by `danon/structure.py`:
 *
 *   - positions are the deposit's C-alpha coordinates, in Angstroms;
 *   - the 60-subunit assembly uses the entry's own `REMARK 350 BIOMT`
 *     operators, applied here with the same matrices;
 *   - the protrusions sit on rotation axes *derived* from those operators by
 *     classifying them (20 three-fold positions on 10 axis lines);
 *   - nothing is placed by formula.
 *
 * The viewer still needs scene units, so lengths are scaled by one recorded
 * factor (`outer_radius_a` -> `SCENE_OUTER_RADIUS`) and no other unit trickery
 * happens anywhere.
 */

export const SCENE_OUTER_RADIUS = 2.0;

export interface DecodedStructure {
  /** C-alpha coordinates of the asymmetric unit, in scene units. */
  ca: Float32Array;
  /** Per-residue metadata, index-aligned with `ca`. */
  residues: StructureResidue[];
  /** The 60 deposited symmetry operators, as THREE matrices. */
  operators: THREE.Matrix4[];
  operatorOrder: number[];
  scale: number;
  outerRadiusScene: number;
  /** Real three-fold protrusion directions (unit vectors, scene orientation). */
  protrusionAxes: THREE.Vector3[];
  threeFoldLines: THREE.Vector3[];
  fiveFoldLines: THREE.Vector3[];
  twoFoldLines: THREE.Vector3[];
  /** Centres of the VR loops, on the real protrusions. */
  vrCentroids: Record<string, THREE.Vector3>;
  structureId: string;
  sha256: string | null;
}

export interface ProtrusionPlacement {
  index: number;
  axis: THREE.Vector3;
  position: THREE.Vector3;
  quaternion: THREE.Quaternion;
}

/** Multiply a decoded residue list into a flat Float32Array of positions. */
function toFloat32(residues: StructureResidue[]): Float32Array {
  const out = new Float32Array(residues.length * 3);
  residues.forEach((r, i) => {
    out[i * 3] = r.x;
    out[i * 3 + 1] = r.y;
    out[i * 3 + 2] = r.z;
  });
  return out;
}

function matricesOf(ops: SymmetryOperator[]): { mats: THREE.Matrix4[]; order: number[] } {
  const mats: THREE.Matrix4[] = [];
  const order: number[] = [];
  for (const op of ops) {
    const r = op.rotation;
    const t = op.translation;
    // column-major Matrix4.set: rows of the deposited rotation, then translation
    const m = new THREE.Matrix4();
    m.set(
      r[0][0], r[0][1], r[0][2], t[0],
      r[1][0], r[1][1], r[1][2], t[1],
      r[2][0], r[2][1], r[2][2], t[2],
      0, 0, 0, 1,
    );
    mats.push(m);
    order.push(op.order);
  }
  return { mats, order };
}

/**
 * Decode the structure artifact into scene-space geometry.
 *
 * Coordinates are centred on the capsid centre recorded by the engine, then
 * scaled so the deposit's outer radius maps to `SCENE_OUTER_RADIUS`.
 */
export function decodeStructure(structure: StructureArtifactData): DecodedStructure {
  const g = structure.geometry;
  const scale = SCENE_OUTER_RADIUS / g.outer_radius_a;
  const [cx, cy, cz] = g.centre;

  const residues = structure.asymmetric_unit.map((r) => ({
    ...r,
    x: (r.x - cx) * scale,
    y: (r.y - cy) * scale,
    z: (r.z - cz) * scale,
  }));

  const { mats, order } = matricesOf(structure.symmetry_operators);

  const vec = (v: number[]) => new THREE.Vector3(v[0], v[1], v[2]).normalize();
  const protrusionAxes = structure.symmetry_operators
    .filter((op) => op.order === 3 && op.axis)
    .map((op) => vec(op.axis as number[]));

  const vrCentroids: Record<string, THREE.Vector3> = {};
  for (const [name, region] of Object.entries(structure.vr_regions)) {
    const c = new THREE.Vector3();
    for (const r of region.residues) {
      c.add(new THREE.Vector3((r.x - cx) * scale, (r.y - cy) * scale, (r.z - cz) * scale));
    }
    vrCentroids[name] = region.residues.length ? c.multiplyScalar(1 / region.residues.length) : c;
  }

  return {
    ca: toFloat32(residues),
    residues,
    operators: mats,
    operatorOrder: order,
    scale,
    outerRadiusScene: SCENE_OUTER_RADIUS,
    protrusionAxes,
    threeFoldLines: structure.axes['3_fold'].lines.map(vec),
    fiveFoldLines: structure.axes['5_fold'].lines.map(vec),
    twoFoldLines: structure.axes['2_fold'].lines.map(vec),
    vrCentroids,
    structureId: structure.structure_id,
    sha256: structure.sha256 ?? null,
  };
}

export const AA_CHARGE: Record<string, number> = {
  R: 1, K: 1, H: 0.5, D: -1, E: -1, S: 0, T: 0, N: 0, Q: 0, A: 0,
  V: 0, L: 0, I: 0, M: 0, F: 0, Y: 0, W: 0, P: 0, G: 0, C: 0,
};

const THREE_TO_ONE: Record<string, string> = {
  ALA: 'A', ARG: 'R', ASN: 'N', ASP: 'D', CYS: 'C', GLN: 'Q', GLU: 'E',
  GLY: 'G', HIS: 'H', ILE: 'I', LEU: 'L', LYS: 'K', MET: 'M', PHE: 'F',
  PRO: 'P', SER: 'S', THR: 'T', TRP: 'W', TYR: 'Y', VAL: 'V',
};

export function oneLetter(three: string): string {
  return THREE_TO_ONE[three] ?? 'X';
}

export const CHARGE_POSITIVE = new THREE.Color('#3b82f6');
export const CHARGE_NEGATIVE = new THREE.Color('#ef4444');
export const CHARGE_NEUTRAL = new THREE.Color('#8fa3bd');
/** The residues this design actually edited. */
export const MUTATION_COLOR = new THREE.Color('#fbbf24');
/** Wild-type epitope residues (the antibody footprint). */
export const EPITOPE_COLOR = new THREE.Color('#f472b6');

/** Map a net-charge value to the deep-blue / deep-red diverging scale. */
export function chargeToColor(charge: number, saturate = 1.0): THREE.Color {
  const c = new THREE.Color();
  const t = THREE.MathUtils.clamp(charge / 3.0, -1, 1) * saturate;
  if (t >= 0) c.copy(CHARGE_NEUTRAL).lerp(CHARGE_POSITIVE, t);
  else c.copy(CHARGE_NEUTRAL).lerp(CHARGE_NEGATIVE, -t);
  return c;
}

export interface BackboneBuild {
  geometry: THREE.BufferGeometry;
  /** Number of points = subunits x asymmetric-unit residues. */
  points: number;
  mutatedSceneIndices: number[];
}

/**
 * The 60-subunit C-alpha backbone.
 *
 * Each residue of the asymmetric unit is drawn once per symmetry operator,
 * using the deposited matrices. Colour encodes what a viewer needs to check
 * the design against the structure: mutated positions, wild-type epitope
 * residues, and otherwise burial (contact number from the engine).
 */
export function buildBackbone(
  structure: DecodedStructure,
  mutations: { position: number; original: string; mutated: string }[] = [],
  epitopeStored: number[] = [],
): BackboneBuild {
  const nRes = structure.residues.length;
  const nSub = structure.operators.length;
  const total = nRes * nSub;
  const positions = new Float32Array(total * 3);
  const colors = new Float32Array(total * 3);

  const mutated = new Set(mutations.map((m) => m.position));
  const epitopes = new Set(epitopeStored);
  const maxContacts = Math.max(...structure.residues.map((r) => r.contacts), 1);

  const base = structure.residues.map((r) => {
    const one = oneLetter(r.aa);
    const c = chargeToColor(AA_CHARGE[one] ?? 0, 0.85);
    // bury inward: exposed residues keep the charge hue, buried ones darken
    const burial = 1 - r.contacts / maxContacts;
    return c.clone().lerp(new THREE.Color('#0b1220'), burial * 0.75);
  });

  const tmp = new THREE.Vector3();
  let k = 0;
  for (let s = 0; s < nSub; s++) {
    for (let i = 0; i < nRes; i++) {
      const r = structure.residues[i];
      tmp.set(r.x, r.y, r.z).applyMatrix4(structure.operators[s]);
      positions[k * 3] = tmp.x;
      positions[k * 3 + 1] = tmp.y;
      positions[k * 3 + 2] = tmp.z;

      let col = base[i];
      const stored = r.stored;
      if (mutated.has(stored)) col = MUTATION_COLOR;
      else if (epitopes.has(stored)) col = EPITOPE_COLOR;
      colors[k * 3] = col.r;
      colors[k * 3 + 1] = col.g;
      colors[k * 3 + 2] = col.b;
      k++;
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

  const mutatedSceneIndices = structure.residues
    .map((r, i) => (mutated.has(r.stored) ? i : -1))
    .filter((i) => i >= 0);

  return { geometry, points: total, mutatedSceneIndices };
}

/**
 * Place a protrusion on each real three-fold axis.
 *
 * The axis set comes from the deposited operators, so these directions are
 * the capsid's symmetry elements rather than an assumed icosahedron. Radius
 * is the deposit's own outer radius.
 */
export function buildProtrusions(structure: DecodedStructure): ProtrusionPlacement[] {
  const up = new THREE.Vector3(0, 1, 0);
  const r = structure.outerRadiusScene * 0.93;
  return structure.protrusionAxes.map((axis, index) => ({
    index,
    axis: axis.clone(),
    position: axis.clone().multiplyScalar(r),
    quaternion: new THREE.Quaternion().setFromUnitVectors(up, axis),
  }));
}

/**
 * A single 3-fold protrusion mesh: three sub-lobes around the axis.
 *
 * Deliberately symbolic. It marks where the deposit's three-fold operator sits;
 * the residues are drawn separately, at their real coordinates. No claim is
 * made that this shape is the capsid surface.
 */
export function buildProtrusionGeometry(): { apex: THREE.BufferGeometry } {
  const lobes: THREE.BufferGeometry[] = [];
  for (let k = 0; k < 3; k++) {
    const ang = (k / 3) * Math.PI * 2;
    const lobe = new THREE.ConeGeometry(0.16, 0.62, 20, 1);
    lobe.translate(0, 0.31, 0);
    lobe.rotateZ(0.3);
    lobe.rotateY(ang);
    lobe.translate(Math.cos(ang) * 0.16, 0.26, Math.sin(ang) * 0.16);
    lobes.push(lobe);
  }
  const merged = new THREE.BufferGeometry();
  let vCount = 0;
  for (const g of lobes) vCount += g.attributes.position.count;
  const positions = new Float32Array(vCount * 3);
  const normals = new Float32Array(vCount * 3);
  let off = 0;
  for (const g of lobes) {
    const p = g.attributes.position.array as ArrayLike<number>;
    g.computeVertexNormals();
    const n = g.attributes.normal.array as ArrayLike<number>;
    positions.set(p, off);
    normals.set(n, off);
    off += p.length;
  }
  merged.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  merged.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
  return { apex: merged };
}

/** Real coordinates of a VR loop's residues, ready to draw as spheres. */
export function regionPoints(
  structure: StructureArtifactData,
  region: string,
): { vp1: number; stored: number; aa: string; vec: THREE.Vector3 }[] {
  const r = structure.vr_regions[region];
  if (!r) return [];
  const g = structure.geometry;
  const scale = SCENE_OUTER_RADIUS / g.outer_radius_a;
  const [cx, cy, cz] = g.centre;
  return r.residues.map((res) => ({
    vp1: res.vp1,
    stored: res.stored,
    aa: res.aa,
    vec: new THREE.Vector3((res.x - cx) * scale, (res.y - cy) * scale, (res.z - cz) * scale),
  }));
}
