import * as THREE from 'three';

const PHI = (1 + Math.sqrt(5)) / 2;

export const AA_CHARGE: Record<string, number> = {
  R: 1, K: 1, H: 0.5, D: -1, E: -1, S: 0, T: 0, N: 0, Q: 0, A: 0,
  V: 0, L: 0, I: 0, M: 0, F: 0, Y: 0, W: 0, P: 0, G: 0, C: 0,
};

/**
 * The 20 three-fold symmetry axes of the icosahedral capsid (dodecahedron vertex
 * directions). In AAV9 the VR-IV / VR-VIII loops assemble into the characteristic
 * trimeric protrusions ("spikes") that surround each 3-fold axis. Because the
 * capsid is 3-fold symmetric, every protrusion carries an identical copy of the
 * engineered VR-IV + VR-VIII substitutions.
 */
export function threeFoldAxes(): THREE.Vector3[] {
  const v: THREE.Vector3[] = [];
  const inv = 1 / PHI;
  for (const x of [-1, 1]) for (const y of [-1, 1]) for (const z of [-1, 1]) v.push(new THREE.Vector3(x, y, z));
  for (const a of [-inv, inv]) for (const b of [-PHI, PHI]) {
    v.push(new THREE.Vector3(0, a, b));
    v.push(new THREE.Vector3(a, b, 0));
    v.push(new THREE.Vector3(b, 0, a));
  }
  return v.map(p => p.normalize());
}

/** The 12 five-fold axes (icosahedron vertex directions) = capsid pores. */
export function fiveFoldAxes(): THREE.Vector3[] {
  const v: THREE.Vector3[] = [];
  for (const a of [-1, 1]) for (const b of [-PHI, PHI]) {
    v.push(new THREE.Vector3(0, a, b));
    v.push(new THREE.Vector3(a, b, 0));
    v.push(new THREE.Vector3(b, 0, a));
  }
  return v.map(p => p.normalize());
}

/** The 30 two-fold axes (edge midpoints) = the inter-spike canyon/dimple. */
export function twoFoldAxes(): THREE.Vector3[] {
  const tf = threeFoldAxes();
  const ff = fiveFoldAxes();
  const set: THREE.Vector3[] = [];
  // edge midpoints between neighbouring 5-fold vertices
  for (let i = 0; i < ff.length; i++) {
    for (let j = i + 1; j < ff.length; j++) {
      if (ff[i].dot(ff[j]) > 0.4) {
        set.push(ff[i].clone().add(ff[j]).normalize());
      }
    }
  }
  return set.length ? set : tf;
}

export const CHARGE_POSITIVE = new THREE.Color('#1d4ed8'); // deep blue R/K
export const CHARGE_NEGATIVE = new THREE.Color('#dc2626'); // deep red D/E
export const CHARGE_NEUTRAL = new THREE.Color('#7c8ba1');

/** Map a net-charge value to the deep-blue / deep-red diverging scale. */
export function chargeToColor(charge: number, saturate = 1.0): THREE.Color {
  const c = new THREE.Color();
  const t = THREE.MathUtils.clamp(charge / 3.0, -1, 1) * saturate;
  if (t >= 0) c.copy(CHARGE_NEUTRAL).lerp(CHARGE_POSITIVE, t);
  else c.copy(CHARGE_NEUTRAL).lerp(CHARGE_NEGATIVE, -t);
  return c;
}

export interface SpikeDescriptor {
  index: number;
  axis: THREE.Vector3;         // unit direction of the 3-fold axis
  position: THREE.Vector3;     // world position on the shell surface
  quaternion: THREE.Quaternion; // rotates +Y to axis
  netChargeIV: number;
  netChargeVIII: number;
  colorIV: THREE.Color;
  colorVIII: THREE.Color;
}

/**
 * Build the 20 protrusion descriptors. The VR-IV and VR-VIII net surface charges
 * (from the backend Poisson-Boltzmann profile) are applied 3-fold-symmetrically
 * to every protrusion — physically correct for an icosahedral T=1 capsid.
 */
export function computeSpikeDescriptors(
  shellRadius: number,
  netChargeIV: number,
  netChargeVIII: number,
): SpikeDescriptor[] {
  const axes = threeFoldAxes();
  const up = new THREE.Vector3(0, 1, 0);
  const colorIV = chargeToColor(netChargeIV, 1.1);
  const colorVIII = chargeToColor(netChargeVIII, 1.2);
  return axes.map((axis, index) => {
    const q = new THREE.Quaternion().setFromUnitVectors(up, axis);
    return {
      index,
      axis,
      position: axis.clone().multiplyScalar(shellRadius * 0.92),
      quaternion: q,
      netChargeIV,
      netChargeVIII,
      colorIV,
      colorVIII,
    };
  });
}

/**
 * A single 3-fold protrusion: a pronounced raised trimeric finger cluster
 * (three sub-lobes around the axis) sitting on a shoulder ring. Apex lobes =
 * VR-VIII (major antigenic apex), shoulder = VR-IV. Returned pre-oriented to +Y.
 */
export function buildProtrusionGeometry(): { apex: THREE.BufferGeometry; shoulder: THREE.BufferGeometry } {
  const lobes: THREE.BufferGeometry[] = [];
  for (let k = 0; k < 3; k++) {
    const ang = (k / 3) * Math.PI * 2;
    const lobe = new THREE.ConeGeometry(0.20, 0.95, 22, 1);
    lobe.translate(0, 0.475, 0);
    lobe.rotateZ(0.34);
    lobe.rotateY(ang);
    lobe.translate(Math.cos(ang) * 0.22, 0.42, Math.sin(ang) * 0.22);
    lobes.push(lobe);
  }
  const apex = mergeGeometries(lobes);

  const shoulder = new THREE.SphereGeometry(0.50, 28, 18, 0, Math.PI * 2, 0, Math.PI * 0.58);
  shoulder.scale(1, 0.8, 1);

  return { apex, shoulder };
}

/** Minimal geometry merge (positions + normals) to avoid an extra dependency. */
function mergeGeometries(geos: THREE.BufferGeometry[]): THREE.BufferGeometry {
  const merged = new THREE.BufferGeometry();
  let vCount = 0;
  for (const g of geos) vCount += g.attributes.position.count;
  const positions = new Float32Array(vCount * 3);
  const normals = new Float32Array(vCount * 3);
  let off = 0;
  for (const g of geos) {
    const p = g.attributes.position.array as ArrayLike<number>;
    g.computeVertexNormals();
    const n = g.attributes.normal.array as ArrayLike<number>;
    positions.set(p, off);
    normals.set(n, off);
    off += p.length;
  }
  merged.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  merged.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
  return merged;
}

export interface ShellBuildResult {
  geometry: THREE.BufferGeometry;
  radius: number;
}

/**
 * Icosahedral capsid shell: a high-subdivision icosphere depressed at the
 * 5-fold pores and the 2-fold canyons, leaving raised platforms around the
 * 3-fold axes where the protrusion meshes attach. Base colouring is a subtle
 * charge gradient; the sharp electrostatics live on the protrusions.
 */
export function buildCapsidShell(
  detail: number,
  netChargeIV: number,
  netChargeVIII: number,
): ShellBuildResult {
  const R = 2.0;
  const geo = new THREE.IcosahedronGeometry(R, detail);
  const pos = geo.attributes.position;
  const threeFold = threeFoldAxes();
  const fiveFold = fiveFoldAxes();
  const twoFold = twoFoldAxes();

  const colors = new Float32Array(pos.count * 3);
  const emissive = new Float32Array(pos.count);
  const tmp = new THREE.Vector3();

  const platformColor = new THREE.Color('#334155');
  const canyonColor = new THREE.Color('#1e2a3d');
  const c = new THREE.Color();
  const cIV = chargeToColor(netChargeIV, 0.55);
  const cVIII = chargeToColor(netChargeVIII, 0.6);

  for (let i = 0; i < pos.count; i++) {
    tmp.set(pos.getX(i), pos.getY(i), pos.getZ(i)).normalize();

    let max3 = -1;
    for (const ax of threeFold) max3 = Math.max(max3, tmp.dot(ax));
    let max5 = -1;
    for (const ax of fiveFold) max5 = Math.max(max5, tmp.dot(ax));
    let max2 = -1;
    for (const ax of twoFold) max2 = Math.max(max2, tmp.dot(ax));

    const platform = Math.pow(Math.max(0, max3), 6) * 0.10;
    const pore = Math.pow(Math.max(0, max5), 26) * 0.34;
    const canyon = Math.pow(Math.max(0, max2), 16) * 0.18;
    const micro = 0.015 * Math.sin(tmp.x * 34) * Math.sin(tmp.y * 34) * Math.sin(tmp.z * 34);

    const r = R + platform - pore - canyon + micro;
    pos.setXYZ(i, tmp.x * r, tmp.y * r, tmp.z * r);

    // base colour: platforms tinted by region charge, canyons dark
    if (max3 > 0.86) {
      const blend = (max3 - 0.86) / 0.14;
      c.copy(platformColor).lerp(max3 > 0.94 ? cVIII : cIV, blend * 0.7);
      emissive[i] = 0.05 + blend * 0.08;
    } else if (max5 > 0.9) {
      c.copy(canyonColor);
      emissive[i] = 0.0;
    } else {
      c.copy(platformColor).lerp(canyonColor, canyon * 3);
      emissive[i] = 0.02;
    }

    colors[i * 3] = c.r;
    colors[i * 3 + 1] = c.g;
    colors[i * 3 + 2] = c.b;
  }

  geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  geo.setAttribute('aEmissive', new THREE.BufferAttribute(emissive, 1));
  geo.computeVertexNormals();
  return { geometry: geo, radius: R };
}
