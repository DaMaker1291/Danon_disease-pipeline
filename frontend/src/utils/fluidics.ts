/**
 * TypeScript port of `danon/microfluidics_core.py` (MicrofluidicsCore).
 *
 * The Python module is the source of truth; this port exists so the sliders in
 * the microfluidics view recompute the SAME correlations instead of a local
 * toy formula (`0.36 * velocity`), which is what the view used before. The
 * view cross-checks this port against the generated artifact at the artifact's
 * own channel configuration, so a drift between the two is visible on screen
 * rather than silent.
 *
 * Ported relations (identical to the Python):
 *   D_h = 2HW/(H+W)                      hydraulic diameter
 *   v   = Q/(60e9 * H * W)               flow rate (µL/min) -> velocity (m/s)
 *   Re  = rho * v * D_h / mu
 *   tau = 2 mu v / H  (wall),  3 mu v / H  (maximum, at the wall centre)
 *   Pe  = v * D_h / D
 *   f   = 16/Re for Re < 1, else 0.079 Re^-0.25 ;  dP = f (L/D_h) (1/2) rho v^2
 *   FRR = Q_org / (Q_aq + Q_org)
 *   eta = 1 - var(c)/sigma_max^2 on c = linspace(1-FRR, FRR)
 */

import type {
  MicrofluidicConfigData,
  MicrofluidicTelemetryData,
} from '../types/artifacts';

export const WATER_DENSITY_KG_M3 = 997.0;
export const ETHANOL_DENSITY_KG_M3 = 789.0;
export const WATER_VISCOSITY_PA_S = 8.9e-4;
export const ETHANOL_VISCOSITY_PA_S = 1.07e-3;

export function hydraulicDiameter(config: MicrofluidicConfigData): number {
  const H = config.channel_height_um * 1e-6;
  const W = config.channel_width_um * 1e-6;
  return (2 * H * W) / (H + W);
}

function flowRateToVelocity(config: MicrofluidicConfigData, flowRateUlMin: number): number {
  const H = config.channel_height_um * 1e-6;
  const W = config.channel_width_um * 1e-6;
  const areaCross = H * W;
  const q = (flowRateUlMin * 1e-9) / 60.0;
  return q / Math.max(areaCross, 1e-20);
}

export function flowRateRatio(config: MicrofluidicConfigData): number {
  const total = config.aqueous_flow_rate_ul_min + config.organic_flow_rate_ul_min;
  return config.organic_flow_rate_ul_min / Math.max(total, 1e-12);
}

/** Same correlations as MicrofluidicsCore.simulate(). */
export function computeTelemetry(config: MicrofluidicConfigData): MicrofluidicTelemetryData {
  const H = config.channel_height_um * 1e-6;
  const Dh = hydraulicDiameter(config);

  const vAq = flowRateToVelocity(config, config.aqueous_flow_rate_ul_min);
  const vOrg = flowRateToVelocity(config, config.organic_flow_rate_ul_min);
  const totalFlow = config.aqueous_flow_rate_ul_min + config.organic_flow_rate_ul_min;
  const fracAq = config.aqueous_flow_rate_ul_min / Math.max(totalFlow, 1e-12);
  const mixedDensity =
    config.aqueous_density_kg_m3 * fracAq + config.organic_density_kg_m3 * (1 - fracAq);
  const mixedViscosity =
    config.aqueous_viscosity_pa_s * fracAq + config.organic_viscosity_pa_s * (1 - fracAq);
  const vMix = flowRateToVelocity(config, totalFlow);

  const reynolds = (v: number, rho: number, mu: number) =>
    (rho * v * Dh) / Math.max(mu, 1e-20);

  const ReAq = reynolds(vAq, config.aqueous_density_kg_m3, config.aqueous_viscosity_pa_s);
  const ReOrg = reynolds(vOrg, config.organic_density_kg_m3, config.organic_viscosity_pa_s);
  const ReMix = reynolds(vMix, mixedDensity, mixedViscosity);

  const tauWall = (2 * mixedViscosity * vMix) / H;
  const tauMax = (3 * mixedViscosity * vMix) / H;

  const Pe = (vMix * Dh) / Math.max(config.diff_coefficient_m2_s, 1e-20);

  const L = config.channel_length_mm * 1e-3;
  const f = ReMix < 1 ? 16 / Math.max(ReMix, 1e-20) : 0.079 * Math.pow(ReMix, -0.25);
  const dP = f * (L / Dh) * 0.5 * mixedDensity * vMix * vMix;

  const frr = flowRateRatio(config);
  const n = 50;
  const c = Array.from({ length: n }, (_, i) => 1 - frr + ((frr - (1 - frr)) * i) / (n - 1));
  const mean = c.reduce((a, b) => a + b, 0) / n;
  const variance = c.reduce((a, b) => a + (b - mean) * (b - mean), 0) / n;
  const cMax = Math.max(...c);
  const cMin = Math.min(...c);
  const sigmaMax = cMax > cMin ? 0.25 * (cMax - cMin) ** 2 : 1.0;
  const eta = 1 - variance / Math.max(sigmaMax, 1e-20);

  const regime =
    ReMix < 1 ? 'creeping (Stokes)'
      : ReMix < 2000 ? 'laminar'
        : ReMix < 4000 ? 'transitional'
          : 'turbulent';

  const RCurve = Dh / (2 * Math.sin((config.inlet_angle_deg * Math.PI) / 180 / 2) + 1e-20);
  const Dn = ReMix * Math.sqrt(Dh / Math.max(RCurve, 1e-20));

  return {
    reynolds_number_aqueous: ReAq,
    reynolds_number_organic: ReOrg,
    reynolds_number_mixed: ReMix,
    wall_shear_stress_pa: tauWall,
    max_shear_stress_pa: tauMax,
    mixing_efficiency: eta,
    peclet_number: Pe,
    deans_number: Dn,
    pressure_drop_pa: dP,
    flow_regime: regime,
    shear_stress_safe: tauMax < 50.0,
    frr,
    hydraulic_diameter_um: Dh * 1e6,
  };
}

export interface PortCheck {
  matches: boolean;
  maxRelativeDelta: number;
  worstField: string;
}

/** Does this port still reproduce the Python artifact at the same config? */
export function checkPort(
  computed: MicrofluidicTelemetryData,
  artifact: MicrofluidicTelemetryData,
  tolerance = 1e-9,
): PortCheck {
  const numeric = [
    'reynolds_number_aqueous', 'reynolds_number_organic', 'reynolds_number_mixed',
    'wall_shear_stress_pa', 'max_shear_stress_pa', 'mixing_efficiency',
    'peclet_number', 'deans_number', 'pressure_drop_pa', 'frr',
  ] as const;
  let worst = 0;
  let worstField = '';
  for (const field of numeric) {
    const a = computed[field];
    const b = artifact[field];
    const denom = Math.max(Math.abs(b), 1e-12);
    const delta = Math.abs(a - b) / denom;
    if (delta > worst) { worst = delta; worstField = field; }
  }
  const regimeMatches = computed.flow_regime === artifact.flow_regime;
  return { matches: worst <= tolerance && regimeMatches, maxRelativeDelta: worst, worstField };
}
