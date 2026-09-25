"""
Microfluidic Flow Telemetry Module: Navier-Stokes flow equations for a Y-junction
microfluidic channel. Computes Reynolds number, wall shear stress, and mixing
efficiency profiles based on the Organic-to-Aqueous Flow Rate Ratio (FRR).

Governing equations:
  - Continuity: ∇·u = 0 (incompressible flow)
  - Navier-Stokes: ρ(∂u/∂t + u·∇u) = -∇p + μ∇²u
  - Reynolds number: Re = ρUD_h/μ
  - Wall shear stress: τ_w = μ(∂u/∂y)|_wall ≈ 2μU/H (for planar Poiseuille)
  - Mixing efficiency: η = 1 - σ²/σ²_max (intensity-based)
"""
import logging
import numpy as np
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Tuple, Dict

logger = logging.getLogger(__name__)

# Physical constants
WATER_DENSITY_KG_M3 = 997.0  # kg/m³ at 25°C
WATER_VISCOSITY_PA_S = 8.9e-4  # Pa·s at 25°C
ETHANOL_VISCOSITY_PA_S = 1.07e-3  # Pa·s at 25°C
ETHANOL_DENSITY_KG_M3 = 789.0  # kg/m³ at 25°C


class MicrofluidicConfig(BaseModel):
    channel_height_um: float = Field(default=100.0, ge=1.0, le=1000.0)
    channel_width_um: float = Field(default=200.0, ge=1.0, le=5000.0)
    inlet_angle_deg: float = Field(default=45.0, ge=10.0, le=90.0)
    aqueous_flow_rate_ul_min: float = Field(default=10.0, ge=0.1, le=1000.0)
    organic_flow_rate_ul_min: float = Field(default=5.0, ge=0.1, le=1000.0)
    aqueous_density_kg_m3: float = Field(default=WATER_DENSITY_KG_M3, ge=500.0, le=2000.0)
    organic_density_kg_m3: float = Field(default=ETHANOL_DENSITY_KG_M3, ge=500.0, le=2000.0)
    aqueous_viscosity_pa_s: float = Field(default=WATER_VISCOSITY_PA_S, ge=1e-5, le=1.0)
    organic_viscosity_pa_s: float = Field(default=ETHANOL_VISCOSITY_PA_S, ge=1e-5, le=1.0)
    diff_coefficient_m2_s: float = Field(default=1e-10, ge=1e-14, le=1e-6)
    channel_length_mm: float = Field(default=20.0, ge=0.1, le=100.0)
    surface_roughness_um: float = Field(default=0.1, ge=0.0, le=10.0)


class FlowTelemetry(BaseModel):
    reynolds_number_aqueous: float
    reynolds_number_organic: float
    reynolds_number_mixed: float
    wall_shear_stress_pa: float
    max_shear_stress_pa: float
    mixing_efficiency: float
    peclet_number: float
    deans_number: float = 0.0
    pressure_drop_pa: float
    flow_regime: str
    shear_stress_safe: bool
    frr: float
    hydraulic_diameter_um: float


class MicrofluidicsCore:
    """
    Computes Navier-Stokes-derived flow telemetry for a Y-junction microchannel.
    """

    def __init__(self, config: Optional[MicrofluidicConfig] = None):
        self.config = config or MicrofluidicConfig()

    @property
    def H(self) -> float:
        return self.config.channel_height_um * 1e-6

    @property
    def W(self) -> float:
        return self.config.channel_width_um * 1e-6

    @property
    def D_h(self) -> float:
        H, W = self.H, self.W
        return 2 * H * W / (H + W)

    @property
    def A_cross(self) -> float:
        return self.H * self.W

    @property
    def FRR(self) -> float:
        aoq = self.config.aqueous_flow_rate_ul_min
        org = self.config.organic_flow_rate_ul_min
        return org / max(aoq + org, 1e-12)

    def _flow_rate_to_velocity(self, flow_rate_ul_min: float) -> float:
        flow_rate_m3_s = flow_rate_ul_min * 1e-9 / 60.0
        return flow_rate_m3_s / max(self.A_cross, 1e-20)

    def compute_reynolds_number(self, velocity: float, density: float, viscosity: float) -> float:
        return density * velocity * self.D_h / max(viscosity, 1e-20)

    def compute_wall_shear_stress(self, velocity: float, viscosity: float) -> float:
        H = self.H
        if H < 1e-12:
            return 0.0
        return 2.0 * viscosity * velocity / H

    def compute_max_shear_stress(self, velocity: float, viscosity: float) -> float:
        H = self.H
        if H < 1e-12:
            return 0.0
        return 3.0 * viscosity * velocity / H

    def compute_mixing_efficiency(self, c_profile: Optional[np.ndarray] = None) -> float:
        if c_profile is not None and len(c_profile) > 1:
            sigma_sq = np.var(c_profile)
            c_max, c_min = c_profile.max(), c_profile.min()
            sigma_sq_max = 0.25 * (c_max - c_min) ** 2 if c_max > c_min else 1.0
            return float(1.0 - sigma_sq / max(sigma_sq_max, 1e-20))
        return 0.0

    def compute_peclet(self, velocity: float, characteristic_length: float = None) -> float:
        L = characteristic_length or self.D_h
        D = self.config.diff_coefficient_m2_s
        return velocity * L / max(D, 1e-20)

    def compute_pressure_drop(self, velocity: float, viscosity: float) -> float:
        L = self.config.channel_length_mm * 1e-3
        D = self.D_h
        if D < 1e-20:
            return 0.0
        Re = self.compute_reynolds_number(velocity, self.config.aqueous_density_kg_m3, viscosity)
        if Re < 1:
            f = 16.0 / max(Re, 1e-20)
        elif Re < 2000:
            f = 0.079 * Re ** (-0.25)
        else:
            f = 0.079 * Re ** (-0.25)
        return f * (L / D) * 0.5 * self.config.aqueous_density_kg_m3 * velocity ** 2

    def simulate(self) -> FlowTelemetry:
        cfg = self.config
        v_aq = self._flow_rate_to_velocity(cfg.aqueous_flow_rate_ul_min)
        v_org = self._flow_rate_to_velocity(cfg.organic_flow_rate_ul_min)
        total_flow = cfg.aqueous_flow_rate_ul_min + cfg.organic_flow_rate_ul_min
        frac_aq = cfg.aqueous_flow_rate_ul_min / max(total_flow, 1e-12)
        mixed_density = cfg.aqueous_density_kg_m3 * frac_aq + cfg.organic_density_kg_m3 * (1 - frac_aq)
        mixed_viscosity = cfg.aqueous_viscosity_pa_s * frac_aq + cfg.organic_viscosity_pa_s * (1 - frac_aq)
        v_mix = self._flow_rate_to_velocity(total_flow)

        Re_aq = self.compute_reynolds_number(v_aq, cfg.aqueous_density_kg_m3, cfg.aqueous_viscosity_pa_s)
        Re_org = self.compute_reynolds_number(v_org, cfg.organic_density_kg_m3, cfg.organic_viscosity_pa_s)
        Re_mix = self.compute_reynolds_number(v_mix, mixed_density, mixed_viscosity)

        tau_w = self.compute_wall_shear_stress(v_mix, mixed_viscosity)
        tau_max = self.compute_max_shear_stress(v_mix, mixed_viscosity)

        Pe = self.compute_peclet(v_mix)
        delta_p = self.compute_pressure_drop(v_mix, mixed_viscosity)

        c_approx = np.linspace(1 - self.FRR, self.FRR, 50)
        eta = self.compute_mixing_efficiency(c_approx)

        if Re_mix < 1:
            regime = "creeping (Stokes)"
        elif Re_mix < 2000:
            regime = "laminar"
        elif Re_mix < 4000:
            regime = "transitional"
        else:
            regime = "turbulent"

        R_curve = self.D_h / (2.0 * np.sin(np.radians(cfg.inlet_angle_deg) / 2.0) + 1e-20)
        Dn = Re_mix * np.sqrt(self.D_h / max(R_curve, 1e-20))

        return FlowTelemetry(
            reynolds_number_aqueous=float(Re_aq),
            reynolds_number_organic=float(Re_org),
            reynolds_number_mixed=float(Re_mix),
            wall_shear_stress_pa=float(tau_w),
            max_shear_stress_pa=float(tau_max),
            mixing_efficiency=float(eta),
            peclet_number=float(Pe),
            deans_number=float(Dn),
            pressure_drop_pa=float(delta_p),
            flow_regime=regime,
            shear_stress_safe=tau_max < 50.0,
            frr=float(self.FRR),
            hydraulic_diameter_um=float(self.D_h * 1e6),
        )

    def _concentration_profile(self, n_points: int = 100) -> np.ndarray:
        y = np.linspace(-self.H / 2, self.H / 2, n_points)
        frac_org = self.FRR
        c0 = np.where(y < 0, 1.0 - frac_org, frac_org)
        eta = self.config.diff_coefficient_m2_s
        v = self._flow_rate_to_velocity(
            self.config.aqueous_flow_rate_ul_min + self.config.organic_flow_rate_ul_min
        )
        L = self.config.channel_length_mm * 1e-3
        t_diff = L / max(v, 1e-20)
        sigma = np.sqrt(2 * eta * t_diff) if t_diff > 0 else 0.01
        H_ch = self.H
        if sigma < H_ch / 2:
            kernel = np.exp(-y ** 2 / (2 * sigma ** 2 + 1e-20))
            kernel /= max(kernel.sum(), 1e-20)
            return np.convolve(c0, kernel, mode="same")
        return c0

    def get_concentration_profile(self, n_points: int = 100) -> List[float]:
        return self._concentration_profile(n_points).tolist()

    def get_velocity_profile(self, n_points: int = 50) -> Dict:
        y = np.linspace(-self.H / 2, self.H / 2, n_points)
        v_total = self._flow_rate_to_velocity(
            self.config.aqueous_flow_rate_ul_min + self.config.organic_flow_rate_ul_min
        )
        u = 1.5 * v_total * (1.0 - (2 * y / self.H) ** 2)
        return {
            "positions_um": [float(yp * 1e6) for yp in y],
            "velocities_m_s": [float(up) for up in u],
        }
