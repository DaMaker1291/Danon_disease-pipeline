import React, { useRef, useEffect, useState, useCallback, useMemo } from 'react';
import { useArtifact, artifactStamp } from '../utils/artifacts';
import {
  checkPort, computeTelemetry, ETHANOL_DENSITY_KG_M3, ETHANOL_VISCOSITY_PA_S,
  WATER_DENSITY_KG_M3, WATER_VISCOSITY_PA_S,
} from './fluidics';
import type { MicrofluidicConfigData, MicrofluidicsArtifactData } from '../types/artifacts';

const CANVAS_W = 640;
const CANVAS_H = 380;
const Y_JUNCTION_X = 220;
const OUTLET_X = 600;
const CH_TOP = 130;
const CH_BOT = 250;
const N_PARTICLES = 220;

interface Particle {
  x: number; y: number; vx: number; vy: number;
  aqueous: boolean; size: number; phase: number; life: number;
}

/**
 * Y-junction mixing visual with REAL channel hydraulics.
 *
 * The numbers this view shows are computed by `utils/fluidics.ts`, a port of
 * `danon/microfluidics_core.MicrofluidicsCore` — the same correlations the
 * Python engine uses. The previous version displayed a local toy model
 * (`shear = 0.36 * velocity`) and an FRR slider that was not a physical input;
 * FRR is now computed from the two flow rates, as it is in the engine. When
 * the sliders sit at the artifact's own configuration, the port is
 * cross-checked against the Python output and the agreement is shown.
 */
export default function MicrofluidicViewer() {
  const { artifact, loading, error } = useArtifact<MicrofluidicsArtifactData>('microfluidics', 'vector_design');
  const canvasRef = useRef<HTMLCanvasElement>(null!);
  const particlesRef = useRef<Particle[]>([]);

  const [config, setConfig] = useState<MicrofluidicConfigData | null>(null);
  useEffect(() => {
    if (artifact && !config) setConfig(artifact.data.config);
  }, [artifact, config]);

  const telemetry = useMemo(() => (config ? computeTelemetry(config) : null), [config]);
  const stateRef = useRef({ frr: 0.33, vel: 1 });
  stateRef.current = { frr: telemetry?.frr ?? 0.33, vel: telemetry ? telemetry.reynolds_number_mixed : 1 };

  const portCheck = useMemo(() => {
    if (!artifact || !config || !telemetry) return null;
    const same = (Object.keys(artifact.data.config) as (keyof MicrofluidicConfigData)[])
      .every((k) => Math.abs(Number(artifact.data.config[k]) - Number(config[k])) < 1e-12);
    if (!same) return null;
    return checkPort(telemetry, artifact.data.telemetry);
  }, [artifact, config, telemetry]);

  const initParticles = useCallback((): Particle[] => {
    const arr: Particle[] = [];
    const f = stateRef.current.frr;
    for (let i = 0; i < N_PARTICLES; i++) {
      const aqueous = i < N_PARTICLES * (1 - f);
      arr.push({
        x: 10 + Math.random() * 100,
        y: aqueous
          ? CH_TOP + 6 + Math.random() * ((CH_BOT - CH_TOP) / 2 - 10)
          : CH_BOT - 6 - Math.random() * ((CH_BOT - CH_TOP) / 2 - 10),
        vx: (aqueous ? 1.4 : 0.8) + Math.random() * 0.4,
        vy: 0, aqueous, size: 1.5 + Math.random() * 2.5,
        phase: Math.random() * Math.PI * 2, life: Math.random(),
      });
    }
    return arr;
  }, []);

  useEffect(() => { particlesRef.current = initParticles(); }, [initParticles, telemetry?.frr]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    let raf = 0;

    const draw = () => {
      const { frr: f } = stateRef.current;
      const shear = telemetry?.max_shear_stress_pa ?? 0;
      const alert = telemetry ? !telemetry.shear_stress_safe : false;
      const vel = Math.max(0.2, (telemetry?.reynolds_number_mixed ?? 1) * 0.6);
      const midY = (CH_TOP + CH_BOT) / 2;

      const bg = ctx.createLinearGradient(0, 0, 0, CANVAS_H);
      bg.addColorStop(0, '#060b18');
      bg.addColorStop(1, '#0a1120');
      ctx.fillStyle = bg;
      ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);

      ctx.save();
      const chGrad = ctx.createLinearGradient(0, CH_TOP, 0, CH_BOT);
      chGrad.addColorStop(0, 'rgba(59,130,246,0.06)');
      chGrad.addColorStop(0.5, 'rgba(99,102,241,0.03)');
      chGrad.addColorStop(1, 'rgba(239,68,68,0.06)');
      ctx.fillStyle = chGrad;
      ctx.beginPath();
      ctx.moveTo(0, CH_TOP); ctx.lineTo(Y_JUNCTION_X, CH_TOP);
      ctx.lineTo(OUTLET_X, midY - 22); ctx.lineTo(OUTLET_X, midY + 22);
      ctx.lineTo(Y_JUNCTION_X, CH_BOT); ctx.lineTo(0, CH_BOT); ctx.closePath();
      ctx.fill();
      ctx.restore();

      ctx.strokeStyle = 'rgba(148,163,184,0.35)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(0, CH_TOP); ctx.lineTo(Y_JUNCTION_X, CH_TOP); ctx.lineTo(OUTLET_X, midY - 22);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(0, CH_BOT); ctx.lineTo(Y_JUNCTION_X, CH_BOT); ctx.lineTo(OUTLET_X, midY + 22);
      ctx.stroke();

      ctx.fillStyle = '#60a5fa'; ctx.font = '600 11px Inter, sans-serif';
      ctx.fillText('Aqueous (RNA + buffer)', 12, CH_TOP - 10);
      ctx.fillStyle = '#f87171';
      ctx.fillText('Organic (ethanol + lipids)', 12, CH_BOT + 22);
      ctx.fillStyle = '#94a3b8'; ctx.font = '600 10px Inter, sans-serif';
      ctx.fillText(`FRR ${f.toFixed(3)} (computed)`, OUTLET_X - 130, midY - 30);
      ctx.fillText('LNP outlet →', OUTLET_X - 74, midY + 40);

      const parts = particlesRef.current;
      for (let i = 0; i < parts.length; i++) {
        const p = parts[i];
        const halfH = (CH_BOT - CH_TOP) / 2;
        if (p.x < Y_JUNCTION_X) {
          p.y += p.vy;
        } else {
          const mix = Math.min(1, (p.x - Y_JUNCTION_X) / (OUTLET_X - Y_JUNCTION_X));
          p.y += (midY - p.y) * 0.03 * mix;
          p.vy += (Math.random() - 0.5) * 0.15 * (1 + Math.min(shear, 50) * 0.1);
          p.vy *= 0.92;
          p.y += p.vy;
        }
        p.x += p.vx * (0.6 + Math.min(vel, 20) * 0.3);
        p.phase += 0.05;

        if (p.x > OUTLET_X + 40) {
          const aqueous = i < N_PARTICLES * (1 - f);
          p.x = 10 + Math.random() * 100;
          p.aqueous = aqueous;
          p.y = aqueous ? CH_TOP + 6 + Math.random() * (halfH - 10) : CH_BOT - 6 - Math.random() * (halfH - 10);
          p.vx = (aqueous ? 1.4 : 0.8) + Math.random() * 0.4;
        }

        const past = p.x > Y_JUNCTION_X;
        const mixT = past ? Math.min(1, (p.x - Y_JUNCTION_X) / (OUTLET_X - Y_JUNCTION_X)) : 0;
        const col = p.aqueous
          ? `rgba(${59 + mixT * 80}, ${130 - mixT * 40}, ${246 - mixT * 20}, 0.9)`
          : `rgba(${239 - mixT * 100}, ${68 + mixT * 20}, ${68 + mixT * 150}, 0.9)`;

        const glow = 0.6 + Math.sin(p.phase) * 0.25;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fillStyle = col;
        ctx.shadowBlur = 8; ctx.shadowColor = col;
        ctx.globalAlpha = glow;
        ctx.fill();
        ctx.globalAlpha = 1; ctx.shadowBlur = 0;
      }

      ctx.strokeStyle = 'rgba(255,255,255,0.15)';
      ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(Y_JUNCTION_X, CH_TOP); ctx.lineTo(Y_JUNCTION_X, CH_BOT); ctx.stroke();
      ctx.setLineDash([]);

      if (alert) {
        ctx.fillStyle = 'rgba(239,68,68,0.12)';
        ctx.fillRect(Y_JUNCTION_X, CH_TOP - 25, OUTLET_X - Y_JUNCTION_X, CH_BOT - CH_TOP + 50);
        ctx.fillStyle = '#ef4444'; ctx.font = 'bold 13px Inter, sans-serif';
        ctx.fillText('⚠ HIGH SHEAR — capsid damage risk (tau_max >= 50 Pa)', OUTLET_X - 320, CH_TOP - 8);
      }

      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [telemetry]);

  if (loading) return <div className="panel-title">Y-Junction Microfluidic Mixing — loading artifact…</div>;
  if (error || !artifact || !config || !telemetry) {
    return <div className="panel-title" style={{ color: 'var(--warning)' }}>Microfluidics artifact unavailable: {error}</div>;
  }

  const setField = (k: keyof MicrofluidicConfigData, v: number) => setConfig({ ...config, [k]: v });

  return (
    <>
      <div className="panel-title" style={{ marginBottom: 6 }}>
        Y-Junction Microfluidic Mixing — channel hydraulics from the engine's own correlations
      </div>
      <div className="canvas-container" style={{ minHeight: CANVAS_H }}>
        <canvas ref={canvasRef} width={CANVAS_W} height={CANVAS_H} style={{ width: '100%', height: '100%', display: 'block' }} />
      </div>

      <div style={{ marginTop: 12, display: 'flex', gap: 16, flexWrap: 'wrap' }}>
        <div className="input-group" style={{ flex: 1, minWidth: 120 }}>
          <label>Aqueous: {config.aqueous_flow_rate_ul_min.toFixed(0)} µL/min</label>
          <input type="range" min={1} max={60} step={1} value={config.aqueous_flow_rate_ul_min}
            onChange={e => setField('aqueous_flow_rate_ul_min', +e.target.value)} />
        </div>
        <div className="input-group" style={{ flex: 1, minWidth: 120 }}>
          <label>Organic: {config.organic_flow_rate_ul_min.toFixed(0)} µL/min</label>
          <input type="range" min={1} max={60} step={1} value={config.organic_flow_rate_ul_min}
            onChange={e => setField('organic_flow_rate_ul_min', +e.target.value)} />
        </div>
        <div className="input-group" style={{ flex: 1, minWidth: 120 }}>
          <label>Channel height: {config.channel_height_um.toFixed(0)} µm</label>
          <input type="range" min={20} max={400} step={10} value={config.channel_height_um}
            onChange={e => setField('channel_height_um', +e.target.value)} />
        </div>
        <div className="input-group" style={{ flex: 1, minWidth: 120 }}>
          <label>Channel width: {config.channel_width_um.toFixed(0)} µm</label>
          <input type="range" min={50} max={1000} step={10} value={config.channel_width_um}
            onChange={e => setField('channel_width_um', +e.target.value)} />
        </div>
      </div>

      <div className="telemetry-grid" style={{ marginTop: 10 }}>
        {[
          ['Reynolds (mixed)', telemetry.reynolds_number_mixed.toFixed(3), true],
          ['FRR (computed)', telemetry.frr.toFixed(3), true],
          ['Wall shear', `${telemetry.wall_shear_stress_pa.toFixed(3)} Pa`, true],
          ['Max shear', `${telemetry.max_shear_stress_pa.toFixed(3)} Pa`, telemetry.shear_stress_safe],
          ['Mixing efficiency', `${(telemetry.mixing_efficiency * 100).toFixed(1)}%`, telemetry.mixing_efficiency > 0.5],
          ['Péclet', telemetry.peclet_number.toExponential(2), true],
          ['Dean number', telemetry.deans_number.toFixed(3), true],
          ['Pressure drop', `${telemetry.pressure_drop_pa.toFixed(3)} Pa`, telemetry.pressure_drop_pa < 500],
          ['Flow regime', telemetry.flow_regime, telemetry.flow_regime !== 'turbulent'],
          ['Hydraulic diameter', `${telemetry.hydraulic_diameter_um.toFixed(1)} µm`, true],
        ].map(([label, value, safe]) => (
          <div key={String(label)} className="telemetry-item">
            <div className="telemetry-label">{label}</div>
            <div className={`telemetry-value ${safe ? 'telemetry-safe' : 'telemetry-danger'}`}>{value}</div>
          </div>
        ))}
      </div>

      <div style={{ marginTop: 8, fontSize: 11, fontWeight: 600, color: telemetry.shear_stress_safe ? 'var(--success)' : 'var(--danger)' }}>
        {telemetry.shear_stress_safe
          ? `✓ Max shear ${telemetry.max_shear_stress_pa.toFixed(3)} Pa — below the 50 Pa structural-integrity limit used by the engine`
          : `⚠ Max shear ${telemetry.max_shear_stress_pa.toFixed(3)} Pa exceeds the 50 Pa limit — reduce flow rates`}
      </div>

      <div style={{ marginTop: 8, fontSize: 11, color: portCheck ? (portCheck.matches ? 'var(--success)' : 'var(--warning)') : 'var(--text3)' }}>
        {portCheck
          ? (portCheck.matches
            ? `✓ TypeScript port reproduces the Python artifact at this configuration (max relative delta ${portCheck.maxRelativeDelta.toExponential(1)})`
            : `⚠ port diverges from the artifact: ${portCheck.worstField} differs by ${portCheck.maxRelativeDelta.toExponential(2)} — treat the sliders as suspect until fixed`)
          : 'Sliders have moved off the artifact configuration; values are recomputed with the ported correlations (no artifact comparison available).'}
      </div>

      <div style={{ marginTop: 6, color: 'var(--text3)', fontSize: 10.5, lineHeight: 1.6 }}>
        Channel {config.channel_width_um.toFixed(0)} × {config.channel_height_um.toFixed(0)} µm, length {config.channel_length_mm} mm ·
        water {WATER_DENSITY_KG_M3} kg/m³ / {WATER_VISCOSITY_PA_S} Pa·s · ethanol {ETHANOL_DENSITY_KG_M3} kg/m³ / {ETHANOL_VISCOSITY_PA_S} Pa·s.
        Particle motion is a schematic animation; only the numbers are computed.
        <br />{artifactStamp(artifact)}
      </div>
    </>
  );
}
