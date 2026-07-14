import React, { useRef, useEffect, useState, useCallback } from 'react';

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

export default function MicrofluidicViewer() {
  const canvasRef = useRef<HTMLCanvasElement>(null!);
  const particlesRef = useRef<Particle[]>([]);
  const [frr, setFrr] = useState(0.33);
  const [flowRateAq, setFlowRateAq] = useState(10);
  const [flowRateOrg, setFlowRateOrg] = useState(5);
  const stateRef = useRef({ frr, flowRateAq, flowRateOrg });
  stateRef.current = { frr, flowRateAq, flowRateOrg };

  const totalFlow = flowRateAq + flowRateOrg;
  const velocity = totalFlow / 15;
  const maxShear = 0.36 * velocity;
  const shearAlert = maxShear > 5.0;

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

  useEffect(() => { particlesRef.current = initParticles(); }, [initParticles, frr]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    let raf = 0;

    const draw = () => {
      const { frr: f, flowRateAq: fa, flowRateOrg: fo } = stateRef.current;
      const vel = (fa + fo) / 15;
      const shear = 0.36 * vel;
      const alert = shear > 5.0;
      const midY = (CH_TOP + CH_BOT) / 2;

      // Clear with subtle gradient
      const bg = ctx.createLinearGradient(0, 0, 0, CANVAS_H);
      bg.addColorStop(0, '#060b18');
      bg.addColorStop(1, '#0a1120');
      ctx.fillStyle = bg;
      ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);

      // Channel body glow
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

      // Channel walls
      ctx.strokeStyle = 'rgba(148,163,184,0.35)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(0, CH_TOP); ctx.lineTo(Y_JUNCTION_X, CH_TOP); ctx.lineTo(OUTLET_X, midY - 22);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(0, CH_BOT); ctx.lineTo(Y_JUNCTION_X, CH_BOT); ctx.lineTo(OUTLET_X, midY + 22);
      ctx.stroke();

      // Labels
      ctx.fillStyle = '#60a5fa'; ctx.font = '600 11px Inter, sans-serif';
      ctx.fillText('Aqueous (RNA + buffer)', 12, CH_TOP - 10);
      ctx.fillStyle = '#f87171';
      ctx.fillText('Organic (ethanol + lipids)', 12, CH_BOT + 22);
      ctx.fillStyle = '#94a3b8'; ctx.font = '600 10px Inter, sans-serif';
      ctx.fillText(`FRR ${f.toFixed(2)}`, OUTLET_X - 70, midY - 30);
      ctx.fillText('LNP outlet →', OUTLET_X - 74, midY + 40);

      // Update + draw particles
      const parts = particlesRef.current;
      for (let i = 0; i < parts.length; i++) {
        const p = parts[i];
        const halfH = (CH_BOT - CH_TOP) / 2;
        if (p.x < Y_JUNCTION_X) {
          p.y += p.vy;
        } else {
          const mix = Math.min(1, (p.x - Y_JUNCTION_X) / (OUTLET_X - Y_JUNCTION_X));
          p.y += (midY - p.y) * 0.03 * mix;
          p.vy += (Math.random() - 0.5) * 0.15 * (1 + shear * 0.1);
          p.vy *= 0.92;
          p.y += p.vy;
        }
        p.x += p.vx * (0.6 + vel * 0.3);
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
        // color transitions toward purple (mixed LNP) after junction
        let col: string;
        if (p.aqueous) col = `rgba(${59 + mixT * 80}, ${130 - mixT * 40}, ${246 - mixT * 20}, 0.9)`;
        else col = `rgba(${239 - mixT * 100}, ${68 + mixT * 20}, ${68 + mixT * 150}, 0.9)`;

        const glow = 0.6 + Math.sin(p.phase) * 0.25;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fillStyle = col;
        ctx.shadowBlur = 8; ctx.shadowColor = col;
        ctx.globalAlpha = glow;
        ctx.fill();
        ctx.globalAlpha = 1; ctx.shadowBlur = 0;
      }

      // Y-junction marker
      ctx.strokeStyle = 'rgba(255,255,255,0.15)';
      ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(Y_JUNCTION_X, CH_TOP); ctx.lineTo(Y_JUNCTION_X, CH_BOT); ctx.stroke();
      ctx.setLineDash([]);

      // Shear alert overlay
      if (alert) {
        ctx.fillStyle = 'rgba(239,68,68,0.12)';
        ctx.fillRect(Y_JUNCTION_X, CH_TOP - 25, OUTLET_X - Y_JUNCTION_X, CH_BOT - CH_TOP + 50);
        ctx.fillStyle = '#ef4444'; ctx.font = 'bold 13px Inter, sans-serif';
        ctx.fillText('⚠ HIGH SHEAR — capsid damage risk', OUTLET_X - 210, CH_TOP - 8);
      }

      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, []);

  return (
    <>
      <div className="panel-title" style={{ marginBottom: 6 }}>
        💧 Y-Junction Microfluidic Mixing — Real-Time Particle Flow
      </div>
      <div className="canvas-container" style={{ minHeight: CANVAS_H }}>
        <canvas ref={canvasRef} width={CANVAS_W} height={CANVAS_H} style={{ width: '100%', height: '100%', display: 'block' }} />
      </div>
      <div style={{ marginTop: 12, display: 'flex', gap: 16, flexWrap: 'wrap' }}>
        <div className="input-group" style={{ flex: 1, minWidth: 120 }}>
          <label>FRR (organic/total): {frr.toFixed(2)}</label>
          <input type="range" min={0.1} max={0.8} step={0.01} value={frr} onChange={e => setFrr(+e.target.value)} />
        </div>
        <div className="input-group" style={{ flex: 1, minWidth: 120 }}>
          <label>Aqueous: {flowRateAq} µL/min</label>
          <input type="range" min={1} max={60} step={1} value={flowRateAq} onChange={e => setFlowRateAq(+e.target.value)} />
        </div>
        <div className="input-group" style={{ flex: 1, minWidth: 120 }}>
          <label>Organic: {flowRateOrg} µL/min</label>
          <input type="range" min={1} max={60} step={1} value={flowRateOrg} onChange={e => setFlowRateOrg(+e.target.value)} />
        </div>
      </div>
      <div style={{ marginTop: 4, fontSize: 11, color: shearAlert ? '#ef4444' : '#22c55e', fontWeight: 600 }}>
        {shearAlert
          ? '⚠ Shear stress exceeds safe threshold — reduce flow rates to protect payload'
          : `✓ Max shear ${maxShear.toFixed(2)} Pa — within safe range for AAV/LNP structural integrity`}
      </div>
    </>
  );
}
