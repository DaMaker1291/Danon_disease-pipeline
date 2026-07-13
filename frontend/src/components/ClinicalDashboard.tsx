import React, { useState } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, AreaChart, Area,
} from 'recharts';

const MUTATION_TYPES = ['Null', 'Splice Site', 'Missense', 'Partial Deletion'] as const;

function generateClinicalData(mutationType: string) {
  const data: { month: number; lvmi: number; survival: number; survLow: number; survHigh: number }[] = [];
  const baseLVMI = mutationType === 'Null' ? 95 : mutationType === 'Splice Site' ? 88 : mutationType === 'Missense' ? 78 : 82;
  const baseSurvival = mutationType === 'Null' ? 0.5 : mutationType === 'Splice Site' ? 0.6 : mutationType === 'Missense' ? 0.8 : 0.7;

  for (let m = 0; m <= 60; m += 3) {
    const lvmiDecay = baseLVMI * Math.exp(-0.015 * m) * (1 - 0.3 * (1 - Math.exp(-m / 24)));
    const surv = baseSurvival * Math.exp(-0.02 * m);
    data.push({
      month: m,
      lvmi: +lvmiDecay.toFixed(1),
      survival: +surv.toFixed(3),
      survLow: +Math.max(0, surv - 0.08).toFixed(3),
      survHigh: +Math.min(1, surv + 0.08).toFixed(3),
    });
  }
  return data;
}

export default function ClinicalDashboard() {
  const [selectedMutation, setSelectedMutation] = useState<string>('Null');
  const data = generateClinicalData(selectedMutation);

  return (
    <div className="panel">
      <div className="panel-title">🏥 Clinical Outcome Projections by Danon Mutation Type</div>

      <div style={{ display: 'flex', gap: 8, marginBottom: 12, flexWrap: 'wrap' }}>
        {MUTATION_TYPES.map(mt => (
          <button
            key={mt}
            className={`nav-btn ${selectedMutation === mt ? 'active' : ''}`}
            onClick={() => setSelectedMutation(mt)}
            style={{ fontSize: 11, padding: '4px 12px' }}
          >
            {mt}
          </button>
        ))}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, height: 300 }}>
        <div>
          <div className="panel-title" style={{ fontSize: 11, marginBottom: 4 }}>LVMI Decay Prediction</div>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={data} margin={{ top: 5, right: 10, bottom: 5, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="month" tick={{ fill: '#94a3b8', fontSize: 10 }} label={{ value: 'Months', fill: '#94a3b8', fontSize: 10 }} />
              <YAxis domain={[30, 100]} tick={{ fill: '#94a3b8', fontSize: 10 }} label={{ value: 'LVMI (g/m²)', angle: -90, fill: '#94a3b8', fontSize: 10 }} />
              <Tooltip contentStyle={{ background: '#1f2937', border: '1px solid #1e293b', fontSize: 11 }} />
              <Line type="monotone" dataKey="lvmi" stroke="#3b82f6" strokeWidth={2} dot={false} name="LVMI Predicted" />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div>
          <div className="panel-title" style={{ fontSize: 11, marginBottom: 4 }}>Kaplan-Meier Survival</div>
          <ResponsiveContainer width="100%" height={250}>
            <AreaChart data={data} margin={{ top: 5, right: 10, bottom: 5, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="month" tick={{ fill: '#94a3b8', fontSize: 10 }} label={{ value: 'Months', fill: '#94a3b8', fontSize: 10 }} />
              <YAxis domain={[0, 1]} tick={{ fill: '#94a3b8', fontSize: 10 }} label={{ value: 'Survival Probability', angle: -90, fill: '#94a3b8', fontSize: 10 }} />
              <Tooltip contentStyle={{ background: '#1f2937', border: '1px solid #1e293b', fontSize: 11 }} />
              <Area type="monotone" dataKey="survival" stroke="#22c55e" fill="#22c55e" fillOpacity={0.2} strokeWidth={2} name="Survival" />
              <Area type="monotone" dataKey="survLow" stroke="#22c55e" strokeWidth={0} fill="#22c55e" fillOpacity={0.05} name="95% CI Lower" />
              <Area type="monotone" dataKey="survHigh" stroke="#22c55e" strokeWidth={0} fill="#22c55e" fillOpacity={0.05} name="95% CI Upper" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div style={{ marginTop: 8, fontSize: 11, color: '#94a3b8', display: 'flex', gap: 16 }}>
        <span>Primary endpoint: Cardiomyocyte LAMP2 Protein Expression</span>
        <span>Secondary: LVMI Reduction</span>
      </div>
    </div>
  );
}
