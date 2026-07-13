import React from 'react';
import {
  ScatterChart, Scatter, XAxis, YAxis, ZAxis,
  CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from 'recharts';
import type { ParetoPoint } from '../types';

interface Props {
  candidates: ParetoPoint[];
}

export default function ParetoFrontChart({ candidates }: Props) {
  const data = candidates.map(c => ({
    cardiac: +(c.cardiacTropism * 100).toFixed(1),
    hepatic: +(c.hepaticAvoidance * 100).toFixed(1),
    immune: +(c.immuneEvasion * 100).toFixed(1),
    lamp2b: +(c.lamp2bExpression * 100).toFixed(1),
    id: c.candidateId,
  }));

  return (
    <div className="panel">
      <div className="panel-title">📊 NSGA-II Pareto Front — 6 Competing Objectives</div>
      <div className="chart-container">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis
              dataKey="cardiac"
              name="Cardiac Tropism"
              domain={[0, 100]}
              tick={{ fill: '#94a3b8', fontSize: 11 }}
              label={{ value: 'Cardiac Tropism (%)', position: 'bottom', fill: '#94a3b8', fontSize: 11 }}
            />
            <YAxis
              dataKey="hepatic"
              name="Hepatic Avoidance"
              domain={[0, 100]}
              tick={{ fill: '#94a3b8', fontSize: 11 }}
              label={{ value: 'Hepatic Avoidance (%)', angle: -90, position: 'left', fill: '#94a3b8', fontSize: 11 }}
            />
            <ZAxis dataKey="lamp2b" range={[40, 300]} name="LAMP2B Expression" />
            <Tooltip
              contentStyle={{ background: '#1f2937', border: '1px solid #1e293b', borderRadius: 8, fontSize: 12 }}
              labelStyle={{ color: '#e2e8f0' }}
              formatter={(value: number) => [`${value}%`]}
            />
            <Legend
              wrapperStyle={{ fontSize: 11, color: '#94a3b8' }}
              formatter={() => 'Pareto-optimized candidates'}
            />
            <Scatter
              name="Pareto-optimized candidates"
              data={data}
              fill="#3b82f6"
              opacity={0.7}
              stroke="#60a5fa"
              strokeWidth={1}
            />
          </ScatterChart>
        </ResponsiveContainer>
      </div>
      <div style={{ marginTop: 8, display: 'flex', gap: 16, fontSize: 11, color: '#94a3b8' }}>
        <span>Objectives: Cardiac Tropism, Hepatic Avoidance, Immune Evasion, LAMP2B Expression, Promoter Score, miRNA Score</span>
      </div>
    </div>
  );
}
