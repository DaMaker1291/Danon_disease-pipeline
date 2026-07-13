import React from 'react';
import type { FlowTelemetry } from '../types';

interface Props {
  telemetry: FlowTelemetry;
  title?: string;
}

export default function TelemetryPanel({ telemetry, title = 'System Telemetry' }: Props) {
  const items = [
    { label: 'Reynolds (aq.)', value: telemetry.reynoldsNumberAqueous.toFixed(2), safe: true },
    { label: 'Reynolds (org.)', value: telemetry.reynoldsNumberOrganic.toFixed(2), safe: true },
    { label: 'Reynolds (mixed)', value: telemetry.reynoldsNumberMixed.toFixed(2), safe: true },
    { label: 'Wall Shear Stress', value: telemetry.wallShearStressPa.toFixed(2) + ' Pa', safe: telemetry.wallShearStressPa < 50 },
    { label: 'Max Shear Stress', value: telemetry.maxShearStressPa.toFixed(2) + ' Pa', safe: telemetry.shearStressSafe },
    { label: 'Mixing Efficiency', value: (telemetry.mixingEfficiency * 100).toFixed(0) + '%', safe: telemetry.mixingEfficiency > 0.5 },
    { label: 'Péclet Number', value: telemetry.pecletNumber.toExponential(1), safe: true },
    { label: 'Pressure Drop', value: telemetry.pressureDropPa.toFixed(1) + ' Pa', safe: telemetry.pressureDropPa < 500 },
    { label: 'Flow Regime', value: telemetry.flowRegime, safe: telemetry.flowRegime !== 'turbulent' },
    { label: 'FRR', value: telemetry.frr.toFixed(3), safe: true },
  ];

  return (
    <>
      <div className="panel-title">{title === 'System Telemetry' ? '📡 ' : '📊 '}{title}</div>
      <div className="telemetry-grid">
        {items.map(item => (
          <div key={item.label} className="telemetry-item">
            <div className="telemetry-label">{item.label}</div>
            <div className={`telemetry-value ${item.safe ? 'telemetry-safe' : 'telemetry-danger'}`}>
              {item.value}
            </div>
          </div>
        ))}
      </div>
    </>
  );
}
