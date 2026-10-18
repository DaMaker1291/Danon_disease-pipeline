/**
 * Presentational primitives shared by the tab views: a titled panel, a bordered
 * table, and the numeric formatter the tables use to stay tabular-aligned.
 *
 * These exist because four views had each grown their own near-identical copy of
 * `Panel` and two their own table. This module is the single home going forward;
 * `Day365Outcome`, `EvolutionPlayer` and `TumorBoard` still carry their own local
 * copies and are NOT migrated here yet -- that is a separate, mechanical pass, and
 * changing those views is out of scope for the change that introduced this file.
 */

import React from 'react';

/** Placeholder for a quantity the record does not carry, never a zero. */
export function num(x: number | null | undefined, d = 1): string {
  if (x === null || x === undefined || Number.isNaN(x)) return '—';
  return x.toFixed(d).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
}

export function Panel({
  title, sub, children,
}: { title: string; sub?: string; children: React.ReactNode }) {
  return (
    <div className="panel" style={{ marginBottom: 14 }}>
      <div className="panel-title">{title}</div>
      {sub && (
        <div style={{ color: 'var(--text2)', fontSize: 11.5, margin: '4px 0 12px', lineHeight: 1.55 }}>
          {sub}
        </div>
      )}
      {children}
    </div>
  );
}

export function DataTable({
  head, rows, minWidth = 640,
}: { head: string[]; rows: React.ReactNode[][]; minWidth?: number }) {
  return (
    <div style={{ overflowX: 'auto' }}>
      <table
        className="mini-table"
        style={{ width: '100%', minWidth, fontSize: 12.5, fontVariantNumeric: 'tabular-nums' }}
      >
        <thead>
          <tr>
            {head.map((h, i) => (
              <th
                key={h}
                style={{
                  textAlign: i === 0 ? 'left' : 'right',
                  color: 'var(--text3)',
                  fontWeight: 600,
                  fontSize: 10.5,
                  textTransform: 'uppercase',
                  letterSpacing: '.7px',
                  padding: '6px 8px',
                }}
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, ri) => (
            <tr key={ri}>
              {r.map((c, ci) => (
                <td
                  key={ci}
                  style={{
                    textAlign: ci === 0 ? 'left' : 'right',
                    padding: '6px 8px',
                    borderTop: '1px solid var(--border)',
                    color: ci === 0 ? 'var(--text2)' : 'var(--text)',
                  }}
                >
                  {c}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
