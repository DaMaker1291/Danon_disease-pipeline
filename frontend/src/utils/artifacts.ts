import { useEffect, useState } from 'react';
import type { AppArtifact } from '../types/artifacts';

/**
 * Which public folder an artifact key is served from.
 *
 * The tumour-twin artifacts sit in `public/artifacts/`; the vector-design
 * study's sit in `public/vector_design/`, mirroring the split of its code into
 * `src/vector_design/`. Both are written by `danon/app_artifacts.py`.
 */
export type ArtifactDir = 'artifacts' | 'vector_design';

/**
 * Load a generated artifact from the app's own base path.
 *
 * Artifacts are written by `danon/app_artifacts.py`, so every view in this app
 * renders recorded engine output with no backend running. A missing artifact
 * throws: a view that invents a plausible number when its data is absent is
 * worse than one that says it has no data.
 */
export async function loadArtifact<T>(
  key: string,
  dir: ArtifactDir = 'artifacts',
): Promise<AppArtifact<T>> {
  const url = `${import.meta.env.BASE_URL}${dir}/${key}.json`;
  const res = await fetch(url, { cache: 'no-store' });
  if (!res.ok) {
    throw new Error(
      `artifact "${key}" not found at ${url} (${res.status}). Generate it with:\n` +
        `python -c "from danon.app_artifacts import write_app_artifacts; write_app_artifacts()"`,
    );
  }
  const payload = (await res.json()) as AppArtifact<T>;
  if (!payload?.data) throw new Error(`artifact "${key}" has no data block`);
  return payload;
}

export interface ArtifactState<T> {
  artifact: AppArtifact<T> | null;
  loading: boolean;
  error: string | null;
}

/** React hook for an artifact: loading, error, and the payload itself. */
export function useArtifact<T>(key: string, dir: ArtifactDir = 'artifacts'): ArtifactState<T> {
  const [artifact, setArtifact] = useState<AppArtifact<T> | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    setLoading(true);
    loadArtifact<T>(key, dir)
      .then((a) => { if (live) { setArtifact(a); setError(null); } })
      .catch((e: Error) => { if (live) { setArtifact(null); setError(e.message); } })
      .finally(() => { if (live) setLoading(false); });
    return () => { live = false; };
  }, [key, dir]);

  return { artifact, loading, error };
}

/**
 * One-line provenance strip every artifact-backed view renders, so the source
 * of its numbers is always on screen.
 */
export function artifactStamp(artifact: AppArtifact<unknown>): string {
  const modelled = Object.entries(artifact.provenance)
    .filter(([, v]) => v === 'unknown')
    .map(([k]) => k);
  const failed = Object.entries(artifact.provenance)
    .filter(([, v]) => v === 'proven_failed')
    .map(([k]) => k);
  const parts = [`source: ${artifact.source}`, `generated ${artifact.generated_utc}`];
  if (modelled.length) parts.push(`modelled/unknown: ${modelled.join(', ')}`);
  if (failed.length) parts.push(`NOT established: ${failed.join(', ')}`);
  return parts.join(' · ');
}
