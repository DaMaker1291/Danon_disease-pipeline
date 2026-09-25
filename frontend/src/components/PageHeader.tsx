import React from 'react';

/**
 * The strip at the top of every tab.
 *
 * It answers three questions before any content is read: which of the two
 * studies am I in, what is this screen for, and where do its numbers come
 * from. The provenance line is required rather than optional -- a view whose
 * source is not stated is the failure mode this app keeps being fixed for.
 */
export interface PageHeaderProps {
  title: string;
  subject: string;
  purpose: string;
  /** Where the numbers on this screen were produced. */
  source: string;
  /** What class of number the screen shows (see nav.ts::ScreenStatus). */
  status: string;
  /** Optional honest caveat shown as a chip, e.g. "needs the local API". */
  caveat?: string;
}

export default function PageHeader({ title, subject, purpose, source, status, caveat }: PageHeaderProps) {
  return (
    <header className={`page-header status-${status.replace(/[^a-z]+/gi, '-')}`}>
      <div className="page-header-top">
        <h2>{title}</h2>
        <span className="chip">{subject}</span>
        <span className="chip chip-real">{status}</span>
        {caveat && <span className="chip chip-warn">{caveat}</span>}
      </div>
      <p className="page-header-purpose">{purpose}</p>
      <p className="page-header-source">source: {source}</p>
    </header>
  );
}
