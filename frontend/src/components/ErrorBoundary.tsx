import React, { Component, type ReactNode } from 'react';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
  label?: string;
  /**
   * When this value changes, a failed boundary clears itself.
   *
   * Without it a crash on one screen stuck to every screen opened afterwards
   * until the visitor pressed Retry: observed when leaving a WebGL tab (the
   * three.js renderer disposing during unmount) and landing on a chart tab,
   * which then showed "Component Error" for a fault it had no part in.
   */
  resetKey?: unknown;
}

interface State {
  hasError: boolean;
  error: Error | null;
  seenKey?: unknown;
}

export default class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false, error: null, seenKey: this.props.resetKey };

  static getDerivedStateFromError(error: Error): Partial<State> {
    return { hasError: true, error };
  }

  static getDerivedStateFromProps(props: Props, state: State): Partial<State> | null {
    if (props.resetKey !== state.seenKey) {
      return { hasError: false, error: null, seenKey: props.resetKey };
    }
    return null;
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    console.error(
      `[ErrorBoundary${this.props.label ? ` ${this.props.label}` : ''}]`,
      error,
      info.componentStack,
    );
  }

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) return this.props.fallback;
      return (
        <div style={{ padding: 16, color: '#f87171', background: '#1a0a0a', borderRadius: 8, border: '1px solid #7f1d1d', margin: 8 }}>
          <strong>{this.props.label ? `${this.props.label} failed to render` : 'This panel failed to render'}</strong>
          <p style={{ fontSize: 12, marginTop: 4, color: '#fca5a5' }}>{this.state.error?.message}</p>
          <p style={{ fontSize: 11, marginTop: 6, color: '#94a3b8' }}>
            The rest of the app is unaffected. This clears automatically when you open another view.
          </p>
          <button
            onClick={() => this.setState({ hasError: false, error: null })}
            style={{ marginTop: 8, padding: '4px 12px', background: '#7f1d1d', color: '#fff', border: 'none', borderRadius: 4, cursor: 'pointer', fontSize: 12 }}
          >
            Retry
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
