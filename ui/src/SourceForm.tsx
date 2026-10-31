import { useId } from 'react';
import type { CaptureView } from './useCapture';

export function SourceForm({ capture }: { capture: CaptureView }) {
  const id = useId();
  return (
    <section className="source" aria-label="Stream source">
      <label htmlFor={`${id}-url`}>Stream address</label>
      <div className="row">
        <input
          id={`${id}-url`} type="text" inputMode="url" spellCheck={false} autoComplete="off"
          placeholder="https://www.youtube.com/watch?v=…  or  https://www.twitch.tv/lolesports"
          value={capture.on && capture.state?.source ? capture.state.source : capture.url} disabled={capture.on} onChange={e => capture.setUrl(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter' && !capture.on) { e.preventDefault(); void capture.start(); } }}
        />
        {capture.on
          ? <button type="button" onClick={() => void capture.stop()} disabled={capture.busy}>Stop reading</button>
          : <button type="button" className="primary" onClick={() => void capture.start()} disabled={capture.busy}>Start reading</button>}
      </div>
      {capture.problem && <p className="error" role="alert">{capture.problem}</p>}
    </section>
  );
}
