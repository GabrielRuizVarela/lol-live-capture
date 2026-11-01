import { useEffect, useMemo, useState } from 'react';
import { fetchChampions, httpApi, type CaptureApi } from './api';
import { Board } from './Board';
import { createDemoApi } from './demo';
import { lolesportsBadge } from './logic';
import { SourceForm } from './SourceForm';
import type { Champion } from './types';
import { useCapture } from './useCapture';

const demo = new URLSearchParams(window.location.search).has('demo');

export function App({ api }: { api?: CaptureApi }) {
  const [champions, setChampions] = useState<Champion[]>([]);
  const names = useMemo(() => new Map(champions.map(c => [c.id, c])), [champions]);
  const nameOf = (id: string) => names.get(id)?.name ?? id;
  const chosen = useMemo(() => api ?? (demo ? createDemoApi(Number(new URLSearchParams(window.location.search).get('step')) || 2500) : httpApi), [api]);
  const capture = useCapture(chosen);

  useEffect(() => { fetchChampions().then(setChampions).catch(() => setChampions([])); }, []);

  const { state } = capture;
  const badge = lolesportsBadge(state?.lolesports, nameOf);
  const teams = state?.teams;

  return (
    <main className="app">
      <header className="top">
        <span className="mark" aria-hidden="true"><i /><i /><i /></span>
        <div className="brand">
          <h1>lol-live-capture</h1>
          <p className="tag">Live pick &amp; ban reading from a broadcast stream</p>
        </div>
        <span className={`pill is-${capture.pill.tone}`} role="status" data-testid="pill"><i aria-hidden="true" />{capture.pill.text}</span>
        {demo && <span className="demo-tag">demo, no stream</span>}
      </header>

      <SourceForm capture={capture} />

      {state?.audio?.enabled && (
        <p className="quiet" data-testid="audio">Audio on{state.audio.llm ? ' · AI' : ''}{state.audio.last_segment ? ` — “${state.audio.last_segment}”` : ''}</p>
      )}
      {badge && <p className={`notice is-${badge.tone}`} role="status" data-testid="lolesports">{badge.text}</p>}

      <Board
        actions={state?.actions ?? []}
        hover={state?.hover ?? null}
        champions={names}
        blueLabel={teams?.blue.name || teams?.blue.tricode || 'Blue'}
        redLabel={teams?.red.name || teams?.red.tricode || 'Red'}
      />

      {!!state?.warnings.length && (
        <ul className="warnings" aria-label="Warnings">{state.warnings.map((w, i) => <li key={i}>{w}</li>)}</ul>
      )}

      <footer className="foot">
        <span>Game {state?.game_index ?? 1}{state?.stream_t != null ? ` · stream ${Math.round(state.stream_t)} s` : ''}</span>
        <a className="site" href="https://lol-draft.pages.dev" target="_blank" rel="noreferrer">Worlds 2026 drafts on lol-draft.pages.dev</a>
        <button type="button" disabled={!state?.actions.length} onClick={() => download(state?.actions ?? [])}>Download actions (JSON)</button>
      </footer>
    </main>
  );
}

function download(actions: unknown[]) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(actions, null, 2)], { type: 'application/json' }));
  const a = document.createElement('a');
  a.href = url; a.download = 'draft-actions.json'; a.click();
  URL.revokeObjectURL(url);
}
