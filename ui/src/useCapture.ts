import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ApiError, type CaptureApi } from './api';
import { isLive, sourceProblem, statusPill } from './logic';
import type { CaptureState, CaptureTeams } from './types';

const LAST_URL_KEY = 'lol-live-capture.url';

export function useCapture(api: CaptureApi) {
  const [state, setState] = useState<CaptureState | null>(null);
  const [url, setUrlState] = useState(() => localStorage.getItem(LAST_URL_KEY) ?? '');
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);
  const [down, setDown] = useState(false);
  const alive = useRef(true);

  useEffect(() => {
    alive.current = true;
    let streamed = false; // an event that arrived first is newer than the snapshot
    api.state().then(s => { if (alive.current && !streamed) setState(s); }).catch(() => { if (alive.current) setDown(true); });
    const sub = api.subscribe(s => { streamed = true; setState(s); setDown(false); }, () => setDown(true));
    return () => { alive.current = false; sub.close(); };
  }, [api]);

  const setUrl = useCallback((value: string) => { setUrlState(value); localStorage.setItem(LAST_URL_KEY, value); setProblem(null); }, []);

  const fail = (e: unknown) => setProblem(e instanceof ApiError ? e.message : 'Something went wrong');

  const start = useCallback(async (teams?: CaptureTeams) => {
    const bad = sourceProblem(url);
    if (bad) { setProblem(bad); return; }
    setBusy(true); setProblem(null);
    try { await api.start(url.trim(), teams); } catch (e) { fail(e); } finally { setBusy(false); }
  }, [api, url]);

  const stop = useCallback(async () => {
    setBusy(true);
    try { await api.stop(); } catch (e) { fail(e); } finally { setBusy(false); }
  }, [api]);

  const on = isLive(state);
  const pill = useMemo(() => statusPill(down ? { status: 'error', frame_age_ms: null, error: 'server unreachable' } : state), [state, down]);
  return { state, url, setUrl, start, stop, busy, problem, on, pill };
}

export type CaptureView = ReturnType<typeof useCapture>;
