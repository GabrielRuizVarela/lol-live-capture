import type { CaptureState, CaptureTeams, Champion } from './types';

export class ApiError extends Error {
  constructor(message: string, readonly status: number, readonly code?: string) { super(message); }
}

export interface Subscription { close(): void }
export interface CaptureApi {
  start(source: string, teams?: CaptureTeams): Promise<unknown>;
  stop(): Promise<unknown>;
  state(): Promise<CaptureState>;
  /** `onState` for every `event: state`; `onDown` when the connection drops (EventSource retries by itself). */
  subscribe(onState: (s: CaptureState) => void, onDown: () => void): Subscription;
}

const ROOT = '/api/capture';

async function post(path: string, body?: unknown): Promise<unknown> {
  let res: Response;
  try {
    res = await fetch(`${ROOT}${path}`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch { throw new ApiError('Cannot reach the server', 0); }
  const data = await res.json().catch(() => undefined) as { error?: { message?: string; code?: string } } | undefined;
  if (!res.ok) throw new ApiError(data?.error?.message ?? `Request failed (${res.status})`, res.status, data?.error?.code);
  return data;
}

export const httpApi: CaptureApi = {
  start: (source, teams) => post('/start', teams ? { source, teams } : { source }),
  stop: () => post('/stop'),
  async state() {
    let res: Response;
    try { res = await fetch(`${ROOT}/state`); } catch { throw new ApiError('Cannot reach the server', 0); }
    if (!res.ok) throw new ApiError(`Request failed (${res.status})`, res.status);
    return res.json() as Promise<CaptureState>;
  },
  subscribe(onState, onDown) {
    if (typeof EventSource === 'undefined') { onDown(); return { close() {} }; }
    const es = new EventSource(`${ROOT}/events`);
    es.addEventListener('state', e => {
      try { onState(JSON.parse((e as MessageEvent<string>).data) as CaptureState); } catch { /* skip a malformed event */ }
    });
    es.onerror = () => onDown();
    return { close: () => es.close() };
  },
};

export async function fetchChampions(): Promise<Champion[]> {
  const res = await fetch('/api/champions');
  if (!res.ok) throw new ApiError(`Request failed (${res.status})`, res.status);
  return (await res.json() as { champions: Champion[] }).champions;
}
