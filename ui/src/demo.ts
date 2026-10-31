/**
 * Demo mode (`?demo`): a fake capture backend that replays one invented draft the way the tracker reports a real one:
 * connecting, waiting for the draft screen, then a hover before every lock. No stream, server or champion art needed.
 * `?demo&step=MS` sets the time between two locks (default 2500).
 */
import type { CaptureApi, Subscription } from './api';
import { slotAt } from './logic';
import type { CaptureAction, CaptureState, CaptureStatus } from './types';

/** Riot champion keys, by sequence 1..20: Olaf, Pantheon, Nocturne, Karma, Cassiopeia, Orianna, Ashe, Rumble, Ezreal, ... */
export const DEMO_DRAFT = ['2', '80', '56', '43', '69', '61', '22', '68', '81', '147', '3', '555', '13', '777', '24', '62', '246', '64', '234', '268'];

const LINES = [
  (n: string) => `${n} is out, that one was always going to go`,
  (n: string) => `${n} locked, and that tells you a lot about the plan`,
  (n: string) => `there it is, ${n}, they wanted that one early`,
];

export function demoFrame(draft: string[], elapsed: number, stepMs: number): { actions: CaptureAction[]; hover: CaptureState['hover']; complete: boolean } {
  const locked = Math.max(0, Math.min(draft.length, Math.floor(elapsed / stepMs)));
  const actions: CaptureAction[] = [];
  for (let i = 0; i < locked; i++) {
    const s = slotAt(i + 1)!;
    actions.push({ sequence: s.sequence, side: s.side, action_type: s.action_type, champion_id: draft[i], confidence: 0.96, locked_at: 1200 + ((i + 1) * stepMs) / 1000, via: 'turn' });
  }
  const next = slotAt(locked + 1);
  let hover: CaptureState['hover'] = null;
  if (next && locked < draft.length) {
    const into = elapsed - locked * stepMs;
    hover = { sequence: next.sequence, side: next.side, action_type: next.action_type, champion_id: into > stepMs / 3 ? draft[locked] : null, confidence: 0.62 };
  }
  return { actions, hover, complete: locked >= draft.length };
}

export function createDemoApi(stepMs = 2500, nameOf: (id: string) => string = id => id): CaptureApi {
  const listeners = new Set<(s: CaptureState) => void>();
  let status: CaptureStatus = 'idle';
  let source: string | null = null;
  let clock = 0;
  let timer: ReturnType<typeof setInterval> | null = null;

  const snapshot = (): CaptureState => {
    const active = status === 'drafting' || status === 'draft_complete';
    const frame = active ? demoFrame(DEMO_DRAFT, clock, stepMs) : { actions: [], hover: null, complete: false };
    const n = frame.actions.length;
    return {
      status, source, layout: 'worlds-2026-1280x720-v1', updated_at: new Date().toISOString(),
      stream_t: status === 'idle' ? null : 1200 + clock / 1000, frame_age_ms: status === 'idle' ? null : 420, game_index: 1,
      teams: active ? { blue: { tricode: 'BLU' }, red: { tricode: 'RED' } } : null,
      actions: frame.actions, hover: frame.hover, warnings: [], error: null,
      audio: status === 'idle' ? null : { enabled: true, llm: false, lag_s: 2.1,
        last_segment: frame.hover?.champion_id ? `they're hovering ${nameOf(frame.hover.champion_id)} here` : n ? LINES[n % LINES.length](nameOf(DEMO_DRAFT[n - 1])) : null },
    };
  };
  let state = snapshot();
  const emit = () => { state = snapshot(); listeners.forEach(l => l(state)); };
  const advance = (ms: number) => {
    if (status === 'idle') return;
    clock += ms;
    if (status === 'connecting' && clock >= -1200) status = 'waiting_for_draft';
    if (status === 'waiting_for_draft' && clock >= 0) status = 'drafting';
    if (status === 'drafting' && demoFrame(DEMO_DRAFT, clock, stepMs).complete) status = 'draft_complete';
    emit();
  };

  return {
    async start(src) { source = src; status = 'connecting'; clock = -2000; emit(); if (timer) clearInterval(timer); timer = setInterval(() => advance(250), 250); return { status }; },
    async stop() { if (timer) clearInterval(timer); timer = null; status = 'idle'; source = null; clock = 0; emit(); return { status }; },
    async state() { return state; },
    subscribe(onState): Subscription {
      listeners.add(onState);
      queueMicrotask(() => { if (listeners.has(onState)) onState(state); });
      return { close: () => { listeners.delete(onState); } };
    },
  };
}
