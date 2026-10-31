import type { ActionType, CaptureLolesports, CaptureState, CaptureStatus, Side } from './types';

export interface Slot { sequence: number; side: Side; action_type: ActionType; /** 1..5 within its side and type */ index: number }

/** The standard 20-action draft order, as the tracker numbers it. */
const ORDER: Array<[Side, ActionType]> = [
  ['blue', 'ban'], ['red', 'ban'], ['blue', 'ban'], ['red', 'ban'], ['blue', 'ban'], ['red', 'ban'],
  ['blue', 'pick'], ['red', 'pick'], ['red', 'pick'], ['blue', 'pick'], ['blue', 'pick'], ['red', 'pick'],
  ['red', 'ban'], ['blue', 'ban'], ['red', 'ban'], ['blue', 'ban'],
  ['red', 'pick'], ['blue', 'pick'], ['blue', 'pick'], ['red', 'pick'],
];

export const SLOTS: Slot[] = (() => {
  const seen: Record<string, number> = {};
  return ORDER.map(([side, action_type], i) => {
    const key = `${side}-${action_type}`;
    seen[key] = (seen[key] ?? 0) + 1;
    return { sequence: i + 1, side, action_type, index: seen[key] };
  });
})();

export const slotAt = (sequence: number): Slot | undefined => SLOTS[sequence - 1];

export interface StatusPill { tone: 'off' | 'busy' | 'live' | 'done' | 'error'; text: string }

export function statusPill(state: Pick<CaptureState, 'status' | 'frame_age_ms' | 'error'> | null): StatusPill {
  if (!state) return { tone: 'off', text: 'Off' };
  const age = state.frame_age_ms !== null && state.frame_age_ms > 5000 ? `, last frame ${Math.round(state.frame_age_ms / 1000)} s ago` : '';
  const map: Record<CaptureStatus, StatusPill> = {
    idle: { tone: 'off', text: 'Off' },
    connecting: { tone: 'busy', text: 'Connecting' },
    waiting_for_draft: { tone: 'busy', text: `Waiting for the draft screen${age}` },
    drafting: { tone: 'live', text: `Live, reading the draft${age}` },
    draft_complete: { tone: 'done', text: `Draft read${age}` },
    error: { tone: 'error', text: state.error ? `Error: ${state.error}` : 'Error' },
  };
  return map[state.status];
}

/** The lolesports verdict after the draft: confirmed, a list of differences, or nothing yet. */
export function lolesportsBadge(lolesports: CaptureLolesports | null | undefined, nameOf: (id: string) => string): { tone: 'ok' | 'warn'; text: string } | null {
  if (!lolesports) return null;
  if (lolesports.status === 'confirmed') return { tone: 'ok', text: 'Confirmed by lolesports' };
  if (lolesports.status !== 'mismatch') return null;
  const items = (lolesports.mismatches ?? []).map(m => `${nameOf(m.captured)} vs ${(m.feed ?? []).map(nameOf).join(' / ')}`);
  return { tone: 'warn', text: items.length ? `lolesports differs: ${items.join(', ')}` : 'lolesports differs' };
}

/** A problem with the address, or null. The server has the final say (and accepts local files under its capture directory). */
export function sourceProblem(input: string): string | null {
  const v = input.trim();
  if (!v) return 'Paste the stream address first.';
  if (v.startsWith('/') || v.startsWith('~/')) return null;
  let u: URL;
  try { u = new URL(v); } catch { return 'That is not a web address.'; }
  if (!/^https?:$/.test(u.protocol)) return 'Use an https:// address.';
  const host = u.hostname.replace(/^www\.|^m\./, '');
  if (!/(^|\.)youtube\.com$|^youtu\.be$|(^|\.)twitch\.tv$/.test(host)) return 'Use a YouTube or Twitch address.';
  return null;
}

/** Actions keyed by sequence (the last one wins, as the server never changes a committed action). */
export function bySequence<T extends { sequence: number }>(items: T[]): Map<number, T> {
  return new Map(items.map(item => [item.sequence, item]));
}

export function isLive(state: CaptureState | null): boolean {
  return !!state && state.status !== 'idle' && state.status !== 'error';
}
