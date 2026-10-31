import { bySequence, lolesportsBadge, slotAt, SLOTS, sourceProblem, statusPill } from './logic';

describe('draft order', () => {
  it('has 20 slots, 10 bans and 10 picks, five of each per side', () => {
    expect(SLOTS).toHaveLength(20);
    for (const side of ['blue', 'red'] as const) {
      expect(SLOTS.filter(s => s.side === side && s.action_type === 'ban')).toHaveLength(5);
      expect(SLOTS.filter(s => s.side === side && s.action_type === 'pick')).toHaveLength(5);
    }
  });
  it('matches the tracker numbering', () => {
    expect(slotAt(1)).toMatchObject({ side: 'blue', action_type: 'ban', index: 1 });
    expect(slotAt(7)).toMatchObject({ side: 'blue', action_type: 'pick', index: 1 });
    expect(slotAt(9)).toMatchObject({ side: 'red', action_type: 'pick', index: 2 });
    expect(slotAt(13)).toMatchObject({ side: 'red', action_type: 'ban', index: 4 });
    expect(slotAt(20)).toMatchObject({ side: 'red', action_type: 'pick', index: 5 });
  });
});

describe('statusPill', () => {
  it('is off without a state and shows a stale frame age', () => {
    expect(statusPill(null).tone).toBe('off');
    expect(statusPill({ status: 'drafting', frame_age_ms: 7400, error: null }).text).toContain('7 s ago');
    expect(statusPill({ status: 'drafting', frame_age_ms: 400, error: null }).text).not.toContain('ago');
  });
  it('shows the error', () => {
    expect(statusPill({ status: 'error', frame_age_ms: null, error: 'boom' })).toEqual({ tone: 'error', text: 'Error: boom' });
  });
});

describe('sourceProblem', () => {
  it.each(['https://www.youtube.com/watch?v=x', 'https://youtu.be/x', 'https://www.twitch.tv/lolesports', '/home/me/capture/game.mp4'])('accepts %s', v => {
    expect(sourceProblem(v)).toBeNull();
  });
  it.each(['', 'nonsense', 'http://example.com/x', 'ftp://youtube.com/x'])('rejects %s', v => {
    expect(sourceProblem(v)).not.toBeNull();
  });
});

describe('lolesportsBadge', () => {
  const nameOf = (id: string) => ({ '1': 'Annie', '2': 'Olaf' }[id] ?? id);
  it('confirms, lists mismatches, and stays quiet otherwise', () => {
    expect(lolesportsBadge({ status: 'confirmed' }, nameOf)).toEqual({ tone: 'ok', text: 'Confirmed by lolesports' });
    expect(lolesportsBadge({ status: 'mismatch', mismatches: [{ sequence: 3, captured: '1', feed: ['2'] }] }, nameOf)?.text).toBe('lolesports differs: Annie vs Olaf');
    expect(lolesportsBadge({ status: 'pending' }, nameOf)).toBeNull();
    expect(lolesportsBadge(null, nameOf)).toBeNull();
  });
});

it('bySequence keeps the last action for a sequence', () => {
  expect(bySequence([{ sequence: 1, v: 'a' }, { sequence: 1, v: 'b' }]).get(1)?.v).toBe('b');
});
