import { act, render, screen, waitFor } from '@testing-library/react';
import userEventLite from './userEventLite';
import { App } from './App';
import type { CaptureApi } from './api';
import { createDemoApi, DEMO_DRAFT, demoFrame } from './demo';
import type { CaptureState } from './types';

describe('demo replay', () => {
  it('locks one action per step in sequence order, hovering the next slot', () => {
    const f = demoFrame(DEMO_DRAFT, 3 * 1000 + 600, 1000);
    expect(f.actions.map(a => a.sequence)).toEqual([1, 2, 3]);
    expect(f.hover).toMatchObject({ sequence: 4, champion_id: DEMO_DRAFT[3] });
    expect(demoFrame(DEMO_DRAFT, 20 * 1000, 1000).complete).toBe(true);
    expect(new Set(DEMO_DRAFT).size).toBe(20);
  });
});

function fakeApi(initial: CaptureState) {
  let push: (s: CaptureState) => void = () => {};
  const calls: string[] = [];
  const api: CaptureApi = {
    start: async source => { calls.push(`start ${source}`); },
    stop: async () => { calls.push('stop'); },
    state: async () => initial,
    subscribe: onState => { push = onState; return { close() {} }; },
  };
  return { api, calls, push: (s: CaptureState) => act(() => push(s)) };
}

const base: CaptureState = { status: 'idle', source: null, layout: 'x', updated_at: 't0', stream_t: null, frame_age_ms: null, game_index: 1, teams: null, actions: [], hover: null, warnings: [], error: null };

describe('App', () => {
  beforeEach(() => { localStorage.clear(); vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify({ champions: [{ id: '22', name: 'Ashe', ddragon_id: 'Ashe', art: false }] })))); });
  afterEach(() => vi.unstubAllGlobals());

  it('shows captured actions on the board, a hover ghost, warnings and the pill', async () => {
    const { api, push } = fakeApi(base);
    render(<App api={api} />);
    expect(screen.getByTestId('pill')).toHaveTextContent('Off');
    await push({ ...base, status: 'drafting', actions: [{ sequence: 7, side: 'blue', action_type: 'pick', champion_id: '22', confidence: 0.9, locked_at: 1 }], hover: { sequence: 8, side: 'red', action_type: 'pick', champion_id: null, confidence: 0.5 }, warnings: ['Ashe locked twice'] });
    expect(screen.getByTestId('pill')).toHaveTextContent('Live');
    await waitFor(() => expect(screen.getByTestId('slot-7')).toHaveTextContent('Ashe'));
    expect(screen.getByTestId('slot-7')).toHaveAttribute('data-state', 'locked');
    expect(screen.getByTestId('slot-8')).toHaveAttribute('data-state', 'hover');
    expect(screen.getByTestId('slot-9')).toHaveAttribute('data-state', 'empty');
    expect(screen.getByText('Ashe locked twice')).toBeInTheDocument();
  });

  it('refuses a bad address without calling the server, and starts with a good one', async () => {
    const { api, calls } = fakeApi(base);
    render(<App api={api} />);
    const input = screen.getByLabelText('Stream address');
    await userEventLite.type(input, 'not a url');
    await userEventLite.click(screen.getByRole('button', { name: 'Start reading' }));
    expect(screen.getByRole('alert')).toHaveTextContent('not a web address');
    expect(calls).toEqual([]);
    await userEventLite.type(input, 'https://youtu.be/abc');
    await userEventLite.click(screen.getByRole('button', { name: 'Start reading' }));
    await waitFor(() => expect(calls).toEqual(['start https://youtu.be/abc']));
  });
});

it('the demo api reports connecting, then the draft screen, then drafting', async () => {
  vi.useFakeTimers();
  const api = createDemoApi(1000);
  const seen: string[] = [];
  api.subscribe(s => { if (seen.at(-1) !== s.status) seen.push(s.status); }, () => {});
  await api.start('demo');
  await vi.advanceTimersByTimeAsync(3000);
  expect(seen).toEqual(['connecting', 'waiting_for_draft', 'drafting']);
  await api.stop();
  vi.useRealTimers();
});
