import { bySequence, SLOTS, type Slot } from './logic';
import type { CaptureAction, CaptureHover, Champion, Side } from './types';

interface Props {
  actions: CaptureAction[];
  hover: CaptureHover | null;
  champions: Map<string, Champion>;
  blueLabel: string;
  redLabel: string;
}

function Art({ champion, kind }: { champion: Champion | undefined; kind: 'square' | 'loading' }) {
  if (!champion?.art) return <span className="art-fallback" aria-hidden="true">{(champion?.name ?? '?').slice(0, 2)}</span>;
  return <img src={`/api/champions/${champion.ddragon_id}/${kind}`} alt="" loading="lazy" />;
}

function Cell({ slot, action, hover, champions }: { slot: Slot; action?: CaptureAction; hover: CaptureHover | null; champions: Map<string, Champion> }) {
  const hovering = !action && hover?.sequence === slot.sequence ? hover : null;
  const id = action?.champion_id ?? hovering?.champion_id ?? undefined;
  const champion = id ? champions.get(id) : undefined;
  const kind = slot.action_type === 'ban' ? 'square' : 'loading';
  const state = action ? 'locked' : hovering ? 'hover' : 'empty';
  const name = champion?.name ?? id ?? '';
  const detail = action
    ? `${name}${action.confidence != null ? `, ${Math.round(action.confidence * 100)}%` : ''}${action.via ? `, via ${action.via}` : ''}`
    : hovering ? `hovering ${name || '…'}` : 'empty';
  return (
    <li className={`cell is-${slot.action_type} is-${state}`} data-testid={`slot-${slot.sequence}`} data-state={state} title={`#${slot.sequence} ${slot.side} ${slot.action_type}: ${detail}`}>
      <div className="art">{id ? <Art champion={champion} kind={kind} /> : <span className="seq">{slot.sequence}</span>}</div>
      {slot.action_type === 'pick' && <span className="name">{state === 'hover' ? `${name || '…'} (hovering)` : name}</span>}
    </li>
  );
}

function Side({ side, label, actions, hover, champions }: { side: Side; label: string } & Omit<Props, 'blueLabel' | 'redLabel'>) {
  const mine = SLOTS.filter(s => s.side === side);
  const bans = mine.filter(s => s.action_type === 'ban');
  const picks = mine.filter(s => s.action_type === 'pick');
  const found = bySequence(actions);
  return (
    <section className={`side is-${side}`} aria-label={`${side} side`}>
      <h2>{label}</h2>
      <ul className="bans" aria-label="Bans">{bans.map(s => <Cell key={s.sequence} slot={s} action={found.get(s.sequence)} hover={hover} champions={champions} />)}</ul>
      <ul className="picks" aria-label="Picks">{picks.map(s => <Cell key={s.sequence} slot={s} action={found.get(s.sequence)} hover={hover} champions={champions} />)}</ul>
    </section>
  );
}

export function Board({ actions, hover, champions, blueLabel, redLabel }: Props) {
  return (
    <div className="board">
      <Side side="blue" label={blueLabel} actions={actions} hover={hover} champions={champions} />
      <Side side="red" label={redLabel} actions={actions} hover={hover} champions={champions} />
    </div>
  );
}
