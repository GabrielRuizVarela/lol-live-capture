/** The capture API's wire types (docs/contract.md, "CaptureState"). `champion_id` is Riot's numeric key as a string. */
export type Side = 'blue' | 'red';
export type ActionType = 'ban' | 'pick';
export type CaptureStatus = 'idle' | 'connecting' | 'waiting_for_draft' | 'drafting' | 'draft_complete' | 'error';
export type CaptureVia = 'audio+image' | 'image' | 'audio' | 'final_set' | 'final_ban' | 'turn';

export interface CaptureAction {
  sequence: number; side: Side; action_type: ActionType; champion_id: string;
  confidence?: number; locked_at: number; via?: CaptureVia;
}
export interface CaptureHover { sequence: number; side: Side; action_type: ActionType; champion_id: string | null; confidence: number }
export interface CaptureLolesports {
  status: string;
  mismatches?: { sequence: number; captured: string; feed: string[] }[];
}
export interface CaptureAudio { enabled: boolean; llm: boolean; last_segment: string | null; lag_s: number | null }
export interface CaptureTeams { blue: { id: string; name: string }; red: { id: string; name: string } }
export interface CaptureState {
  status: CaptureStatus;
  source: string | null;
  layout: string;
  updated_at: string;
  stream_t: number | null;
  frame_age_ms: number | null;
  game_index: number;
  teams: { blue: { tricode: string | null; name?: string | null }; red: { tricode: string | null; name?: string | null } } | null;
  actions: CaptureAction[];
  hover: CaptureHover | null;
  warnings: string[];
  error: string | null;
  lolesports?: CaptureLolesports | null;
  audio?: CaptureAudio | null;
}
export interface Champion { id: string; name: string; ddragon_id: string; art: boolean }
