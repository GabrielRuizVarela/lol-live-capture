# Contract: how lol-live-capture works

The tool reads a League of Legends **pick/ban draft from a broadcast while it happens** (the official lolesports stream
on YouTube or Twitch, or a saved video) and publishes it as `CaptureState` over an HTTP API and an SSE stream. It
produces the draft **actions** only; what you do with them (predictions, overlays, a draft tool) is up to the consumer.

## Pipeline

```
stream URL ──yt-dlp -o - ──ffmpeg (fps=2, 1280x720, bgr24)──▶ frames
frames ──layout.py (slot boxes for the Worlds 2026 overlay)──▶ crops per slot
crops ──recognize.py (champion matcher vs reference bank)──▶ SlotRead per slot per frame
SlotReads ──tracker.py (state machine)──▶ CaptureState (actions, hover, status)
CaptureState ──run/capture-state.json (atomic write)──▶ API ──SSE──▶ ui/ (or any client)
```

Code lives in the package `src/lol_live_capture/`. Speech-to-text (`faster-whisper`) is the optional extra `audio`; the tests that need Riot's footage or champion art skip without them.

### `stream.py`: frame source
- `open_frames(source: str, *, fps: float = 2.0, width=1280, height=720, realtime=False) -> Iterator[Frame]`
  - `Frame = (t_seconds: float, image: np.ndarray HxWx3 uint8 BGR)`.
- `source` is one of:
  - a local video file, read with ffmpeg directly. With `realtime=True` ffmpeg gets `-re`, so the file plays at 1× like a live stream;
  - an `http(s)` page URL (YouTube or Twitch, VOD or live). The command is `yt-dlp --js-runtimes node -f "best[height<=720]/bestvideo[height<=720]" -o - URL | ffmpeg -i pipe:0 -vf fps=F,scale=W:H -f rawvideo -pix_fmt bgr24 pipe:1`.
  - ffmpeg must never fetch googlevideo URLs itself: YouTube answers 403 unless yt-dlp downloads them.
- Subprocesses are started in their own process group and killed on close or exception.
- A `--start`/`--end` seek is supported for local files, for replay tests.

### `layout.py`: where things are on screen
- `LAYOUT_ID = "worlds-2026-1280x720-v1"`.
- Normalised boxes `(x0, y0, x1, y1)` in 0..1 for:
  - the 10 ban slots: blue b1..b5 and red b1..b5, in on-screen order;
  - the 10 pick slots: blue p1..p5 and red p1..p5;
  - the two team tricode areas;
  - the phase/timer area.
- `is_draft_screen(frame) -> (bool, score)` is a cheap check (template or colour signature) that the champion-select overlay is visible.
- The boxes are calibrated by hand from Play-In frames. They are data, not code logic.

### `recognize.py`: which champion is in a slot
- **Reference bank:** built from the Data Dragon squares (bans) and loading-screen portraits (picks) that `lol-live-capture setup` downloads into `$LIVE_CAPTURE_HOME/champions/`, and cached in `$LIVE_CAPTURE_HOME/capture/refs/`. Nothing from Riot is stored in this repository.
- `read_slot(crop, kind: "ban"|"pick") -> SlotRead(state: "empty"|"active"|"hover"|"locked", champion_id: str|None, score: float, margin: float)`:
  - `score` is the best match;
  - `margin` is best minus second best;
  - `state` comes from the overlay's own cues (greyed hover vs full colour, the lock border). `active` is an empty slot being chosen right now (the lighter team-colour ban slot with the slash): the recognizer reports `active` instead of matching a champion. For picks, `active` is only reported when the highlight is detected reliably, otherwise never.
- **Speed:** all 20 slots in under 150 ms per frame on CPU.

### `tracker.py`: from noisy reads to draft actions (pure Python, no cv2)
- **Input:** a stream of `(t, {slot_key: SlotRead})` plus `is_draft_screen`.
- **Output:** `CaptureState`; see the schema below.
- **Rules (lock rule v2, "commit on advance"):**
  - For each slot the tracker keeps its **latest stable read**: the same `champion_id` in ≥ `stable_frames` (`K = 3`) consecutive draft frames, with `state` `hover` or `locked` and `margin ≥ M`. A non-draft frame neither counts nor breaks a streak; a different champion or an `empty`/`active` read breaks it.
  - A slot with sequence `s` is **committed** (a final action) with its latest stable champion once a slot with a **higher** sequence has been non-empty (`active`, or `hover`/`locked` with a champion) for ≥ `K` consecutive draft frames. Joining the stream mid-draft therefore commits every slot that is already filled when the next one advances.
  - The **last pending** action (no higher slot can move) is committed once its read is `locked` and stable for ≥ `final_stable_frames` (`L = 6`) draft frames. After 19 actions are committed, a change in the read of an already-committed pick slot marks the end-of-draft **role reorder** of the pick portraits; the last slot is then committed with its latest stable read.
  - **Draft over (end-of-draft reconciliation).** The draft is over when the turn reached 20 and no slot has been `active` for ≥ `no_active_frames` (4) draft frames, or when the at-rest pick layout shows a side's portraits in a different order than the committed picks (the role reorder — several committed pick slots at once, or one reading another committed champion of the same side), whichever comes first. From then on positional pick commits stop for good (after the reorder a pick slot no longer holds the champion of its own sequence). Slot 20 is committed with its pre-reorder stable read, a missing **ban** is committed with its stable locked read (`final_ban`), and each side's **final pick set** is read from ≥ `final_stable_frames` consecutive at-rest frames where all five of that side's portraits read the same five distinct champions (`locked`, or an at-rest `hover`). Because it is a set, the role reorder does not change it:
    - exactly one pick missing: it is the final set's one uncommitted champion, committed only when the slot's turn evidence does not contradict it (`final_set`);
    - two or more missing: the leftover champions are assigned from the champion read most often in each missing slot during its own turn (including hover reads) and committed only when that assignment is unique and consistent (`final_set`);
    - a committed pick not in the final set is left as is (committed is final), a warning is added, and that side's final-set commits are skipped.
  - Every action records its origin in `via`: `turn` (positional commit, including slot 20), `final_set` or `final_ban`.
  - A committed action never changes, except by an explicit `reset()` or a new draft.
  - `sequence` comes from the standard 20-action order (`B1 R1 B2 R2 B3 R3 | B R R B B R | R4 B4 R5 B5 | R B B R`), mapped from the slot: blue ban k, red pick k, and so on.
  - Out-of-order commits are allowed; the frontend applies them in `sequence` order and only up to the first gap.
  - **Hover:** the slot that is currently hovering, with its champion if one can be read.
  - **New draft:** the draft screen reappears with all slots empty after a draft completed, or after ≥ 60 s without the draft screen. The tracker then starts `game_index + 1`.
  - Inputs that contradict the rules are reported in `warnings`: a champion locked twice, or a champion that is illegal under Fearless.

## Run model
- The capture worker is **its own process**: `python -m lol_live_capture.run --source URL [--state-file F] [--blue-team NAME --red-team NAME] [--realtime]`.
- The team names are stored in `CaptureState.teams[side].name` next to the broadcast `tricode`. While the worker runs the API may rewrite `run/capture-teams.json`; the worker re-reads that side-car file every 5 s (Next game with swapped sides).
- `--realtime` makes a local file play at 1× (ffmpeg `-re`); the API sets it for local files (saved broadcasts replay like the live stream), never for stream URLs.
- It writes `CaptureState` atomically (temp file + rename) to `$LIVE_CAPTURE_HOME/run/capture-state.json` after every frame that changes something, and at least every 2 s as a heartbeat.
- The API starts and stops it. It runs in its own process group and its pid is kept in `run/capture.pid`. At most one runs.

## API (prefix `/api/capture`; no authentication: bind to localhost)
All paths are under `/api`.

| Method | Path | Body / result |
|---|---|---|
| POST | `/capture/start` | `{source: "https://…", teams?: {blue: {id, name}, red: {id, name}}}` → `{status}`. Allowed sources: `youtube.com`, `youtu.be`, `twitch.tv`, or a local file under `$LIVE_CAPTURE_HOME/capture/`. Team ids/names are printable strings of at most 80 characters; the names reach the worker as `--blue-team`/`--red-team`. Anything else → 422. |
| POST | `/capture/teams` | `{blue: {id, name}, red: {id, name}}` → `{status:"ok"}`. Rewrites `run/capture-teams.json`; the running worker picks it up within 5 s. Invalid teams → 422. |
| POST | `/capture/stop` | → `{status:"idle"}` |
| GET | `/capture/state` | `CaptureState` |
| GET | `/capture/events` | SSE: `event: state` with a `CaptureState` JSON every time `updated_at` changes. Keep-alive comment every 15 s. |

### `CaptureState`
```json
{ "status": "idle|connecting|waiting_for_draft|drafting|draft_complete|error",
  "source": "https://…", "layout": "worlds-2026-1280x720-v1",
  "updated_at": "ISO-8601", "stream_t": 1234.5, "frame_age_ms": 400,
  "game_index": 1,
  "teams": {"blue": {"tricode": "KC", "name": "Karmine Corp"},
            "red": {"tricode": "MVK", "name": "MVK Esports"}},
  "actions": [{"sequence": 1, "side": "blue", "action_type": "ban", "champion_id": "888",
               "confidence": 0.97, "locked_at": 1200.0, "via": "turn"}],
  "hover": {"sequence": 7, "side": "blue", "action_type": "pick", "champion_id": "157", "confidence": 0.6},
  "warnings": ["…"], "error": null }
```
`champion_id` is Riot's numeric champion key as a string (`"62"` = Wukong). `GET /api/champions` maps keys to names and Data Dragon ids, and `GET /api/champions/{ddragon_id}/{square|loading}` serves the downloaded art.

## UI (`ui/`)
A small React app, served by `lol-live-capture serve` at `/`:
- a stream address field (remembered), Start/Stop, and a status pill (Connecting / Waiting for the draft screen / Live / Draft read / Error, with the frame age when it is over 5 s);
- the draft board: 5 bans and 5 picks per side with the champion art, a dashed "hovering" ghost on the slot being chosen, confidence and `via` in each slot's tooltip;
- the casters' last line when audio is on, the lolesports verdict, the tracker's warnings, and a JSON download of the actions;
- `/?demo` replays an invented draft with no stream, server-side art or setup.

## Measured accuracy
The image-only baseline on 7 recorded Worlds 2026 Play-In drafts was 132/140 actions exact, 1 wrong, 7 missing (94.3% exact).
Audio fusion (below) exists to close that gap. A wrong lock is worse than a missed one, so consumers should treat
`confidence` and `via` as the trust signal and keep a human correction path; `confidence >= 0.85`, or `via` of
`audio+image` or `final_set`, is the trusted band.

## Audio first, image confirms, lolesports checks

**Why:** video alone is fragile (overlay animations, pre-hovers, look-alike splash art). The casters name almost every ban and pick, so the audio gives a strong prior. The image then only has to confirm a few named candidates instead of choosing among about 170 champions. After the draft, the lolesports game feed confirms the result.

### Audio: `audio.py`, `speech.py`, `spotter.py`
- **`audio.py`:** reads the same stream as the frames (one yt-dlp → one ffmpeg) and returns 16 kHz mono s16le chunks with stream timestamps aligned with the frames.
- **`speech.py`:** local faster-whisper, English. On with a CUDA device (`large-v3-turbo`); `LIVE_WHISPER_DEVICE=cpu` opts in to the CPU (`small.en`, may lag a live stream); `LIVE_WHISPER_MODEL` overrides the model.
  - It uses rolling windows of about 4 s with 1 s of overlap, and a hotword or prompt bias toward the champion names likely on this patch.
  - Every segment comes out as `(t0, t1, text, avg_logprob)`.
- **`spotter.py`:** champion mentions from the text.
  - Names and caster aliases are matched fuzzily (Wukong/Monkey King, TF, J4, Xin, Mundo, Kog, K'Sante, ASol, LB, MF, GP and so on).
  - Each mention has a **cue**, taken from the words around it:
    - `ban`: "ban", "banned", "takes away", "won't see";
    - `pick`: "lock in", "first pick", "they go", "picks up", "comes out";
    - `speculative`: "could", "maybe", "if", "would", "what about";
    - `neutral`: none of the above.
  - Output: `Mention(t, champion_id, cue, text, score)`.

### Fusion in the tracker
- **Candidates:** for the slot whose turn it is, the candidate set is the `ban`/`pick`/`neutral` mentions of the acting side's phase, from 20 s before the turn started until commit. Speculative mentions are excluded, as is any champion already committed in this game.
- **Audio evidence:** `DraftTracker.observe_mentions(mentions)` stores mentions with cue in {ban, pick, neutral}. Ignores "speculative" mentions and any champion already committed. `audio_support(champion_id, t0, t1, action_type) -> float` returns the best matching mention in [t0, t1]; a ban cue counts for bans, a pick cue for picks, and a neutral cue counts half for both.
- **Fusion at commit time** (slot s with image stable champion X, window W = [turn_start - 20s, now]):
  - **(a)** X has audio_support > 0 in W → commit X, `via="audio+image"`, confidence ≥ 0.9.
  - **(b)** X has no support, and some Y in the slot's top-3 candidates has strong support (pick/ban cue) with image score ≥ 0.5 × X's score → commit Y, `via="audio+image"`, confidence 0.8. This fixes look-alikes.
  - **(c)** Otherwise commit X with `via="image"` and confidence 0.7 if its open-set margin is clear. If the margin is weak, leave the slot uncommitted, exactly as today.
- **Missing slots:** after the turn passed s, if the image never produced a stable champion but exactly one unclaimed champion Y has a strong cue for s's side and type in W, and Y is among the slot's candidates in at least 2 frames, commit Y with `via="audio+image"`, confidence 0.6. If Y is never among the candidates, do NOT commit: record {"sequence": s, "suggestion": Y} in state["suggestions"] for the UI.
- **Confidence upgrades:** a committed action's confidence may later be RAISED by audio support arriving within 15 s (same champion only); `via` then becomes "audio+image". A committed champion never changes.
- **Via values:** `turn` (positional commit), `audio+image`, `image`, `final_set`, `final_ban`.

### lolesports confirmation: `lolesports.py`
- **Source:** the public, unofficial lolesports endpoints that the lolesports.com site uses: `esports-api.lolesports.com/persisted/gw/getLive`, `getEventDetails`, and `feed.lolesports.com/livestats/v1/window/{gameId}`. The API key is the public one embedded in the lolesports.com JavaScript; it is not a secret.
- **Matching:** after the draft, poll for the live game whose teams match the session. Once the game feed lists the 10 champions, compare them with the captured picks.
  - If the feed has bans, compare those too.
  - Mark each captured action `confirmed` or `mismatch`, in `CaptureState.lolesports = {game_id, status, picks, bans|null, checked_at}`.
- **Mismatches:** never silently rewrite a committed action. The UI shows the mismatch with a one-tap fix.

### Interpreting the transcript: `interpret.py` (optional)
- An LLM turns caster speech into draft events through any OpenAI-compatible chat-completions endpoint. It is on only when `LIVE_LLM_API_KEY` is set; `LIVE_LLM_BASE_URL` (the full `.../chat/completions` URL, default OpenAI), `LIVE_LLM_MODEL` (default `gpt-4o-mini`) and `LIVE_LLM_EXTRA_BODY` (a JSON object merged into the request, for provider options) configure it. The request uses `response_format: json_object`; about 300 tokens a call.
- **Input:** the last 30 s of transcript plus the draft context: teams, side, acting side and current turn, champions already used, Fearless locks.
- **Output:** `{"events": [{"champion", "team": "blue|red|null", "action": "ban|pick|hover|speculative", "confidence": 0..1}]}`. Champion names are mapped to ids with the spotter's alias table; unknown names are dropped.
- **Calls:** at most one in flight. A new call starts when a new transcript segment ends and the previous call has returned.
- **Fallback:** if a call fails or takes over 6 s, use the rule-based `spotter.py`.
- **Key:** read from the environment only, never stored by the app and never logged. The transcript is sent to the provider you configure.

### lolesports confirmation: verified facts (2026-10-19; re-check, the endpoints are unofficial)
- **Key:** the public `x-api-key` the lolesports.com web client uses (community-documented; it has rotated before). The code ships one as a constant, overridable with env `LOLESPORTS_API_KEY`.
- **Feed (`feed.lolesports.com/livestats/v1/window/{gameId}?startingTime=…` and `/details/…`) needs no key.** `startingTime` must be round-to-10 s ISO in the past; an unnecessary-too-early time answers HTTP 204 (no frames after it yet).
- **Champions:** every window response carries `gameMetadata.{blue,red}TeamMetadata.participantMetadata[*].championId` + `role` (top/jungle/mid/bottom/support). The id is a Data Dragon string (`Jayce`, `MonkeyKing`, `KSante`); map to Riot numeric keys with the Data Dragon champion table (`champions.py`; `data[id]["key"]`: MonkeyKing→62, KSante→897; "Wukong" is not a ddragon id).
- **No bans** in `/window` or `/details` (checked full frame dumps); picks only. The livestats window is per-game role order, not pick order, so confirmation compares champion **sets** per side.
- **Availability:** for KC–MVK g1 (scheduled 2026-10-16 18:00Z) the first window data is at 2026-10-16T18:11:40Z; earlier startingTimes give 204. Finished-game windows stay fetchable days later. For a live game, ask with `startingTime = now − 90 s` (rounded) and retry with older offsets.
- **Matching to a session:** `getEventDetails.data.event.match.games[i].teams = [{id, side}]` and `match.teams[].code` give the broadcast tricodes (KC, MVK, FUR, C9…). Sides are swapped per game in a series (" Knights"), so match on the unordered code pair; series game number is `game.number` (1-based = CaptureState.game_index). `getLive` answers with the same match shape (and `gameState: "inProgress"`), and is empty between games — fall back to today's `getSchedule` event + `getEventDetails`.
