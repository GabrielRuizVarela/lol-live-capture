# Notice

lol-live-capture isn't endorsed by Riot Games and doesn't reflect the views or opinions of Riot Games or anyone
officially involved in producing or managing Riot Games properties. Riot Games and all associated properties are
trademarks or registered trademarks of Riot Games, Inc.

What this repository does and does not contain:

- **No Riot art is stored here.** `lol-live-capture setup` downloads champion tiles and loading-screen portraits from
  Riot's public Data Dragon CDN onto your machine, under Riot's terms for that content.
- `src/lol_live_capture/data/champion.json.gz` is a copy of the Data Dragon champion table (names and numeric keys),
  bundled so the tool and its tests work offline. `setup` replaces it with the latest patch's table.
- `src/lol_live_capture/assets/draft_center.png` is a small crop of the Worlds 2026 champion-select overlay, used as a
  template to tell when the draft screen is visible. It is Riot's broadcast graphic; remove it from your fork if you
  are not comfortable shipping it, and `layout.is_draft_screen` will need another template.
- The screenshots in `docs/img/` show champion tiles and portraits from Data Dragon: the UI's demo mode (an
  invented draft) and the public lol-draft site (`site-*.jpg`, real Worlds 2026 games). They are Riot's art; replace them if you are not comfortable shipping them.
- Broadcast stills used by the recognizer tests are **not** in the repository (`tests/fixtures/private/` is git-ignored).
  Those tests skip without them.
- `lolesports.py` calls the unofficial lolesports.com API with the public key the lolesports.com web client uses.
  These endpoints are undocumented and can change or be restricted at any time; keep the request rate low.
- Video is read from the stream with `yt-dlp` and `ffmpeg`. Make sure that reading a given stream is allowed where you
  use it. Don't redistribute recordings.
