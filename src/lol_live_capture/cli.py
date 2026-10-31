"""``lol-live-capture``: ``setup`` (download champion art), ``serve`` (API + UI), ``run`` (the worker, headless)."""
from __future__ import annotations

import argparse
import sys

from . import assets, paths


def _setup(args: argparse.Namespace) -> int:
    result = assets.setup(version=args.version, force=args.force)
    print(f"patch {result['version']}: {result['fetched']} downloaded, {result['kept']} kept, "
          f"{len(result['failed'])} failed -> {result['dir']}")
    for line in result["failed"]:
        print(f"  FAILED {line}", file=sys.stderr)
    return 1 if result["failed"] else 0


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    if not assets.is_ready():
        print("note: champion art is missing; run `lol-live-capture setup` first (the reader needs it)",
              file=sys.stderr)
    uvicorn.run("lol_live_capture.server:app", host=args.host, port=args.port, log_level="info")
    return 0


def _run(args: argparse.Namespace) -> int:
    from . import run

    return run.main(args.worker_args)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lol-live-capture", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("setup", help="download champion art from Riot's Data Dragon CDN")
    p.add_argument("--version", default=None, help="patch, e.g. 16.9.1 (default: latest)")
    p.add_argument("--force", action="store_true", help="download files that already exist")
    p.set_defaults(func=_setup)

    p = sub.add_parser("serve", help="start the API and the UI")
    p.add_argument("--host", default="127.0.0.1", help="bind address (the API can start processes: keep it local)")
    p.add_argument("--port", type=int, default=8040)
    p.set_defaults(func=_serve)

    p = sub.add_parser("run", help="run the capture worker without the server (args as `python -m lol_live_capture.run`)")
    p.add_argument("worker_args", nargs=argparse.REMAINDER)
    p.set_defaults(func=_run)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
