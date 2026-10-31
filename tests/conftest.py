import pytest

from lol_live_capture import paths

# Where `lol-live-capture setup` put the champion art, captured before the tests move the home.
ART_DIR = paths.champions_dir()


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path_factory, monkeypatch):
    """No test reads or writes the user's real data directory (and all use the bundled champion table)."""
    monkeypatch.setenv("LIVE_CAPTURE_HOME", str(tmp_path_factory.mktemp("home")))
