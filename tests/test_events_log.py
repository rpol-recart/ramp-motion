import json
from pathlib import Path

from ramp_motion.events_log import EventsLog


def test_appends_event_as_jsonl(tmp_path: Path):
    path = tmp_path / "events.log"
    log = EventsLog(path)
    log.write({"event": "cycle_complete", "stream_id": "cam-01"})
    log.write({"event": "disconnect", "stream_id": "cam-02"})

    lines = path.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["event"] == "cycle_complete"
    assert json.loads(lines[1])["stream_id"] == "cam-02"


def test_creates_parent_directory(tmp_path: Path):
    path = tmp_path / "nested" / "events.log"
    log = EventsLog(path)
    log.write({"event": "start"})
    assert path.exists()
