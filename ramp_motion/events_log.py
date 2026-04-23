from __future__ import annotations
import json
import threading
from pathlib import Path
from typing import Mapping


class EventsLog:
    """Thread-safe JSONL appender for pipeline events (disconnects, cycles, resets)."""

    def __init__(self, path: Path):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, event: Mapping) -> None:
        line = json.dumps(dict(event), separators=(",", ":"))
        with self._lock, open(self._path, "a") as f:
            f.write(line)
            f.write("\n")
            f.flush()
