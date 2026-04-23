from __future__ import annotations
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List


@dataclass
class CycleManifest:
    stream_id: str
    cycle_id: str
    ref_path: str
    ref_at: str
    query_path: str
    query_at: str
    cycle_duration_sec: float
    motion_peak_fraction: float
    motion_peak_blob_area_px: int
    motion_samples_count: int
    calibration_frames: List[str]
    new_ref_path: str


def write_manifest(manifest: CycleManifest, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(asdict(manifest), f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, dest)
