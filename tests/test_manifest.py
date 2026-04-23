import json
from pathlib import Path

import pytest

from ramp_motion.manifest import CycleManifest, write_manifest


def _sample_manifest(tmp_path: Path) -> CycleManifest:
    return CycleManifest(
        stream_id="cam-01",
        cycle_id="cam-01-2026-04-23T10-22-41Z",
        ref_path="streams/cam-01/ref/2026-04-23T10-15-03Z.jpg",
        ref_at="2026-04-23T10:15:03Z",
        query_path="streams/cam-01/query/2026-04-23T10-22-41Z.jpg",
        query_at="2026-04-23T10:22:41Z",
        cycle_duration_sec=458.3,
        motion_peak_fraction=0.34,
        motion_peak_blob_area_px=18420,
        motion_samples_count=1234,
        calibration_frames=[
            "streams/cam-01/calibration/2026-04-23T10-19-12Z_peak=0.34.jpg",
        ],
        new_ref_path="streams/cam-01/ref/2026-04-23T10-22-41Z.jpg",
    )


def test_write_manifest_creates_valid_json(tmp_path: Path):
    m = _sample_manifest(tmp_path)
    dest = tmp_path / "cycles" / "2026-04-23T10-22-41Z.json"
    write_manifest(m, dest)

    assert dest.exists()
    data = json.loads(dest.read_text())
    assert data["stream_id"] == "cam-01"
    assert data["motion_peak_fraction"] == 0.34


def test_write_manifest_is_atomic_no_leftover_tmp(tmp_path: Path):
    m = _sample_manifest(tmp_path)
    dest = tmp_path / "cycles" / "out.json"
    write_manifest(m, dest)
    tmp_files = list(dest.parent.glob("*.tmp"))
    assert tmp_files == []
