from pathlib import Path
import textwrap

import pytest
from pydantic import ValidationError
from ramp_motion.config import load_config, StreamConfig, MotionConfig, AppConfig


def test_load_minimal_config(tmp_path: Path):
    yaml_text = textwrap.dedent("""
        streams:
          - stream_id: cam-01
            rtsp_url: rtsp://example/stream1
            roi: { x: 10, y: 20, width: 100, height: 200 }
        motion:
          thresh: 0.02
          t_quiet_sec: 5.0
          t_enter_sec: 0.5
          min_blob_area_px: 500
          calib_rate_limit_sec: 1.0
          t_stale_sec: 10.0
        mog2:
          history: 500
          var_threshold: 16.0
          detect_shadows: true
        clahe:
          clip_limit: 2.0
          tile_grid: [8, 8]
        pipeline:
          target_fps: 5
          rtsp_reconnect_interval_sec: 5
          rtsp_reconnect_attempts: -1
          batch_size_auto: true
        storage:
          root: /tmp/data
          jpeg_quality: 85
        logging:
          level: INFO
          events_log: /tmp/data/events.log
    """)
    path = tmp_path / "config.yaml"
    path.write_text(yaml_text)

    cfg = load_config(path)

    assert isinstance(cfg, AppConfig)
    assert len(cfg.streams) == 1
    assert cfg.streams[0].stream_id == "cam-01"
    assert cfg.streams[0].roi.width == 100
    assert cfg.motion.thresh == pytest.approx(0.02)
    assert cfg.mog2.detect_shadows is True
    assert cfg.clahe.tile_grid == (8, 8)


def test_motion_thresh_must_be_between_0_and_1(tmp_path: Path):
    yaml_text = textwrap.dedent("""
        streams: []
        motion:
          thresh: 1.5
          t_quiet_sec: 5.0
          t_enter_sec: 0.5
          min_blob_area_px: 500
          calib_rate_limit_sec: 1.0
          t_stale_sec: 10.0
        mog2: { history: 500, var_threshold: 16.0, detect_shadows: true }
        clahe: { clip_limit: 2.0, tile_grid: [8, 8] }
        pipeline:
          target_fps: 5
          rtsp_reconnect_interval_sec: 5
          rtsp_reconnect_attempts: -1
          batch_size_auto: true
        storage: { root: /tmp/data, jpeg_quality: 85 }
        logging: { level: INFO, events_log: /tmp/data/events.log }
    """)
    path = tmp_path / "config.yaml"
    path.write_text(yaml_text)

    with pytest.raises(ValidationError):
        load_config(path)
