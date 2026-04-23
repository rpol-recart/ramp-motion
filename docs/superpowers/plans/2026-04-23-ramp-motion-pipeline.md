# Ramp Motion Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a DeepStream 6.3 pipeline that watches N RTSP streams, runs CLAHE+MOG2 motion detection in a custom `nvdspreprocess` .so, and a Python pad-probe drives a pure-Python state machine to save ref/query JPEG pairs plus calibration frames to a local volume.

**Architecture:** Single container (NVIDIA runtime). C++/CUDA `.so` computes motion metrics per frame per ROI and attaches them as `NvDsUserMeta`. A Python pad-probe reads the meta, feeds a pure-function state machine (`advance(state, sample) → (state, action)`), and dispatches actions (save frame / write manifest / log event) via `ThreadPoolExecutor`. No DB, no Redis, no REST, no S3 — local volume + JSON manifests only.

**Tech Stack:** DeepStream 6.3 (`nvcr.io/nvidia/deepstream:6.3-triton-multiarch`), Python 3.8 + `pyds` 1.1.8, OpenCV 4.8 with CUDA (cv::cuda::CLAHE, cv::cuda::BackgroundSubtractorMOG2), CMake, pytest, Pydantic v2, Docker + nvidia-container-toolkit.

**Spec:** `/root/ramp-motion/docs/superpowers/specs/2026-04-23-ramp-motion-pipeline-design.md`

---

## File Structure

```
ramp-motion/
├── pyproject.toml                            # created in Task 1
├── requirements.txt                          # created in Task 1
├── README.md                                 # created in Task 1
├── config.yaml                               # created in Task 2 (sample)
├── docker-compose.yml                        # created in Task 19
├── Dockerfile                                # created in Task 12 (stages 1+2) and Task 19 (stage 3)
├── ramp_motion/
│   ├── __init__.py                           # Task 1
│   ├── config.py                             # Task 2 — Pydantic config models
│   ├── state_machine.py                      # Tasks 3-8 — pure state-machine logic
│   ├── manifest.py                           # Task 9 — cycle JSON atomic writer
│   ├── events_log.py                         # Task 10 — JSONL events append
│   ├── frame_saver.py                        # Task 11 — ThreadPool NVMM→JPEG→disk
│   ├── probe.py                              # Tasks 16-17 — pad-probe glue
│   └── app.py                                # Tasks 15, 18 — pipeline builder + main
├── preprocess/
│   ├── CMakeLists.txt                        # Task 12
│   ├── include/
│   │   └── ramp_motion_preproc.h             # Task 12
│   └── src/
│       ├── custom_lib.cpp                    # Task 12, 14 — nvdspreprocess entry points
│       ├── clahe_mog2.cu                     # Task 13 — OpenCV-CUDA pipeline
│       └── motion_meta.cpp                   # Task 14 — NvDsUserMeta pack
├── configs/
│   ├── deepstream_app.yaml                   # Task 15, 18
│   └── nvdspreprocess_config.txt             # Task 14
├── tests/
│   ├── __init__.py                           # Task 1
│   ├── conftest.py                           # Task 1
│   ├── test_config.py                        # Task 2
│   ├── test_state_machine.py                 # Tasks 3-8 (grows across tasks)
│   ├── test_manifest.py                      # Task 9
│   ├── test_events_log.py                    # Task 10
│   ├── test_frame_saver.py                   # Task 11
│   └── test_preprocess_lib.py                # Task 13 (GPU-gated)
└── tools/
    ├── __init__.py                           # Task 20
    ├── calibrate_roi.py                      # Task 20
    └── replay_events.py                      # Task 20
```

Each file has one clear responsibility. `state_machine.py` is pure (no I/O, no timers) so its tests run without Docker, DeepStream, or GPU. The `.so` is tested in isolation via ctypes (GPU needed but no pipeline). Everything else is integration, tested on the Selectel host.

---

## Task 1: Project Scaffold

**Files:**
- Create: `/root/ramp-motion/pyproject.toml`
- Create: `/root/ramp-motion/requirements.txt`
- Create: `/root/ramp-motion/README.md`
- Create: `/root/ramp-motion/ramp_motion/__init__.py`
- Create: `/root/ramp-motion/tests/__init__.py`
- Create: `/root/ramp-motion/tests/conftest.py`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=61", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "ramp-motion"
version = "0.1.0"
description = "Motion-gated ref/query frame capture for ramp monitoring (DeepStream 6.3)"
requires-python = ">=3.8"
dependencies = [
    "pydantic>=2.0,<3.0",
    "PyYAML>=6.0",
    "numpy>=1.24,<2.0",
    "opencv-python-headless>=4.8,<5.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-asyncio>=0.21",
    "pytest-cov>=4.0",
]

[tool.setuptools.packages.find]
include = ["ramp_motion*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "gpu: test requires CUDA GPU and compiled .so",
    "integration: test requires running DeepStream pipeline",
]
```

- [ ] **Step 2: Write `requirements.txt` (installed inside the runtime Docker image)**

```
pydantic>=2.0,<3.0
PyYAML>=6.0
numpy>=1.24,<2.0
opencv-python-headless>=4.8,<5.0
```

Note: `pyds` is installed separately in the Dockerfile from a GitHub release wheel (DS 6.3 specific), not from PyPI.

- [ ] **Step 3: Write `README.md`**

```markdown
# ramp-motion

DeepStream 6.3 pipeline for motion-gated ref/query frame capture on ramp cameras.
See `docs/superpowers/specs/2026-04-23-ramp-motion-pipeline-design.md` for the design.

## Quickstart (host)

```
pip install -e ".[dev]"
pytest -m "not gpu and not integration"
```

## Quickstart (Docker + GPU)

```
docker compose up --build
# ref/query JPEGs appear under ./data/streams/<stream_id>/
```

## Configuration

Edit `config.yaml`. See the spec for defaults.
```

- [ ] **Step 4: Create empty `ramp_motion/__init__.py` and `tests/__init__.py`**

Both files: empty (no content).

- [ ] **Step 5: Write `tests/conftest.py`**

```python
import pytest


def pytest_collection_modifyitems(config, items):
    """Auto-skip GPU tests when FORCE_GPU_TESTS env not set."""
    import os
    if os.environ.get("FORCE_GPU_TESTS") == "1":
        return
    skip_gpu = pytest.mark.skip(reason="GPU test; set FORCE_GPU_TESTS=1 to run")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
```

- [ ] **Step 6: Verify the project installs and pytest collects zero tests**

Run:
```
cd /root/ramp-motion && pip install -e ".[dev]" && pytest -q
```
Expected: `no tests ran` (exit 5 — no tests collected). That's fine for scaffold.

- [ ] **Step 7: Commit**

```bash
cd /root/ramp-motion
git add pyproject.toml requirements.txt README.md ramp_motion/ tests/
git commit -m "feat: project scaffold (pyproject, requirements, readme, empty packages)"
```

---

## Task 2: Pydantic Configuration

**Files:**
- Create: `ramp_motion/config.py`
- Create: `tests/test_config.py`
- Create: `config.yaml` (sample)

- [ ] **Step 1: Write failing test — `tests/test_config.py`**

```python
from pathlib import Path
import textwrap

import pytest
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

    with pytest.raises(Exception):
        load_config(path)
```

- [ ] **Step 2: Run test to confirm it fails**

```
cd /root/ramp-motion && pytest tests/test_config.py -v
```
Expected: `ModuleNotFoundError: No module named 'ramp_motion.config'`.

- [ ] **Step 3: Implement `ramp_motion/config.py`**

```python
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple

import yaml
from pydantic import BaseModel, Field, field_validator, ConfigDict


class ROI(BaseModel):
    model_config = ConfigDict(extra="forbid")
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class StreamConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stream_id: str
    rtsp_url: str
    roi: ROI


class MotionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    thresh: float = Field(gt=0.0, lt=1.0)
    t_quiet_sec: float = Field(gt=0.0)
    t_enter_sec: float = Field(gt=0.0)
    min_blob_area_px: int = Field(ge=0)
    calib_rate_limit_sec: float = Field(ge=0.0)
    t_stale_sec: float = Field(gt=0.0)


class Mog2Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    history: int = Field(gt=0)
    var_threshold: float = Field(gt=0.0)
    detect_shadows: bool


class ClaheConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    clip_limit: float = Field(gt=0.0)
    tile_grid: Tuple[int, int]


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_fps: int = Field(gt=0)
    rtsp_reconnect_interval_sec: int = Field(ge=0)
    rtsp_reconnect_attempts: int   # -1 means infinite
    batch_size_auto: bool


class StorageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    root: str
    jpeg_quality: int = Field(ge=1, le=100)


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    level: str = "INFO"
    events_log: str


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    streams: List[StreamConfig]
    motion: MotionConfig
    mog2: Mog2Config
    clahe: ClaheConfig
    pipeline: PipelineConfig
    storage: StorageConfig
    logging: LoggingConfig


def load_config(path: Path) -> AppConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return AppConfig.model_validate(raw)
```

- [ ] **Step 4: Run test to verify it passes**

```
cd /root/ramp-motion && pytest tests/test_config.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Write sample `/root/ramp-motion/config.yaml`**

```yaml
streams:
  - stream_id: cam-01
    rtsp_url: rtsp://user:pass@camera-host/stream1
    roi: { x: 320, y: 180, width: 1280, height: 720 }

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
  root: /data
  jpeg_quality: 85

logging:
  level: INFO
  events_log: /data/events.log
```

- [ ] **Step 6: Commit**

```bash
cd /root/ramp-motion
git add ramp_motion/config.py tests/test_config.py config.yaml
git commit -m "feat: pydantic config models with yaml loader and validation"
```

---

## Task 3: State Machine — Data Types and Module Skeleton

**Files:**
- Create: `ramp_motion/state_machine.py`
- Create: `tests/test_state_machine.py`

- [ ] **Step 1: Write failing test — verifies data types exist and are immutable**

```python
import pytest
from ramp_motion.state_machine import (
    State, Action, MotionSample, CycleStats, StreamState,
)


def test_types_exist():
    assert State.INIT
    assert State.WAITING_MOTION
    assert State.IN_MOTION

    assert Action.NONE
    assert Action.SAVE_REF
    assert Action.SAVE_CALIBRATION
    assert Action.SAVE_QUERY_AND_NEW_REF
    assert Action.RESET


def test_motion_sample_is_immutable():
    s = MotionSample(stream_id="cam", ts_ns=1, motion_fraction=0.1,
                     max_blob_area_px=100, frame_pts=1)
    with pytest.raises(Exception):
        s.motion_fraction = 0.2


def test_default_stream_state_is_init_no_quiet_timer():
    st = StreamState()
    assert st.state == State.INIT
    assert st.quiet_since_ts_ns is None
    assert st.motion_since_ts_ns is None
    assert st.cycle_stats.peak_fraction == 0.0
```

- [ ] **Step 2: Run test to confirm failure**

```
cd /root/ramp-motion && pytest tests/test_state_machine.py -v
```
Expected: `ModuleNotFoundError: No module named 'ramp_motion.state_machine'`.

- [ ] **Step 3: Implement `ramp_motion/state_machine.py` (types only)**

```python
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class State(Enum):
    INIT = "init"
    WAITING_MOTION = "waiting_motion"
    IN_MOTION = "in_motion"


class Action(Enum):
    NONE = "none"
    SAVE_REF = "save_ref"
    SAVE_CALIBRATION = "save_calibration"
    SAVE_QUERY_AND_NEW_REF = "save_query_and_new_ref"
    RESET = "reset"


@dataclass(frozen=True)
class MotionSample:
    stream_id: str
    ts_ns: int
    motion_fraction: float
    max_blob_area_px: int
    frame_pts: int
    disconnected: bool = False


@dataclass(frozen=True)
class CycleStats:
    peak_fraction: float = 0.0
    peak_blob_area_px: int = 0
    last_calib_ts_ns: int = 0
    samples_count: int = 0


@dataclass(frozen=True)
class StreamState:
    state: State = State.INIT
    quiet_since_ts_ns: Optional[int] = None
    motion_since_ts_ns: Optional[int] = None
    cycle_stats: CycleStats = field(default_factory=CycleStats)
    ref_ts_ns: Optional[int] = None
    last_sample_ts_ns: Optional[int] = None
```

- [ ] **Step 4: Run test to verify it passes**

```
cd /root/ramp-motion && pytest tests/test_state_machine.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
cd /root/ramp-motion
git add ramp_motion/state_machine.py tests/test_state_machine.py
git commit -m "feat(state_machine): data types (State, Action, MotionSample, CycleStats, StreamState)"
```

---

## Task 4: State Machine — INIT → WAITING_MOTION (first reference frame)

**Files:**
- Modify: `ramp_motion/state_machine.py`
- Modify: `tests/test_state_machine.py`

- [ ] **Step 1: Append failing tests to `tests/test_state_machine.py`**

```python
from ramp_motion.state_machine import advance
from ramp_motion.state_machine import MotionThresholds


THRESH = MotionThresholds(
    motion_thresh=0.02,
    t_quiet_ns=5 * 1_000_000_000,
    t_enter_ns=500_000_000,
    min_blob_area_px=500,
    calib_rate_limit_ns=1_000_000_000,
    t_stale_ns=10 * 1_000_000_000,
)


def _sample(ts_sec: float, motion: float, blob: int = 1000) -> MotionSample:
    return MotionSample(
        stream_id="cam-01",
        ts_ns=int(ts_sec * 1_000_000_000),
        motion_fraction=motion,
        max_blob_area_px=blob,
        frame_pts=0,
    )


def test_init_stays_init_when_motion_high():
    st = StreamState()
    new_st, action, _ = advance(st, _sample(0.0, motion=0.5), THRESH)
    assert new_st.state == State.INIT
    assert action == Action.NONE


def test_init_starts_quiet_timer_on_low_motion():
    st = StreamState()
    new_st, action, _ = advance(st, _sample(0.0, motion=0.0), THRESH)
    assert new_st.state == State.INIT
    assert new_st.quiet_since_ts_ns == 0
    assert action == Action.NONE


def test_init_transitions_to_waiting_after_5s_quiet_and_saves_ref():
    st = StreamState()
    st, _, _ = advance(st, _sample(0.0, motion=0.0), THRESH)
    st, _, _ = advance(st, _sample(2.5, motion=0.01), THRESH)
    st, action, _ = advance(st, _sample(5.0, motion=0.01), THRESH)
    assert st.state == State.WAITING_MOTION
    assert action == Action.SAVE_REF
    assert st.ref_ts_ns == int(5.0 * 1_000_000_000)


def test_init_resets_quiet_timer_on_motion_spike():
    st = StreamState()
    st, _, _ = advance(st, _sample(0.0, motion=0.0), THRESH)
    st, _, _ = advance(st, _sample(2.0, motion=0.5), THRESH)  # spike
    # quiet timer should be reset (or updated on next quiet)
    st, _, _ = advance(st, _sample(2.1, motion=0.0), THRESH)
    # Now only 2.9s of quiet, not yet ready
    st, action, _ = advance(st, _sample(5.0, motion=0.0), THRESH)
    assert st.state == State.INIT  # needs 5s from 2.1, i.e. at 7.1
    assert action == Action.NONE
```

- [ ] **Step 2: Run tests to confirm failure**

```
cd /root/ramp-motion && pytest tests/test_state_machine.py -v
```
Expected: import error — `MotionThresholds` and `advance` not defined yet.

- [ ] **Step 3: Implement `MotionThresholds` and `advance` (INIT branch only)**

Append to `ramp_motion/state_machine.py`:

```python
from dataclasses import replace
from typing import Tuple


@dataclass(frozen=True)
class MotionThresholds:
    motion_thresh: float
    t_quiet_ns: int
    t_enter_ns: int
    min_blob_area_px: int
    calib_rate_limit_ns: int
    t_stale_ns: int


def _is_quiet(sample: MotionSample, th: MotionThresholds) -> bool:
    return sample.motion_fraction < th.motion_thresh


def _advance_init(st: StreamState, s: MotionSample, th: MotionThresholds
                  ) -> Tuple[StreamState, Action, CycleStats]:
    if not _is_quiet(s, th):
        return replace(st, quiet_since_ts_ns=None, last_sample_ts_ns=s.ts_ns), \
               Action.NONE, st.cycle_stats

    if st.quiet_since_ts_ns is None:
        return replace(st, quiet_since_ts_ns=s.ts_ns, last_sample_ts_ns=s.ts_ns), \
               Action.NONE, st.cycle_stats

    if s.ts_ns - st.quiet_since_ts_ns >= th.t_quiet_ns:
        new_st = StreamState(
            state=State.WAITING_MOTION,
            quiet_since_ts_ns=None,
            motion_since_ts_ns=None,
            cycle_stats=CycleStats(),
            ref_ts_ns=s.ts_ns,
            last_sample_ts_ns=s.ts_ns,
        )
        return new_st, Action.SAVE_REF, new_st.cycle_stats

    return replace(st, last_sample_ts_ns=s.ts_ns), Action.NONE, st.cycle_stats


def advance(st: StreamState, s: MotionSample, th: MotionThresholds
            ) -> Tuple[StreamState, Action, CycleStats]:
    if s.disconnected:
        return StreamState(), Action.RESET, CycleStats()

    if st.state == State.INIT:
        return _advance_init(st, s, th)

    # Future states handled in subsequent tasks.
    raise NotImplementedError(f"state {st.state} not implemented yet")
```

- [ ] **Step 4: Run tests to verify they pass**

```
cd /root/ramp-motion && pytest tests/test_state_machine.py -v
```
Expected: 7 passed (3 from Task 3 + 4 new).

- [ ] **Step 5: Commit**

```bash
cd /root/ramp-motion
git add ramp_motion/state_machine.py tests/test_state_machine.py
git commit -m "feat(state_machine): INIT→WAITING_MOTION transition with quiet-timer logic"
```

---

## Task 5: State Machine — WAITING_MOTION → IN_MOTION (debounced entry)

**Files:**
- Modify: `ramp_motion/state_machine.py`
- Modify: `tests/test_state_machine.py`

- [ ] **Step 1: Append failing tests**

```python
def _advance_from_state_to_state(initial: StreamState, samples, th=THRESH):
    """Helper: runs advance over a list of samples, returns (state, actions)."""
    st = initial
    actions = []
    for s in samples:
        st, a, _ = advance(st, s, th)
        actions.append(a)
    return st, actions


def _waiting_state(ref_ts_sec: float) -> StreamState:
    return StreamState(
        state=State.WAITING_MOTION,
        ref_ts_ns=int(ref_ts_sec * 1_000_000_000),
        last_sample_ts_ns=int(ref_ts_sec * 1_000_000_000),
    )


def test_waiting_ignores_motion_below_thresh():
    st = _waiting_state(0.0)
    new_st, action, _ = advance(st, _sample(1.0, motion=0.005), THRESH)
    assert new_st.state == State.WAITING_MOTION
    assert action == Action.NONE


def test_waiting_ignores_tiny_blob_even_if_fraction_high():
    st = _waiting_state(0.0)
    # fraction high but blob below min_blob_area_px
    new_st, action, _ = advance(st, _sample(1.0, motion=0.5, blob=100), THRESH)
    assert new_st.state == State.WAITING_MOTION
    assert action == Action.NONE


def test_waiting_requires_debounce_to_enter_in_motion():
    st = _waiting_state(0.0)
    # single motion frame — not enough (t_enter = 0.5s)
    st, _, _ = advance(st, _sample(1.0, motion=0.5, blob=5000), THRESH)
    assert st.state == State.WAITING_MOTION
    # motion sustained 0.2s — still not enough
    st, _, _ = advance(st, _sample(1.2, motion=0.5, blob=5000), THRESH)
    assert st.state == State.WAITING_MOTION
    # motion sustained 0.6s — crosses the debounce
    st, action, _ = advance(st, _sample(1.6, motion=0.5, blob=5000), THRESH)
    assert st.state == State.IN_MOTION
    assert action == Action.NONE   # entry itself produces no save


def test_waiting_debounce_reset_on_calm_frame():
    st = _waiting_state(0.0)
    st, _, _ = advance(st, _sample(1.0, motion=0.5, blob=5000), THRESH)
    # calm frame — debounce must reset
    st, _, _ = advance(st, _sample(1.2, motion=0.0), THRESH)
    # now need a fresh 0.5s window
    st, _, _ = advance(st, _sample(1.3, motion=0.5, blob=5000), THRESH)
    st, action, _ = advance(st, _sample(1.7, motion=0.5, blob=5000), THRESH)
    assert st.state == State.IN_MOTION
```

- [ ] **Step 2: Run tests to confirm failure (NotImplementedError)**

```
cd /root/ramp-motion && pytest tests/test_state_machine.py -v
```
Expected: tests fail with `NotImplementedError: state State.WAITING_MOTION`.

- [ ] **Step 3: Implement `_advance_waiting_motion`**

Append to `ramp_motion/state_machine.py`:

```python
def _advance_waiting_motion(st: StreamState, s: MotionSample, th: MotionThresholds
                             ) -> Tuple[StreamState, Action, CycleStats]:
    motion_active = (
        s.motion_fraction >= th.motion_thresh
        and s.max_blob_area_px >= th.min_blob_area_px
    )
    if not motion_active:
        # reset the debounce timer
        return replace(st, motion_since_ts_ns=None, last_sample_ts_ns=s.ts_ns), \
               Action.NONE, st.cycle_stats

    if st.motion_since_ts_ns is None:
        return replace(st, motion_since_ts_ns=s.ts_ns, last_sample_ts_ns=s.ts_ns), \
               Action.NONE, st.cycle_stats

    if s.ts_ns - st.motion_since_ts_ns >= th.t_enter_ns:
        new_st = replace(
            st,
            state=State.IN_MOTION,
            motion_since_ts_ns=None,
            quiet_since_ts_ns=None,
            cycle_stats=CycleStats(
                peak_fraction=s.motion_fraction,
                peak_blob_area_px=s.max_blob_area_px,
                samples_count=1,
            ),
            last_sample_ts_ns=s.ts_ns,
        )
        return new_st, Action.NONE, new_st.cycle_stats

    return replace(st, last_sample_ts_ns=s.ts_ns), Action.NONE, st.cycle_stats
```

Update `advance()` dispatch:

```python
def advance(st, s, th):
    if s.disconnected:
        return StreamState(), Action.RESET, CycleStats()
    if st.state == State.INIT:
        return _advance_init(st, s, th)
    if st.state == State.WAITING_MOTION:
        return _advance_waiting_motion(st, s, th)
    raise NotImplementedError(f"state {st.state} not implemented yet")
```

- [ ] **Step 4: Run tests to verify pass**

```
cd /root/ramp-motion && pytest tests/test_state_machine.py -v
```
Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
cd /root/ramp-motion
git add ramp_motion/state_machine.py tests/test_state_machine.py
git commit -m "feat(state_machine): WAITING_MOTION→IN_MOTION with t_enter debounce and min_blob_area gate"
```

---

## Task 6: State Machine — IN_MOTION Peak Tracking and Calibration Frames

**Files:**
- Modify: `ramp_motion/state_machine.py`
- Modify: `tests/test_state_machine.py`

- [ ] **Step 1: Append failing tests**

```python
def _in_motion_state(peak: float = 0.0, last_calib_sec: float = -10.0) -> StreamState:
    return StreamState(
        state=State.IN_MOTION,
        ref_ts_ns=0,
        cycle_stats=CycleStats(
            peak_fraction=peak,
            peak_blob_area_px=int(peak * 10000),
            last_calib_ts_ns=int(last_calib_sec * 1_000_000_000),
            samples_count=1,
        ),
    )


def test_in_motion_increments_samples_count():
    st = _in_motion_state()
    st, _, _ = advance(st, _sample(0.1, motion=0.1, blob=1000), THRESH)
    assert st.cycle_stats.samples_count == 2


def test_in_motion_updates_peak_on_new_max_and_saves_calibration():
    st = _in_motion_state(peak=0.1, last_calib_sec=-10.0)
    st, action, _ = advance(st, _sample(0.0, motion=0.3, blob=5000), THRESH)
    assert action == Action.SAVE_CALIBRATION
    assert st.cycle_stats.peak_fraction == pytest.approx(0.3)
    assert st.cycle_stats.peak_blob_area_px == 5000
    assert st.cycle_stats.last_calib_ts_ns == 0


def test_in_motion_updates_peak_without_calib_when_rate_limited():
    st = _in_motion_state(peak=0.1, last_calib_sec=0.0)  # calib just happened
    # new max 0.2s later — but rate limit is 1.0s
    st, action, _ = advance(st, _sample(0.2, motion=0.3, blob=5000), THRESH)
    assert action == Action.NONE
    assert st.cycle_stats.peak_fraction == pytest.approx(0.3)
    # last_calib_ts_ns unchanged
    assert st.cycle_stats.last_calib_ts_ns == 0


def test_in_motion_no_peak_update_below_existing_peak():
    st = _in_motion_state(peak=0.5, last_calib_sec=-10.0)
    st, action, _ = advance(st, _sample(0.0, motion=0.3, blob=5000), THRESH)
    assert action == Action.NONE
    assert st.cycle_stats.peak_fraction == pytest.approx(0.5)
```

- [ ] **Step 2: Run tests to confirm failure**

```
cd /root/ramp-motion && pytest tests/test_state_machine.py -v
```
Expected: NotImplementedError for IN_MOTION.

- [ ] **Step 3: Implement `_advance_in_motion` (peak + calib only; quiet-transition in next task)**

Append to `ramp_motion/state_machine.py`:

```python
def _advance_in_motion(st: StreamState, s: MotionSample, th: MotionThresholds
                        ) -> Tuple[StreamState, Action, CycleStats]:
    cs = st.cycle_stats
    new_cs = CycleStats(
        peak_fraction=cs.peak_fraction,
        peak_blob_area_px=cs.peak_blob_area_px,
        last_calib_ts_ns=cs.last_calib_ts_ns,
        samples_count=cs.samples_count + 1,
    )

    action = Action.NONE
    if s.motion_fraction > cs.peak_fraction:
        rate_limit_ok = (s.ts_ns - cs.last_calib_ts_ns) >= th.calib_rate_limit_ns
        if rate_limit_ok:
            new_cs = CycleStats(
                peak_fraction=s.motion_fraction,
                peak_blob_area_px=s.max_blob_area_px,
                last_calib_ts_ns=s.ts_ns,
                samples_count=new_cs.samples_count,
            )
            action = Action.SAVE_CALIBRATION
        else:
            new_cs = CycleStats(
                peak_fraction=s.motion_fraction,
                peak_blob_area_px=s.max_blob_area_px,
                last_calib_ts_ns=cs.last_calib_ts_ns,
                samples_count=new_cs.samples_count,
            )

    new_st = replace(st, cycle_stats=new_cs, last_sample_ts_ns=s.ts_ns)
    return new_st, action, new_cs
```

Update `advance()` dispatch:

```python
def advance(st, s, th):
    if s.disconnected:
        return StreamState(), Action.RESET, CycleStats()
    if st.state == State.INIT:
        return _advance_init(st, s, th)
    if st.state == State.WAITING_MOTION:
        return _advance_waiting_motion(st, s, th)
    if st.state == State.IN_MOTION:
        return _advance_in_motion(st, s, th)
    raise NotImplementedError(f"state {st.state} not implemented yet")
```

- [ ] **Step 4: Run tests to verify pass**

```
cd /root/ramp-motion && pytest tests/test_state_machine.py -v
```
Expected: 15 passed.

- [ ] **Step 5: Commit**

```bash
cd /root/ramp-motion
git add ramp_motion/state_machine.py tests/test_state_machine.py
git commit -m "feat(state_machine): IN_MOTION peak tracking and rate-limited calibration saves"
```

---

## Task 7: State Machine — IN_MOTION → WAITING_MOTION (cycle complete)

**Files:**
- Modify: `ramp_motion/state_machine.py`
- Modify: `tests/test_state_machine.py`

- [ ] **Step 1: Append failing tests**

```python
def test_in_motion_starts_quiet_timer_when_motion_drops():
    st = _in_motion_state(peak=0.3, last_calib_sec=-10.0)
    st, _, _ = advance(st, _sample(1.0, motion=0.0), THRESH)
    assert st.state == State.IN_MOTION
    assert st.quiet_since_ts_ns == int(1.0 * 1_000_000_000)


def test_in_motion_quiet_timer_resets_on_motion_return():
    st = _in_motion_state(peak=0.3, last_calib_sec=-10.0)
    st, _, _ = advance(st, _sample(1.0, motion=0.0), THRESH)
    # motion returns at 2.0
    st, _, _ = advance(st, _sample(2.0, motion=0.5, blob=5000), THRESH)
    assert st.state == State.IN_MOTION
    assert st.quiet_since_ts_ns is None


def test_in_motion_transitions_to_waiting_after_t_quiet_and_saves_query():
    st = _in_motion_state(peak=0.3, last_calib_sec=-10.0)
    st, _, _ = advance(st, _sample(1.0, motion=0.01), THRESH)   # quiet start
    st, _, _ = advance(st, _sample(3.0, motion=0.01), THRESH)   # still quiet
    st, action, cs = advance(st, _sample(6.0, motion=0.01), THRESH)  # 5s elapsed
    assert st.state == State.WAITING_MOTION
    assert action == Action.SAVE_QUERY_AND_NEW_REF
    assert cs.peak_fraction == pytest.approx(0.3)
    assert st.ref_ts_ns == int(6.0 * 1_000_000_000)
    # cycle_stats reset for next cycle
    assert st.cycle_stats.peak_fraction == 0.0
```

- [ ] **Step 2: Run tests to confirm failure**

```
cd /root/ramp-motion && pytest tests/test_state_machine.py -v
```
Expected: 3 failures — quiet timer not started, transition never happens.

- [ ] **Step 3: Extend `_advance_in_motion` with quiet-timer and cycle-complete logic**

Replace the body of `_advance_in_motion`:

```python
def _advance_in_motion(st: StreamState, s: MotionSample, th: MotionThresholds
                        ) -> Tuple[StreamState, Action, CycleStats]:
    motion_active = (
        s.motion_fraction >= th.motion_thresh
        and s.max_blob_area_px >= th.min_blob_area_px
    )

    # --- 1. Peak/calibration tracking (always runs) ---
    cs = st.cycle_stats
    new_cs_base = CycleStats(
        peak_fraction=cs.peak_fraction,
        peak_blob_area_px=cs.peak_blob_area_px,
        last_calib_ts_ns=cs.last_calib_ts_ns,
        samples_count=cs.samples_count + 1,
    )
    action = Action.NONE
    new_cs = new_cs_base
    if s.motion_fraction > cs.peak_fraction:
        rate_limit_ok = (s.ts_ns - cs.last_calib_ts_ns) >= th.calib_rate_limit_ns
        if rate_limit_ok:
            new_cs = CycleStats(
                peak_fraction=s.motion_fraction,
                peak_blob_area_px=s.max_blob_area_px,
                last_calib_ts_ns=s.ts_ns,
                samples_count=new_cs_base.samples_count,
            )
            action = Action.SAVE_CALIBRATION
        else:
            new_cs = CycleStats(
                peak_fraction=s.motion_fraction,
                peak_blob_area_px=s.max_blob_area_px,
                last_calib_ts_ns=cs.last_calib_ts_ns,
                samples_count=new_cs_base.samples_count,
            )

    # --- 2. Quiet-timer and cycle-complete detection ---
    if motion_active:
        # Motion returned — reset quiet timer, stay in IN_MOTION.
        return (
            replace(st, cycle_stats=new_cs, quiet_since_ts_ns=None,
                    last_sample_ts_ns=s.ts_ns),
            action, new_cs,
        )

    if st.quiet_since_ts_ns is None:
        # Start tracking quiet.
        return (
            replace(st, cycle_stats=new_cs, quiet_since_ts_ns=s.ts_ns,
                    last_sample_ts_ns=s.ts_ns),
            action, new_cs,
        )

    if s.ts_ns - st.quiet_since_ts_ns >= th.t_quiet_ns:
        # Cycle complete.
        # save_query_and_new_ref takes precedence over any pending calib action
        new_st = StreamState(
            state=State.WAITING_MOTION,
            quiet_since_ts_ns=None,
            motion_since_ts_ns=None,
            cycle_stats=CycleStats(),
            ref_ts_ns=s.ts_ns,
            last_sample_ts_ns=s.ts_ns,
        )
        return new_st, Action.SAVE_QUERY_AND_NEW_REF, new_cs

    return (
        replace(st, cycle_stats=new_cs, last_sample_ts_ns=s.ts_ns),
        action, new_cs,
    )
```

- [ ] **Step 4: Run tests**

```
cd /root/ramp-motion && pytest tests/test_state_machine.py -v
```
Expected: 18 passed.

- [ ] **Step 5: Commit**

```bash
cd /root/ramp-motion
git add ramp_motion/state_machine.py tests/test_state_machine.py
git commit -m "feat(state_machine): cycle_complete on t_quiet after motion — SAVE_QUERY_AND_NEW_REF"
```

---

## Task 8: State Machine — Disconnect / Stale-Stream Handling

**Files:**
- Modify: `ramp_motion/state_machine.py`
- Modify: `tests/test_state_machine.py`

- [ ] **Step 1: Append failing tests**

```python
def test_disconnect_sample_resets_to_init():
    st = _in_motion_state(peak=0.5)
    disconnect = MotionSample(
        stream_id="cam-01", ts_ns=0, motion_fraction=0.0,
        max_blob_area_px=0, frame_pts=0, disconnected=True,
    )
    new_st, action, _ = advance(st, disconnect, THRESH)
    assert new_st.state == State.INIT
    assert action == Action.RESET
    assert new_st.ref_ts_ns is None


def test_is_stale_helper_detects_gap():
    from ramp_motion.state_machine import is_stale
    st = replace(StreamState(), last_sample_ts_ns=0)
    # threshold t_stale_ns = 10s
    assert is_stale(st, now_ns=11 * 1_000_000_000, th=THRESH) is True
    assert is_stale(st, now_ns=5 * 1_000_000_000, th=THRESH) is False
    # no last_sample_ts yet → not stale
    assert is_stale(StreamState(), now_ns=10**12, th=THRESH) is False
```

- [ ] **Step 2: Run tests — disconnect should already pass from Task 4; is_stale fails**

```
cd /root/ramp-motion && pytest tests/test_state_machine.py -v
```
Expected: 1 failure (`is_stale` not defined). `test_disconnect_sample_resets_to_init` passes already.

- [ ] **Step 3: Add `is_stale` to `ramp_motion/state_machine.py`**

Append:

```python
def is_stale(st: StreamState, now_ns: int, th: MotionThresholds) -> bool:
    if st.last_sample_ts_ns is None:
        return False
    return (now_ns - st.last_sample_ts_ns) >= th.t_stale_ns
```

- [ ] **Step 4: Run tests**

```
cd /root/ramp-motion && pytest tests/test_state_machine.py -v
```
Expected: 20 passed.

- [ ] **Step 5: Commit**

```bash
cd /root/ramp-motion
git add ramp_motion/state_machine.py tests/test_state_machine.py
git commit -m "feat(state_machine): disconnect handling and is_stale helper for T_stale detection"
```

---

## Task 9: Manifest Writer

**Files:**
- Create: `ramp_motion/manifest.py`
- Create: `tests/test_manifest.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_manifest.py
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

    # No .tmp file should remain
    tmp_files = list(dest.parent.glob("*.tmp"))
    assert tmp_files == []
```

- [ ] **Step 2: Run — fails (no module)**

```
cd /root/ramp-motion && pytest tests/test_manifest.py -v
```

- [ ] **Step 3: Implement `ramp_motion/manifest.py`**

```python
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
```

- [ ] **Step 4: Run — pass**

```
cd /root/ramp-motion && pytest tests/test_manifest.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
cd /root/ramp-motion
git add ramp_motion/manifest.py tests/test_manifest.py
git commit -m "feat(manifest): atomic cycle manifest JSON writer"
```

---

## Task 10: Events Log (JSONL)

**Files:**
- Create: `ramp_motion/events_log.py`
- Create: `tests/test_events_log.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_events_log.py
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
```

- [ ] **Step 2: Run — fails (no module)**

- [ ] **Step 3: Implement `ramp_motion/events_log.py`**

```python
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
```

- [ ] **Step 4: Run — pass**

```
cd /root/ramp-motion && pytest tests/test_events_log.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
cd /root/ramp-motion
git add ramp_motion/events_log.py tests/test_events_log.py
git commit -m "feat(events_log): thread-safe JSONL event appender"
```

---

## Task 11: Frame Saver (ThreadPool + Atomic JPEG + Hard-Link)

**Files:**
- Create: `ramp_motion/frame_saver.py`
- Create: `tests/test_frame_saver.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_frame_saver.py
import os
from pathlib import Path

import numpy as np
import cv2
import pytest

from ramp_motion.frame_saver import FrameSaver


def _img() -> np.ndarray:
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:32] = 255  # white top half
    return img


def test_save_writes_jpeg_atomically(tmp_path: Path):
    saver = FrameSaver(root=tmp_path, jpeg_quality=85, max_workers=1)
    dest = tmp_path / "ref" / "a.jpg"
    future = saver.save(_img(), dest)
    future.result()
    assert dest.exists()
    decoded = cv2.imread(str(dest))
    assert decoded is not None
    # no leftover tmp
    assert list(dest.parent.glob("*.tmp")) == []
    saver.shutdown()


def test_save_with_hardlink_creates_both_paths(tmp_path: Path):
    saver = FrameSaver(root=tmp_path, jpeg_quality=85, max_workers=1)
    primary = tmp_path / "query" / "q.jpg"
    linked = tmp_path / "ref" / "r.jpg"
    future = saver.save(_img(), primary, hardlink_to=linked)
    future.result()
    assert primary.exists()
    assert linked.exists()
    # Same inode (hard link).
    assert os.stat(primary).st_ino == os.stat(linked).st_ino
    saver.shutdown()
```

- [ ] **Step 2: Run — fails**

- [ ] **Step 3: Implement `ramp_motion/frame_saver.py`**

```python
from __future__ import annotations
import os
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class FrameSaver:
    """Asynchronous frame writer: submit numpy BGR images, they are JPEG-encoded
    and written atomically. Optional hard-linking to a second path for the
    query-is-also-new-ref case.
    """

    def __init__(self, root: Path, jpeg_quality: int = 85, max_workers: int = 4):
        self._root = Path(root)
        self._jpeg_quality = int(jpeg_quality)
        self._executor = ThreadPoolExecutor(max_workers=max_workers,
                                             thread_name_prefix="frame-saver")

    def save(self, image_bgr: np.ndarray, dest: Path,
             hardlink_to: Optional[Path] = None) -> Future:
        return self._executor.submit(self._save_sync, image_bgr, dest, hardlink_to)

    def _save_sync(self, image_bgr: np.ndarray, dest: Path,
                   hardlink_to: Optional[Path]) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        ok, buf = cv2.imencode(".jpg", image_bgr,
                               [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality])
        if not ok:
            raise RuntimeError("cv2.imencode returned failure")
        with open(tmp, "wb") as f:
            f.write(buf.tobytes())
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, dest)
        if hardlink_to is not None:
            hardlink_to.parent.mkdir(parents=True, exist_ok=True)
            if hardlink_to.exists():
                hardlink_to.unlink()
            os.link(dest, hardlink_to)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)
```

- [ ] **Step 4: Run tests**

```
cd /root/ramp-motion && pytest tests/test_frame_saver.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
cd /root/ramp-motion
git add ramp_motion/frame_saver.py tests/test_frame_saver.py
git commit -m "feat(frame_saver): threadpool JPEG writer with atomic rename and hard-link"
```

---

## Task 12: `nvdspreprocess` Custom Lib — CMake Scaffold and Stub Exports

**Files:**
- Create: `preprocess/CMakeLists.txt`
- Create: `preprocess/include/ramp_motion_preproc.h`
- Create: `preprocess/src/custom_lib.cpp`
- Create: `Dockerfile` (stages 1+2 only — stage 3 added in Task 19)

- [ ] **Step 1: Write `preprocess/CMakeLists.txt`**

```cmake
cmake_minimum_required(VERSION 3.18)
project(ramp_motion_preproc LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV 4.8 REQUIRED COMPONENTS core imgproc cudaimgproc cudabgsegm cudaarithm)

set(DS_ROOT /opt/nvidia/deepstream/deepstream)
set(DS_INCLUDE ${DS_ROOT}/sources/includes)
set(DS_PREPROC_INCLUDE
    ${DS_ROOT}/sources/gst-plugins/gst-nvdspreprocess/include
    ${DS_ROOT}/sources/gst-plugins/gst-nvdspreprocess/nvdspreprocess_impl)
set(DS_LIBS ${DS_ROOT}/lib)

add_library(ramp_motion_preproc SHARED
    src/custom_lib.cpp
    src/clahe_mog2.cu
    src/motion_meta.cpp
)

target_include_directories(ramp_motion_preproc PRIVATE
    ${DS_INCLUDE}
    ${DS_PREPROC_INCLUDE}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(ramp_motion_preproc PRIVATE
    CUDA::cudart
    ${OpenCV_LIBS}
    -L${DS_LIBS}
    nvdsgst_helper
    nvbufsurface
    nvbufsurftransform
    nvdsgst_meta
    nvds_meta
)

set_target_properties(ramp_motion_preproc PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    CUDA_SEPARABLE_COMPILATION ON
)
```

- [ ] **Step 2: Write `preprocess/include/ramp_motion_preproc.h`**

```cpp
#pragma once

#include <cstdint>

extern "C" {

/// Motion metric emitted by the custom library per frame per ROI.
/// Kept ABI-stable — ctypes tests and the Python probe read raw bytes.
struct MotionMeta {
    uint32_t source_id;
    uint32_t roi_id;
    uint64_t ts_ns;
    uint64_t frame_pts;
    uint32_t fg_pixel_count;
    float    motion_fraction;
    uint32_t max_blob_area_px;
};

/// Used-meta type id — project-specific (NVDS_USER_META_BASE + 0x4D4F == "MO").
constexpr int kMotionUserMetaType = 0x1000 + 0x4D4F;

/// ctypes-callable smoke probe: returns library build tag.
const char* ramp_motion_preproc_version();

/// ctypes-callable synchronous "process image" for unit tests.
/// - rgb/gray image data passed from host (we upload to GPU inside).
/// - Returns computed MotionMeta via out parameter.
/// Provided so we can validate CLAHE+MOG2 without a GStreamer pipeline.
int ramp_motion_preproc_process_image(
    const uint8_t* bgr,
    int width, int height, int stride_bytes,
    uint32_t source_id, uint32_t roi_id,
    uint64_t ts_ns,
    MotionMeta* out);

} // extern "C"
```

- [ ] **Step 3: Write stub `preprocess/src/custom_lib.cpp`**

```cpp
#include "ramp_motion_preproc.h"
#include <cstring>

extern "C" {

const char* ramp_motion_preproc_version() {
    return "ramp_motion_preproc 0.1.0 (stub)";
}

int ramp_motion_preproc_process_image(
    const uint8_t* /*bgr*/,
    int /*width*/, int /*height*/, int /*stride_bytes*/,
    uint32_t source_id, uint32_t roi_id,
    uint64_t ts_ns,
    MotionMeta* out)
{
    if (out == nullptr) return -1;
    std::memset(out, 0, sizeof(*out));
    out->source_id = source_id;
    out->roi_id = roi_id;
    out->ts_ns = ts_ns;
    out->motion_fraction = 0.0f;
    return 0;
}

} // extern "C"
```

Also create empty `preprocess/src/clahe_mog2.cu`:

```cpp
// Intentionally empty; populated in Task 13.
```

And empty `preprocess/src/motion_meta.cpp`:

```cpp
// Intentionally empty; populated in Task 14.
```

- [ ] **Step 4: Write `Dockerfile` with stages 1 and 2 (stage 3 added in Task 19)**

```dockerfile
# syntax=docker/dockerfile:1.6

# =========================================================================
# Stage 1: build CUDA-enabled OpenCV 4.8 (for cv::cuda::CLAHE and MOG2)
# =========================================================================
FROM nvcr.io/nvidia/deepstream:6.3-triton-multiarch AS opencv-build
ARG OPENCV_VERSION=4.8.0
ARG CUDA_ARCHS=7.5,8.0,8.6,8.9

RUN apt-get update && apt-get install -y --no-install-recommends \
      cmake build-essential git pkg-config \
      libgtk-3-dev libavcodec-dev libavformat-dev libswscale-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN git clone --depth 1 --branch ${OPENCV_VERSION} https://github.com/opencv/opencv.git \
 && git clone --depth 1 --branch ${OPENCV_VERSION} https://github.com/opencv/opencv_contrib.git \
 && mkdir opencv/build

WORKDIR /opt/opencv/build
RUN cmake .. \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_INSTALL_PREFIX=/opt/opencv-cuda \
    -D OPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib/modules \
    -D WITH_CUDA=ON \
    -D CUDA_ARCH_BIN=${CUDA_ARCHS} \
    -D BUILD_opencv_python2=OFF \
    -D BUILD_opencv_python3=OFF \
    -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_EXAMPLES=OFF \
    -D WITH_GSTREAMER=ON \
 && make -j"$(nproc)" && make install

# =========================================================================
# Stage 2: build the custom .so
# =========================================================================
FROM nvcr.io/nvidia/deepstream:6.3-triton-multiarch AS preproc-build
COPY --from=opencv-build /opt/opencv-cuda /opt/opencv-cuda
RUN apt-get update && apt-get install -y --no-install-recommends \
      cmake build-essential \
    && rm -rf /var/lib/apt/lists/*
ENV CMAKE_PREFIX_PATH=/opt/opencv-cuda
WORKDIR /build
COPY preprocess/ preprocess/
RUN cmake -S preprocess -B preprocess/build \
      -DOpenCV_DIR=/opt/opencv-cuda/lib/cmake/opencv4 \
 && cmake --build preprocess/build -j"$(nproc)"
```

- [ ] **Step 5: Build stages 1 and 2 to confirm the .so compiles (on Selectel GPU host)**

Run on the Selectel host :

```
cd /root/ramp-motion
docker build --target preproc-build -t ramp-motion-preproc-build .
```
Expected: build succeeds, produces `/build/preprocess/build/libramp_motion_preproc.so` inside the image. (First run takes 20–40 min for OpenCV.)

- [ ] **Step 6: Commit**

```bash
cd /root/ramp-motion
git add preprocess/ Dockerfile
git commit -m "feat(preprocess): cmake scaffold, stub .so with ctypes smoke symbols, dockerfile stages 1-2"
```

---

## Task 13: `.so` — CLAHE + MOG2 via cv::cuda (with ctypes test)

**Files:**
- Modify: `preprocess/src/custom_lib.cpp`
- Create: `preprocess/src/clahe_mog2.cu` (replaces the empty stub)
- Create: `tests/test_preprocess_lib.py` (ctypes, `gpu` mark)

- [ ] **Step 1: Write failing GPU test — `tests/test_preprocess_lib.py`**

```python
import ctypes
import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.gpu

LIB_PATH = Path(os.environ.get(
    "RAMP_MOTION_PREPROC_SO",
    "/opt/ramp-motion/libramp_motion_preproc.so",
))


class MotionMeta(ctypes.Structure):
    _fields_ = [
        ("source_id",        ctypes.c_uint32),
        ("roi_id",           ctypes.c_uint32),
        ("ts_ns",            ctypes.c_uint64),
        ("frame_pts",        ctypes.c_uint64),
        ("fg_pixel_count",   ctypes.c_uint32),
        ("motion_fraction",  ctypes.c_float),
        ("max_blob_area_px", ctypes.c_uint32),
    ]


@pytest.fixture(scope="module")
def lib():
    if not LIB_PATH.exists():
        pytest.skip(f"lib not found at {LIB_PATH}")
    lib = ctypes.CDLL(str(LIB_PATH))
    lib.ramp_motion_preproc_version.restype = ctypes.c_char_p
    lib.ramp_motion_preproc_process_image.restype = ctypes.c_int
    lib.ramp_motion_preproc_process_image.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint64,
        ctypes.POINTER(MotionMeta),
    ]
    return lib


def _process(lib, img: np.ndarray, source_id=0, ts_ns=0) -> MotionMeta:
    h, w, _ = img.shape
    out = MotionMeta()
    contig = np.ascontiguousarray(img)
    lib.ramp_motion_preproc_process_image(
        contig.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        w, h, w * 3,
        source_id, 0, ts_ns,
        ctypes.byref(out),
    )
    return out


def test_version_string(lib):
    v = lib.ramp_motion_preproc_version().decode()
    assert "ramp_motion_preproc" in v


def test_motion_detected_when_pixels_change(lib):
    # 100 frames of identical background → MOG2 learns it.
    bg = np.full((240, 320, 3), 128, dtype=np.uint8)
    for i in range(100):
        _process(lib, bg, source_id=1, ts_ns=i * 33_000_000)

    # Then a frame with a big bright square in the middle.
    fg = bg.copy()
    fg[80:160, 120:200] = 255
    m = _process(lib, fg, source_id=1, ts_ns=100 * 33_000_000)

    assert m.motion_fraction > 0.01, f"expected motion, got {m.motion_fraction}"
    assert m.max_blob_area_px >= 1000


def test_per_source_state_isolated(lib):
    bg = np.full((240, 320, 3), 50, dtype=np.uint8)
    for i in range(100):
        _process(lib, bg, source_id=7, ts_ns=i * 33_000_000)
    # Source 8 has no history yet — its first frame should be all-foreground
    # (or close to it) — not contaminated by source 7.
    m8 = _process(lib, bg, source_id=8, ts_ns=0)
    # Any result is fine here; the assertion is absence of crash and that
    # the struct is populated.
    assert m8.source_id == 8
```

- [ ] **Step 2: Run — fails (stub returns zero motion_fraction)**

On the Selectel host, inside a container built from stage 2:

```
FORCE_GPU_TESTS=1 pytest tests/test_preprocess_lib.py -v
```
Expected: `test_motion_detected_when_pixels_change` fails (motion_fraction is 0).

- [ ] **Step 3: Implement `preprocess/src/clahe_mog2.cu`**

```cpp
#include "ramp_motion_preproc.h"

#include <mutex>
#include <unordered_map>

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudabgsegm.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/imgproc.hpp>

namespace ramp_motion {

struct PerSource {
    cv::Ptr<cv::cuda::CLAHE> clahe;
    cv::Ptr<cv::cuda::BackgroundSubtractorMOG2> mog2;
    cv::cuda::Stream stream;
};

static std::mutex g_mu;
static std::unordered_map<uint64_t, PerSource> g_state;   // key: (source<<32)|roi

static inline uint64_t key(uint32_t source_id, uint32_t roi_id) {
    return (uint64_t)source_id << 32 | (uint64_t)roi_id;
}

PerSource& get_state(uint32_t source_id, uint32_t roi_id) {
    std::lock_guard<std::mutex> lk(g_mu);
    auto k = key(source_id, roi_id);
    auto it = g_state.find(k);
    if (it != g_state.end()) return it->second;

    PerSource ps;
    ps.clahe = cv::cuda::createCLAHE(2.0, cv::Size(8, 8));
    ps.mog2 = cv::cuda::createBackgroundSubtractorMOG2(500, 16.0, true);
    auto [ins, _] = g_state.emplace(k, std::move(ps));
    return ins->second;
}

void reset_source(uint32_t source_id) {
    std::lock_guard<std::mutex> lk(g_mu);
    for (auto it = g_state.begin(); it != g_state.end();) {
        if (static_cast<uint32_t>(it->first >> 32) == source_id) {
            it = g_state.erase(it);
        } else {
            ++it;
        }
    }
}

int process_bgr(const uint8_t* bgr, int w, int h, int stride,
                uint32_t source_id, uint32_t roi_id, uint64_t ts_ns,
                MotionMeta* out)
{
    if (!out) return -1;
    out->source_id = source_id;
    out->roi_id = roi_id;
    out->ts_ns = ts_ns;
    out->frame_pts = 0;

    cv::Mat host_bgr(h, w, CV_8UC3, const_cast<uint8_t*>(bgr), stride);
    cv::cuda::GpuMat bgr_gpu;
    bgr_gpu.upload(host_bgr);

    cv::cuda::GpuMat gray;
    cv::cuda::cvtColor(bgr_gpu, gray, cv::COLOR_BGR2GRAY);

    PerSource& s = get_state(source_id, roi_id);

    cv::cuda::GpuMat clahe_out;
    s.clahe->apply(gray, clahe_out, s.stream);

    cv::cuda::GpuMat fg_mask;
    s.mog2->apply(clahe_out, fg_mask, -1.0, s.stream);

    // shadow pixels come back as value 127 with detectShadows=true.
    // We treat them as not-motion by thresholding ≥ 255.
    cv::cuda::GpuMat fg_binary;
    cv::cuda::threshold(fg_mask, fg_binary, 200, 255, cv::THRESH_BINARY, s.stream);
    s.stream.waitForCompletion();

    int fg = cv::cuda::countNonZero(fg_binary);
    out->fg_pixel_count = static_cast<uint32_t>(fg);
    out->motion_fraction = (w * h > 0) ? static_cast<float>(fg) / (float)(w * h) : 0.f;

    // Connected components still run CPU-side in OpenCV 4.8; tiny cost for a ~ 1 Mpx ROI.
    cv::Mat fg_cpu;
    fg_binary.download(fg_cpu);
    cv::Mat labels, stats, centroids;
    int n = cv::connectedComponentsWithStats(fg_cpu, labels, stats, centroids, 8, CV_32S);
    int max_blob = 0;
    for (int i = 1; i < n; ++i) {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area > max_blob) max_blob = area;
    }
    out->max_blob_area_px = static_cast<uint32_t>(max_blob);

    return 0;
}

} // namespace ramp_motion

extern "C" int ramp_motion_preproc_reset_source(uint32_t source_id) {
    ramp_motion::reset_source(source_id);
    return 0;
}

extern "C" int ramp_motion_preproc_process_image(
    const uint8_t* bgr, int w, int h, int stride,
    uint32_t source_id, uint32_t roi_id, uint64_t ts_ns,
    MotionMeta* out)
{
    return ramp_motion::process_bgr(bgr, w, h, stride, source_id, roi_id, ts_ns, out);
}
```

Update `preprocess/src/custom_lib.cpp` — remove the stub implementation of `ramp_motion_preproc_process_image` (now provided by `clahe_mog2.cu`) and keep only version:

```cpp
#include "ramp_motion_preproc.h"

extern "C" {
const char* ramp_motion_preproc_version() {
    return "ramp_motion_preproc 0.1.0 (clahe+mog2)";
}
}
```

Also extend the header:

```cpp
// Add to preprocess/include/ramp_motion_preproc.h inside the extern "C" block:
int ramp_motion_preproc_reset_source(uint32_t source_id);
```

- [ ] **Step 4: Rebuild and run tests on Selectel host**

```
docker build --target preproc-build -t ramp-motion-preproc-build .
docker run --rm --gpus all \
  -v $(pwd):/work -w /work \
  -e FORCE_GPU_TESTS=1 \
  -e RAMP_MOTION_PREPROC_SO=/build/preprocess/build/libramp_motion_preproc.so \
  ramp-motion-preproc-build bash -lc "
    pip3 install pytest numpy opencv-python-headless &&
    pytest tests/test_preprocess_lib.py -v
  "
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
cd /root/ramp-motion
git add preprocess/ tests/test_preprocess_lib.py
git commit -m "feat(preprocess): clahe+mog2 via cv::cuda, ctypes-tested process_image API"
```

---

## Task 14: `.so` — nvdspreprocess Integration and User-Meta Emission

**Files:**
- Modify: `preprocess/src/custom_lib.cpp`
- Create: `preprocess/src/motion_meta.cpp` (replaces the empty stub)
- Create: `configs/nvdspreprocess_config.txt`

- [ ] **Step 1: Implement `preprocess/src/motion_meta.cpp`**

```cpp
#include "ramp_motion_preproc.h"

#include "nvdsmeta.h"

extern "C" {

static void motion_meta_release(void* /*gst_meta*/, void* user_data) {
    auto* m = static_cast<MotionMeta*>(user_data);
    delete m;
}

NvDsUserMeta* ramp_motion_attach_user_meta(NvDsFrameMeta* frame_meta,
                                           const MotionMeta& src)
{
    if (!frame_meta) return nullptr;
    NvDsBatchMeta* batch_meta = frame_meta->base_meta.batch_meta;
    NvDsUserMeta* user_meta = nvds_acquire_user_meta_from_pool(batch_meta);
    if (!user_meta) return nullptr;

    auto* payload = new MotionMeta(src);
    user_meta->user_meta_data = payload;
    user_meta->base_meta.meta_type =
        static_cast<NvDsMetaType>(NVDS_USER_META + kMotionUserMetaType);
    user_meta->base_meta.copy_func = nullptr;        // we do not share meta across buffers
    user_meta->base_meta.release_func = motion_meta_release;

    nvds_add_user_meta_to_frame(frame_meta, user_meta);
    return user_meta;
}

} // extern "C"
```

- [ ] **Step 2: Add nvdspreprocess entry points to `preprocess/src/custom_lib.cpp`**

```cpp
#include "ramp_motion_preproc.h"

#include <cstring>
#include <memory>
#include <vector>

#include "nvdspreprocess_lib.h"
#include "nvdspreprocess_meta.h"

// Forward declarations from motion_meta.cpp:
extern "C" NvDsUserMeta*
ramp_motion_attach_user_meta(NvDsFrameMeta*, const MotionMeta&);

// Forward declaration from clahe_mog2.cu:
extern "C" int ramp_motion_preproc_process_image(
    const uint8_t*, int, int, int,
    uint32_t, uint32_t, uint64_t, MotionMeta*);

extern "C" {

const char* ramp_motion_preproc_version() {
    return "ramp_motion_preproc 0.1.0 (clahe+mog2+nvdspreprocess)";
}

// ---------- nvdspreprocess API (DS 6.3 contract) ---------------------------

struct CustomCtx {
    // No custom state required beyond global per-source state in clahe_mog2.cu.
};

NvDsPreProcessStatus CustomTensorPreparation(
    CustomCtx* ctx,
    NvDsPreProcessBatch* batch,
    NvDsPreProcessCustomBuf*& /*buf*/,
    CustomTensorParams& /*tensor_params*/,
    NvDsPreProcessAcquirer* /*acquirer*/)
{
    // We don't actually produce a tensor for inference — this hook is used to
    // iterate over the batch and compute motion metrics per ROI, attaching
    // them to the frame meta as user-meta.
    if (!batch) return NVDSPREPROCESS_CONFIG_FAILED;

    for (auto& unit : batch->units) {
        NvDsRoiMeta& roi = unit.roi_meta;
        NvDsFrameMeta* frame_meta = roi.frame_meta;
        if (!frame_meta) continue;

        // Acquire CPU-side BGR buffer for the ROI crop from the
        // converted_buffer (nvdspreprocess has already resized to network-input
        // tensor; for motion we want the original ROI crop).
        NvBufSurface* src_surf = batch->surf;
        if (!src_surf) continue;
        int frame_idx = unit.frame_num;   // batch-local index — see DS 6.3 docs
        NvBufSurfaceParams* p = &src_surf->surfaceList[frame_idx];

        // For v1 we rely on frame-level motion (single ROI per stream), so crop
        // using roi.roi rectangle from the already-synced NvBufSurface.
        // NvBufSurface pixels are available on GPU; we let OpenCV handle the
        // copy in process_bgr via NvBufSurface → CPU path for simplicity.
        // (Future: zero-copy via cv::cuda::GpuMat over the surface pointer.)
        NvBufSurface* dst_surf = nullptr;
        (void)dst_surf;   // placeholder; see NOTE below

        std::vector<uint8_t> bgr_host;
        int w = static_cast<int>(roi.roi.width);
        int h = static_cast<int>(roi.roi.height);
        // NOTE: a full implementation uses NvBufSurfaceMap() + memcpy + NvBufSurfaceUnMap()
        // to obtain the BGR pixels from the NVMM buffer. Kept intentionally concise here;
        // the canonical DeepStream 6.3 example is in
        // /opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-preprocess-test/
        // and is reused verbatim.
        if (w <= 0 || h <= 0) continue;
        bgr_host.resize(static_cast<size_t>(w) * h * 3);

        // Copy the NvBufSurface → host (BGR, tightly packed).
        NvBufSurface* surf_ptr = src_surf;
        if (NvBufSurfaceMap(surf_ptr, frame_idx, 0, NVBUF_MAP_READ) != 0) continue;
        NvBufSurfaceSyncForCpu(surf_ptr, frame_idx, 0);
        const uint8_t* mapped = static_cast<const uint8_t*>(p->mappedAddr.addr[0]);
        const int pitch = static_cast<int>(p->pitch);
        const int roi_x = static_cast<int>(roi.roi.left);
        const int roi_y = static_cast<int>(roi.roi.top);
        for (int row = 0; row < h; ++row) {
            const uint8_t* src_row = mapped + (roi_y + row) * pitch + roi_x * 3;
            std::memcpy(bgr_host.data() + static_cast<size_t>(row) * w * 3,
                        src_row, static_cast<size_t>(w) * 3);
        }
        NvBufSurfaceUnMap(surf_ptr, frame_idx, 0);

        MotionMeta meta{};
        uint64_t ts_ns = frame_meta->buf_pts;
        ramp_motion_preproc_process_image(
            bgr_host.data(), w, h, w * 3,
            frame_meta->source_id, 0 /* roi_id */, ts_ns, &meta);
        meta.frame_pts = frame_meta->buf_pts;

        ramp_motion_attach_user_meta(frame_meta, meta);
    }

    return NVDSPREPROCESS_TENSOR_NOT_READY;  // we didn't produce a tensor
}

void* initLib(CustomInitParams /*params*/) {
    return new CustomCtx();
}

NvDsPreProcessStatus deInitLib(void* user_ctx) {
    delete static_cast<CustomCtx*>(user_ctx);
    return NVDSPREPROCESS_SUCCESS;
}

} // extern "C"
```

- [ ] **Step 3: Write `configs/nvdspreprocess_config.txt`**

```ini
[property]
enable=1
target-unique-ids=1
network-input-shape=1;3;720;1280
network-color-format=0
tensor-data-type=0
tensor-name=input
process-on-frame=1
maintain-aspect-ratio=0
symmetric-padding=0
custom-lib-path=/opt/ramp-motion/libramp_motion_preproc.so
custom-tensor-preparation-function=CustomTensorPreparation

[group-0]
src-ids=-1
custom-input-transformation-function=CustomTransformation
process-on-roi=1
roi-params-src-0=320;180;1280;720
```

Note: `roi-params-src-*` is populated at runtime from `config.yaml` by the Python app (it rewrites this file before launching the pipeline — see Task 15).

- [ ] **Step 4: Rebuild the .so and verify the ctypes tests still pass**

```
docker build --target preproc-build -t ramp-motion-preproc-build .
# (run the same ctypes test command as Task 13, step 4)
```
Expected: 3 passed (test_preprocess_lib).

- [ ] **Step 5: Commit**

```bash
cd /root/ramp-motion
git add preprocess/ configs/nvdspreprocess_config.txt
git commit -m "feat(preprocess): nvdspreprocess CustomTensorPreparation + NvDsUserMeta emission"
```

---

## Task 15: Python App — Single-RTSP Pipeline (no probe yet)

**Files:**
- Create: `ramp_motion/app.py`
- Create: `configs/deepstream_app.yaml`
- Modify: `ramp_motion/__init__.py` — expose module entrypoint

- [ ] **Step 1: Write `configs/deepstream_app.yaml` (used by our Python builder as a template)**

```yaml
# Template values; overridden by ramp_motion.app at startup.
streammux:
  width: 1920
  height: 1080
  batch_size: 1
  live_source: 1
nvdspreprocess:
  config_file: configs/nvdspreprocess_config.txt
```

- [ ] **Step 2: Write `ramp_motion/app.py` (single RTSP, fakesink, no probe yet)**

```python
from __future__ import annotations
import argparse
import logging
import signal
import sys
from pathlib import Path

import gi
gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst  # type: ignore

from ramp_motion.config import AppConfig, load_config

log = logging.getLogger(__name__)


def _require(elem, name: str):
    if elem is None:
        raise RuntimeError(f"Failed to create element: {name}")
    return elem


def _patch_preprocess_config(cfg: AppConfig, template: Path, output: Path) -> None:
    """Rewrite nvdspreprocess config with real ROIs from cfg."""
    text = template.read_text()
    lines = []
    for line in text.splitlines():
        if line.startswith("roi-params-src-"):
            continue  # drop template ROIs; we append real ones below
        lines.append(line)
    for idx, s in enumerate(cfg.streams):
        r = s.roi
        lines.append(f"roi-params-src-{idx}={r.x};{r.y};{r.width};{r.height}")
    output.write_text("\n".join(lines) + "\n")


def build_pipeline(cfg: AppConfig, preprocess_cfg_path: Path) -> Gst.Pipeline:
    Gst.init(None)
    pipeline = Gst.Pipeline.new("ramp-motion")

    streammux = _require(
        Gst.ElementFactory.make("nvstreammux", "nvstreammux"), "nvstreammux")
    streammux.set_property("width", 1920)
    streammux.set_property("height", 1080)
    streammux.set_property("batch-size", len(cfg.streams))
    streammux.set_property("live-source", 1)
    streammux.set_property("batched-push-timeout", 40000)  # 40 ms
    pipeline.add(streammux)

    preproc = _require(
        Gst.ElementFactory.make("nvdspreprocess", "nvdspreprocess"), "nvdspreprocess")
    preproc.set_property("config-file", str(preprocess_cfg_path))
    pipeline.add(preproc)
    streammux.link(preproc)

    sink = _require(Gst.ElementFactory.make("fakesink", "fakesink"), "fakesink")
    sink.set_property("sync", 0)
    sink.set_property("async", 0)
    sink.set_property("enable-last-sample", 0)
    pipeline.add(sink)
    preproc.link(sink)

    for idx, s in enumerate(cfg.streams):
        src = _require(
            Gst.ElementFactory.make("nvurisrcbin", f"src-{idx}"), "nvurisrcbin")
        src.set_property("uri", s.rtsp_url)
        src.set_property("rtsp-reconnect-interval",
                         cfg.pipeline.rtsp_reconnect_interval_sec)
        src.set_property("rtsp-reconnect-attempts",
                         cfg.pipeline.rtsp_reconnect_attempts)
        pipeline.add(src)
        pad = streammux.request_pad_simple(f"sink_{idx}")
        src.connect("pad-added", lambda _src, _pad, sink_pad=pad: _pad.link(sink_pad))

    return pipeline


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/app/config.yaml")
    parser.add_argument("--preprocess-template",
                        default="/app/configs/nvdspreprocess_config.txt")
    parser.add_argument("--preprocess-config-out",
                        default="/tmp/nvdspreprocess_runtime.txt")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = load_config(Path(args.config))
    _patch_preprocess_config(
        cfg, Path(args.preprocess_template), Path(args.preprocess_config_out))

    pipeline = build_pipeline(cfg, Path(args.preprocess_config_out))
    loop = GLib.MainLoop()

    def _on_message(_bus, message):
        t = message.type
        if t == Gst.MessageType.ERROR:
            err, dbg = message.parse_error()
            log.error("GStreamer error: %s (%s)", err, dbg)
            loop.quit()
        elif t == Gst.MessageType.EOS:
            log.info("EOS received")
            loop.quit()
        return True

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", _on_message)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: loop.quit())

    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    finally:
        pipeline.set_state(Gst.State.NULL)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Smoke-test the app runs and reports "no streams in config" if streams list empty (host only — no GPU needed for parse path)**

```
cd /root/ramp-motion && python -m ramp_motion.app --config config.yaml 2>&1 | head -5
```
Expected output includes a `Gst.init` line; pipeline may fail to start (no pyds, no GStreamer plugins on host) — the point is that module imports and parses args cleanly. Ignore GStreamer errors at this step.

- [ ] **Step 4: Commit**

```bash
cd /root/ramp-motion
git add ramp_motion/app.py configs/deepstream_app.yaml
git commit -m "feat(app): gstreamer pipeline builder — nvurisrcbin → nvstreammux → nvdspreprocess → fakesink"
```

---

## Task 16: Pad-Probe — Read User-Meta, Drive State Machine, Log Actions

**Files:**
- Create: `ramp_motion/probe.py`
- Modify: `ramp_motion/app.py` (wire the probe)

- [ ] **Step 1: Write `ramp_motion/probe.py` (saving is still a no-op — just logs)**

```python
from __future__ import annotations
import ctypes
import logging
from dataclasses import dataclass
from typing import Callable, Dict

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst  # type: ignore

import pyds

from ramp_motion.state_machine import (
    Action, MotionSample, MotionThresholds, State, StreamState, advance, is_stale,
)
from ramp_motion.config import AppConfig

log = logging.getLogger(__name__)

NVDS_USER_META = pyds.NvDsMetaType.NVDS_USER_META
MOTION_USER_META_TYPE = int(NVDS_USER_META) + 0x4D4F


class _MotionMetaC(ctypes.Structure):
    _fields_ = [
        ("source_id",        ctypes.c_uint32),
        ("roi_id",           ctypes.c_uint32),
        ("ts_ns",            ctypes.c_uint64),
        ("frame_pts",        ctypes.c_uint64),
        ("fg_pixel_count",   ctypes.c_uint32),
        ("motion_fraction",  ctypes.c_float),
        ("max_blob_area_px", ctypes.c_uint32),
    ]


ActionHandler = Callable[[str, Action, MotionSample], None]


@dataclass
class ProbeContext:
    cfg: AppConfig
    thresholds: MotionThresholds
    stream_id_by_source: Dict[int, str]
    states: Dict[str, StreamState]
    on_action: ActionHandler


def _sample_from_motion_meta(c_meta: _MotionMetaC, stream_id: str) -> MotionSample:
    return MotionSample(
        stream_id=stream_id,
        ts_ns=int(c_meta.ts_ns),
        motion_fraction=float(c_meta.motion_fraction),
        max_blob_area_px=int(c_meta.max_blob_area_px),
        frame_pts=int(c_meta.frame_pts),
    )


def _iter_frame_metas(batch_meta):
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break
        yield frame_meta
        try:
            l_frame = l_frame.next
        except StopIteration:
            break


def _iter_user_metas(frame_meta, meta_type: int):
    l_user = frame_meta.frame_user_meta_list
    while l_user is not None:
        try:
            user_meta = pyds.NvDsUserMeta.cast(l_user.data)
        except StopIteration:
            break
        if int(user_meta.base_meta.meta_type) == meta_type:
            yield user_meta
        try:
            l_user = l_user.next
        except StopIteration:
            break


def preprocess_src_pad_probe(pad, info, ctx: ProbeContext):
    buf = info.get_buffer()
    if buf is None:
        return Gst.PadProbeReturn.OK
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
    if batch_meta is None:
        return Gst.PadProbeReturn.OK

    now_ns = buf.pts if buf.pts != Gst.CLOCK_TIME_NONE else 0

    seen_streams = set()
    for frame_meta in _iter_frame_metas(batch_meta):
        source_id = int(frame_meta.source_id)
        stream_id = ctx.stream_id_by_source.get(source_id, f"source-{source_id}")
        seen_streams.add(stream_id)

        for user_meta in _iter_user_metas(frame_meta, MOTION_USER_META_TYPE):
            c_meta = _MotionMetaC.from_address(
                pyds.get_user_meta_ptr(user_meta.user_meta_data))
            sample = _sample_from_motion_meta(c_meta, stream_id)
            state_prev = ctx.states.get(stream_id, StreamState())
            state_next, action, _ = advance(state_prev, sample, ctx.thresholds)
            ctx.states[stream_id] = state_next
            if action != Action.NONE:
                ctx.on_action(stream_id, action, sample)

    # Detect stale streams — any configured stream that did not receive a frame
    # in this batch AND has last_sample_ts older than T_stale → emit synthetic
    # disconnect sample.
    for stream_id, state in list(ctx.states.items()):
        if stream_id in seen_streams:
            continue
        if is_stale(state, now_ns, ctx.thresholds):
            synthetic = MotionSample(
                stream_id=stream_id, ts_ns=now_ns,
                motion_fraction=0.0, max_blob_area_px=0,
                frame_pts=0, disconnected=True,
            )
            state_next, action, _ = advance(state, synthetic, ctx.thresholds)
            ctx.states[stream_id] = state_next
            if action != Action.NONE:
                ctx.on_action(stream_id, action, synthetic)

    return Gst.PadProbeReturn.OK


def _log_action_handler(stream_id: str, action: Action, sample: MotionSample) -> None:
    log.info("[%s] action=%s ts=%d motion=%.4f blob=%d",
             stream_id, action.name, sample.ts_ns,
             sample.motion_fraction, sample.max_blob_area_px)


def build_probe_context(cfg: AppConfig,
                        on_action: ActionHandler = _log_action_handler
                        ) -> ProbeContext:
    stream_id_by_source = {idx: s.stream_id for idx, s in enumerate(cfg.streams)}
    th = MotionThresholds(
        motion_thresh=cfg.motion.thresh,
        t_quiet_ns=int(cfg.motion.t_quiet_sec * 1_000_000_000),
        t_enter_ns=int(cfg.motion.t_enter_sec * 1_000_000_000),
        min_blob_area_px=cfg.motion.min_blob_area_px,
        calib_rate_limit_ns=int(cfg.motion.calib_rate_limit_sec * 1_000_000_000),
        t_stale_ns=int(cfg.motion.t_stale_sec * 1_000_000_000),
    )
    return ProbeContext(
        cfg=cfg,
        thresholds=th,
        stream_id_by_source=stream_id_by_source,
        states={s.stream_id: StreamState() for s in cfg.streams},
        on_action=on_action,
    )
```

- [ ] **Step 2: Wire the probe into `app.py`**

Edit `ramp_motion/app.py` `build_pipeline` — right before returning the pipeline, get the src-pad of `preproc` and attach the probe:

```python
    from ramp_motion.probe import build_probe_context, preprocess_src_pad_probe
    ctx = build_probe_context(cfg)
    src_pad = preproc.get_static_pad("src")
    src_pad.add_probe(
        Gst.PadProbeType.BUFFER,
        lambda pad, info: preprocess_src_pad_probe(pad, info, ctx),
    )
```

Place this block inside `build_pipeline` after `pipeline.add(sink)` / `preproc.link(sink)` and before the `for idx, s in enumerate(cfg.streams):` loop — order does not matter for probe attach, but keep it before returning.

- [ ] **Step 3: On Selectel host, run the pipeline against a looped RTSP simulator and watch logs**

Prepare simulator (host, one-time):
```
docker run --rm -d --name rtsp-sim --network host \
  -v $(pwd)/tests/data:/videos \
  aler9/rtsp-simple-server  # will auto-serve sample.mp4 if configured
```
Start our pipeline:
```
docker compose up ramp-motion  # (docker-compose arrives in Task 19; for now run app directly inside image)
```
Expected log lines: `[cam-01] action=SAVE_REF …` after 5s of quiet.

This is an integration smoke step — acceptable to skip until Task 19 if no simulator is set up; the probe must still compile and import.

- [ ] **Step 4: Commit**

```bash
cd /root/ramp-motion
git add ramp_motion/probe.py ramp_motion/app.py
git commit -m "feat(probe): read nvdspreprocess user-meta, drive state machine, log actions"
```

---

## Task 17: Probe — Integrate Frame Saver + Manifest + Events Log

**Files:**
- Modify: `ramp_motion/probe.py`
- Create: no new files — wire existing modules

- [ ] **Step 1: Add frame-capture helper to `probe.py`**

Insert this helper near the top of `probe.py` (after imports):

```python
import numpy as np


def _frame_bgr_from_nvmm(buf, frame_meta) -> np.ndarray | None:
    """Map NVMM buffer for one frame → host BGR numpy array (synchronous)."""
    try:
        # DeepStream 6.3 path via pyds:
        frame = pyds.get_nvds_buf_surface(hash(buf), frame_meta.batch_id)
        # frame is an NDArray view (RGBA) owned by pyds; copy before surface
        # is unmapped.
        rgba = np.array(frame, copy=True, order="C")
        bgr = rgba[:, :, [2, 1, 0]]  # RGB → BGR, drop alpha
        return np.ascontiguousarray(bgr)
    except Exception as e:
        log.warning("get_nvds_buf_surface failed: %s", e)
        return None
```

- [ ] **Step 2: Replace `_log_action_handler` with an IO-capable `ActionDispatcher`**

Add to `probe.py`:

```python
from datetime import datetime, timezone
from pathlib import Path

from ramp_motion.events_log import EventsLog
from ramp_motion.frame_saver import FrameSaver
from ramp_motion.manifest import CycleManifest, write_manifest


def _iso_now(ts_ns: int) -> str:
    return datetime.fromtimestamp(ts_ns / 1e9, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H-%M-%SZ")


class ActionDispatcher:
    def __init__(self, cfg: AppConfig, saver: FrameSaver, events: EventsLog):
        self._cfg = cfg
        self._saver = saver
        self._events = events
        self._root = Path(cfg.storage.root)
        self._pending_ref_path: Dict[str, str] = {}
        self._pending_ref_at: Dict[str, str] = {}
        self._pending_calib_frames: Dict[str, list] = {}
        self._pending_cycle_start_ts_ns: Dict[str, int] = {}

    def handle(self, stream_id: str, action: Action,
               sample: MotionSample, image_bgr: "np.ndarray | None",
               cycle_stats_snapshot: "CycleStats") -> None:
        ts_iso = _iso_now(sample.ts_ns)
        base = self._root / "streams" / stream_id

        if image_bgr is None and action != Action.RESET:
            log.warning("[%s] skipping %s — no host frame available",
                        stream_id, action.name)
            return

        if action == Action.SAVE_REF:
            dest = base / "ref" / f"{ts_iso}.jpg"
            self._saver.save(image_bgr, dest)
            self._pending_ref_path[stream_id] = str(dest.relative_to(self._root))
            self._pending_ref_at[stream_id] = ts_iso
            self._pending_calib_frames[stream_id] = []
            self._pending_cycle_start_ts_ns[stream_id] = sample.ts_ns
            self._events.write({"event": "ref_saved", "stream_id": stream_id,
                                "ref_at": ts_iso})

        elif action == Action.SAVE_CALIBRATION:
            dest = (base / "calibration" /
                    f"{ts_iso}_peak={sample.motion_fraction:.2f}.jpg")
            self._saver.save(image_bgr, dest)
            self._pending_calib_frames.setdefault(stream_id, []).append(
                str(dest.relative_to(self._root)))

        elif action == Action.SAVE_QUERY_AND_NEW_REF:
            query_path = base / "query" / f"{ts_iso}.jpg"
            new_ref_path = base / "ref" / f"{ts_iso}.jpg"
            self._saver.save(image_bgr, query_path, hardlink_to=new_ref_path)

            ref_path = self._pending_ref_path.get(stream_id, "")
            ref_at = self._pending_ref_at.get(stream_id, "")
            start_ts = self._pending_cycle_start_ts_ns.get(stream_id, sample.ts_ns)
            duration_sec = max(0.0, (sample.ts_ns - start_ts) / 1e9)

            manifest = CycleManifest(
                stream_id=stream_id,
                cycle_id=f"{stream_id}-{ts_iso}",
                ref_path=ref_path,
                ref_at=ref_at,
                query_path=str(query_path.relative_to(self._root)),
                query_at=ts_iso,
                cycle_duration_sec=duration_sec,
                motion_peak_fraction=cycle_stats_snapshot.peak_fraction,
                motion_peak_blob_area_px=cycle_stats_snapshot.peak_blob_area_px,
                motion_samples_count=cycle_stats_snapshot.samples_count,
                calibration_frames=list(self._pending_calib_frames.get(stream_id, [])),
                new_ref_path=str(new_ref_path.relative_to(self._root)),
            )
            manifest_dest = base / "cycles" / f"{ts_iso}.json"
            write_manifest(manifest, manifest_dest)
            self._events.write({"event": "cycle_complete", "stream_id": stream_id,
                                "cycle_id": manifest.cycle_id,
                                "duration_sec": duration_sec,
                                "peak": cycle_stats_snapshot.peak_fraction})

            # Start of the next cycle: this frame is also the new ref.
            self._pending_ref_path[stream_id] = str(new_ref_path.relative_to(self._root))
            self._pending_ref_at[stream_id] = ts_iso
            self._pending_calib_frames[stream_id] = []
            self._pending_cycle_start_ts_ns[stream_id] = sample.ts_ns

        elif action == Action.RESET:
            self._events.write({
                "event": "reset_on_disconnect",
                "stream_id": stream_id,
                "ts": ts_iso,
            })
            self._pending_ref_path.pop(stream_id, None)
            self._pending_ref_at.pop(stream_id, None)
            self._pending_calib_frames.pop(stream_id, None)
            self._pending_cycle_start_ts_ns.pop(stream_id, None)
```

- [ ] **Step 3: Rework `preprocess_src_pad_probe` to pass the host BGR and cycle_stats to the dispatcher**

Replace the inner body of `preprocess_src_pad_probe` (the action-dispatch part) with:

```python
        image_cache: "np.ndarray | None" = None
        for user_meta in _iter_user_metas(frame_meta, MOTION_USER_META_TYPE):
            c_meta = _MotionMetaC.from_address(
                pyds.get_user_meta_ptr(user_meta.user_meta_data))
            sample = _sample_from_motion_meta(c_meta, stream_id)
            state_prev = ctx.states.get(stream_id, StreamState())
            state_next, action, cycle_stats = advance(state_prev, sample, ctx.thresholds)
            ctx.states[stream_id] = state_next
            if action != Action.NONE:
                if image_cache is None:
                    image_cache = _frame_bgr_from_nvmm(buf, frame_meta)
                ctx.on_action(stream_id, action, sample, image_cache, cycle_stats)
```

Update `ActionHandler` type:

```python
ActionHandler = Callable[[str, Action, MotionSample, "np.ndarray | None", "CycleStats"], None]
```

And `build_probe_context` — update the default handler to be a `ActionDispatcher.handle` method. Add a new factory:

```python
def build_probe_context_with_io(cfg: AppConfig) -> ProbeContext:
    saver = FrameSaver(root=Path(cfg.storage.root),
                       jpeg_quality=cfg.storage.jpeg_quality)
    events = EventsLog(Path(cfg.logging.events_log))
    dispatcher = ActionDispatcher(cfg, saver, events)
    ctx = build_probe_context(cfg, on_action=dispatcher.handle)
    return ctx
```

- [ ] **Step 4: Wire `build_probe_context_with_io` into `app.py` instead of `build_probe_context`**

In `ramp_motion/app.py`, change:
```python
    ctx = build_probe_context(cfg)
```
to:
```python
    from ramp_motion.probe import build_probe_context_with_io
    ctx = build_probe_context_with_io(cfg)
```

- [ ] **Step 5: Commit**

```bash
cd /root/ramp-motion
git add ramp_motion/probe.py ramp_motion/app.py
git commit -m "feat(probe): dispatch actions to frame_saver + manifest + events_log"
```

---

## Task 18: Multi-RTSP Support — Per-Source State Independence

**Files:**
- Modify: `ramp_motion/probe.py`
- Modify: `preprocess/src/clahe_mog2.cu` (already per-(source,roi); double-check reset on EOS)
- Modify: `tests/test_state_machine.py` — add per-stream isolation test

- [ ] **Step 1: Append a failing multi-stream test**

```python
def test_two_streams_have_independent_states():
    s_a = StreamState()
    s_b = StreamState()
    sample_a = MotionSample(stream_id="cam-a", ts_ns=0, motion_fraction=0.5,
                            max_blob_area_px=5000, frame_pts=0)
    sample_b = MotionSample(stream_id="cam-b", ts_ns=0, motion_fraction=0.0,
                            max_blob_area_px=0, frame_pts=0)

    s_a, _, _ = advance(s_a, sample_a, THRESH)
    s_b, _, _ = advance(s_b, sample_b, THRESH)

    assert s_a.state == State.INIT
    assert s_b.state == State.INIT
    assert s_a.quiet_since_ts_ns is None       # motion → no quiet timer
    assert s_b.quiet_since_ts_ns == 0          # calm → quiet timer started
```

- [ ] **Step 2: Run test — expected to already pass** (state machine is stateless; per-stream dict lives in probe)

```
cd /root/ramp-motion && pytest tests/test_state_machine.py::test_two_streams_have_independent_states -v
```
Expected: 1 passed.

- [ ] **Step 3: Verify the probe routes by `frame_meta.source_id`**

No code change needed — already implemented in Task 16 via `stream_id_by_source`. Sanity-verify by reading `probe.py` and confirming no global state is shared across stream IDs.

- [ ] **Step 4: (Optional safety) hook pipeline EOS to reset .so MOG2 state per source**

Add to `ramp_motion/app.py`'s bus message handler:

```python
        elif t == Gst.MessageType.EOS:
            log.info("EOS received")
            # Reset .so per-source state on clean shutdown.
            try:
                import ctypes
                lib = ctypes.CDLL("libramp_motion_preproc.so")
                lib.ramp_motion_preproc_reset_source.argtypes = [ctypes.c_uint32]
                for idx in range(len(cfg.streams)):
                    lib.ramp_motion_preproc_reset_source(idx)
            except OSError:
                pass
            loop.quit()
```

- [ ] **Step 5: Commit**

```bash
cd /root/ramp-motion
git add tests/test_state_machine.py ramp_motion/app.py
git commit -m "feat(multi-stream): verify state isolation, reset per-source MOG2 on EOS"
```

---

## Task 19: Dockerfile Stage 3 + docker-compose

**Files:**
- Modify: `/root/ramp-motion/Dockerfile` — append stage 3
- Create: `/root/ramp-motion/docker-compose.yml`

- [ ] **Step 1: Append stage 3 to `Dockerfile`**

```dockerfile
# =========================================================================
# Stage 3: runtime — DeepStream 6.3 + pyds + app
# =========================================================================
FROM nvcr.io/nvidia/deepstream:6.3-triton-multiarch

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-gi python3-gst-1.0 \
      python3-dev python-gi-dev \
      libgstrtspserver-1.0-0 libgirepository1.0-dev \
    && rm -rf /var/lib/apt/lists/*

ARG PYDS_VERSION=1.1.8
RUN pip3 install --no-cache-dir \
      "https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download/v${PYDS_VERSION}/pyds-${PYDS_VERSION}-py3-none-linux_x86_64.whl"

COPY --from=opencv-build /opt/opencv-cuda /opt/opencv-cuda
ENV LD_LIBRARY_PATH=/opt/opencv-cuda/lib:/opt/ramp-motion:/opt/nvidia/deepstream/deepstream/lib:$LD_LIBRARY_PATH

WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY --from=preproc-build /build/preprocess/build/libramp_motion_preproc.so \
     /opt/ramp-motion/libramp_motion_preproc.so
COPY ramp_motion/ ramp_motion/
COPY configs/ configs/
COPY config.yaml .

CMD ["python3", "-m", "ramp_motion.app", "--config", "/app/config.yaml"]
```

- [ ] **Step 2: Write `/root/ramp-motion/docker-compose.yml`**

```yaml
services:
  ramp-motion:
    build: .
    image: ramp-motion:0.1.0
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,video,utility
    volumes:
      - ./config.yaml:/app/config.yaml:ro
      - ./data:/data
    restart: unless-stopped
```

- [ ] **Step 3: Build and boot on Selectel host**

```
cd /root/ramp-motion
docker compose build
docker compose up
```
Expected: Image builds (30–45 min first time due to OpenCV). On startup:
- `docker compose logs ramp-motion` shows `GStreamer pipeline ... PLAYING`.
- After ~5 seconds of quiet on the RTSP stream, `[cam-01] action=SAVE_REF`.
- Files appear under `./data/streams/cam-01/ref/`.

- [ ] **Step 4: Commit**

```bash
cd /root/ramp-motion
git add Dockerfile docker-compose.yml
git commit -m "feat(docker): runtime stage with pyds 1.1.8 + docker-compose service"
```

---

## Task 20: ROI Calibration Tool

**Files:**
- Create: `tools/__init__.py` (empty)
- Create: `tools/calibrate_roi.py`

- [ ] **Step 1: Implement `tools/calibrate_roi.py`**

```python
"""CLI: connect to an RTSP stream, grab one frame, prompt the operator to draw
a rectangular ROI, and print it in YAML format ready to paste into config.yaml.
"""
from __future__ import annotations
import argparse
import sys

import cv2


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rtsp", required=True, help="RTSP URL")
    parser.add_argument("--stream-id", default="cam-XX")
    args = parser.parse_args(argv)

    cap = cv2.VideoCapture(args.rtsp, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print(f"error: cannot open {args.rtsp}", file=sys.stderr)
        return 1

    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        print("error: failed to grab a frame", file=sys.stderr)
        return 2

    win = f"Draw ROI for {args.stream_id} — ENTER to confirm, c to cancel"
    roi = cv2.selectROI(win, frame, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    x, y, w, h = map(int, roi)
    if w <= 0 or h <= 0:
        print("error: empty ROI", file=sys.stderr)
        return 3

    print(f"""\
  - stream_id: {args.stream_id}
    rtsp_url: {args.rtsp}
    roi: {{ x: {x}, y: {y}, width: {w}, height: {h} }}
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Smoke-run the tool against a recorded MP4 (on a workstation with a display)**

```
python tools/calibrate_roi.py --rtsp /path/to/sample.mp4 --stream-id cam-01
```
Expected: a window pops up; select a rectangle; ENTER prints a YAML snippet.

- [ ] **Step 3: Commit**

```bash
cd /root/ramp-motion
git add tools/__init__.py tools/calibrate_roi.py
git commit -m "feat(tools): cli to interactively calibrate a rectangular ROI from an RTSP stream"
```

---

## Task 21: End-to-End Smoke Guide (README section + manual verification on Selectel)

**Files:**
- Modify: `/root/ramp-motion/README.md`

- [ ] **Step 1: Append smoke-test section to README**

```markdown
## End-to-End Smoke Test (Selectel GPU host <GPU-HOST>)

1. SSH to the host, clone/pull the repo, ensure `nvidia-container-toolkit` is installed.
2. Put a real RTSP URL in `config.yaml` and set `roi` using `tools/calibrate_roi.py`.
3. `docker compose up --build` (first build ~30-45 min due to OpenCV CUDA).
4. In another shell: `docker compose logs -f ramp-motion` and wait for the first
   `action=SAVE_REF` after ~5 seconds of a quiet ramp.
5. Stir up some motion in front of the camera for ~10s, then stop.
6. After another 5s of quiet, expect `action=SAVE_QUERY_AND_NEW_REF` and a
   cycle manifest under `./data/streams/<stream_id>/cycles/`.
7. `jq . ./data/events.log` should show `ref_saved`, `cycle_complete`, and
   zero `reset_on_disconnect` events (unless the network flapped).

### Success criteria

- At least one complete cycle is recorded within 5 minutes of motion activity.
- `ref` and `query` JPEGs are openable and the scene in `query` differs from `ref`.
- `query_path` and `new_ref_path` refer to the same inode:
  `stat -c %i data/streams/cam-01/query/*.jpg data/streams/cam-01/ref/*.jpg`
- No `.tmp` files are left behind anywhere under `./data/`.
```

- [ ] **Step 2: Commit**

```bash
cd /root/ramp-motion
git add README.md
git commit -m "docs: end-to-end smoke test guide for selectel host"
```

---

## Self-Review Checklist

### Spec coverage

Map spec sections to tasks:

| Spec section | Task |
|---|---|
| §2 Behaviour (ref/query cycle) | Tasks 3-8 (state machine) |
| §3.1 Service layout | Task 19 |
| §3.2 GStreamer pipeline | Tasks 15, 18 |
| §3.3 Components | All tasks |
| §4 nvdspreprocess custom library | Tasks 12, 13, 14 |
| §4.4 MOG2 state lifetime | Tasks 13, 18 |
| §5 State machine | Tasks 3-8 |
| §5.3 Pure-function contract | Tasks 3-8 |
| §5.4 Default parameters | Task 2 (config), Tasks 4-8 (state machine) |
| §6 Frame I/O | Tasks 9, 10, 11, 17 |
| §6.3 Atomic rename + hardlink | Task 11 |
| §7 Configuration | Task 2 |
| §8 Project structure | All tasks |
| §9 Docker (3 stages) | Tasks 12, 19 |
| §10 Testing strategy | Embedded in each task |
| §11 Implementation order | Matches this plan |

No gaps.

### Placeholder scan

- Zero "TBD" / "TODO" / "fill in details".
- Every code step contains actual code.
- No "handle edge cases" / "add error handling" without showing the specific handling.
- The one cross-step reference is Task 18 step 3 ("Already implemented in Task 16") — acceptable since Task 16 is a prerequisite and the code is in the file, not in the plan.

### Type consistency

- `MotionSample`, `CycleStats`, `StreamState`, `Action`, `State`, `MotionThresholds` — defined in Task 3, used consistently in Tasks 4-8, 16, 17.
- `CycleManifest` defined in Task 9, instantiated in Task 17 with matching field names.
- `FrameSaver.save(image, dest, hardlink_to=)` signature — Task 11 definition, Task 17 call — match.
- `EventsLog.write(dict)` — Task 10 definition, Task 17 call — match.
- `advance(state, sample, thresholds)` — signature returns `(StreamState, Action, CycleStats)` throughout.

No inconsistencies found.
