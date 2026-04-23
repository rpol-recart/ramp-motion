# Ramp Motion Pipeline — Design Specification

**Date:** 2026-04-23
**Status:** Draft (hypothesis test)
**Stack:** DeepStream 6.3 (+ python bindings `pyds`), `nvdspreprocess` custom library (C++/CUDA, OpenCV with CUDA), Python 3.8, Docker (NVIDIA runtime)
**Target host:** Selectel GPU server (<GPU-HOST>)

## 1. Goal

Test the hypothesis that **motion-gated capture of reference/query frame pairs** is sufficient to detect "something changed on the ramp" — without running object detection in the pipeline. A downstream consumer (VLM, embedding model, or human operator) compares `ref` → `query` pairs to infer whether a container was placed or removed.

This is a **separate, minimal-dependency project** (`/root/ramp-motion`), independent from the existing YOLO-based pipeline in `/root/ramp`.

## 2. High-level behaviour

For each RTSP stream:

1. Continuously compute `motion_fraction` = share of ROI pixels classified as foreground by MOG2 (after CLAHE).
2. When `motion_fraction < thresh` continuously for ≥ **5 seconds**, save the current frame as the **reference** (`ref`). State becomes `WAITING_MOTION`.
3. When `motion_fraction > thresh` for a short debounce window, enter `IN_MOTION`. Track peak `motion_fraction` and `max_blob_area_px` for the cycle. Periodically (rate-limited) save **calibration frames** at local peaks for operator review.
4. When `motion_fraction < thresh` continuously for ≥ **5 seconds** again, save the current frame as the **query** (paired with the previous `ref`) and simultaneously as the **new reference** for the next cycle. Emit a `cycle_complete` event with peak motion statistics.

Failure modes:

- **Stream disconnect** — `nvurisrcbin` auto-reconnects. If user-meta for a stream is silent for ≥ `T_stale` (10 s), the state machine resets to `INIT`; baseline is re-established on reconnect (old `ref` is invalidated because lighting/scene may have shifted).
- **Noise blobs** — blobs smaller than `min_blob_area_px` are ignored (shadow flicker, birds).

## 3. Architecture

### 3.1 Service layout

```
docker-compose (NVIDIA runtime, single GPU)
├── ramp-motion           # main service: GStreamer + pad-probe + frame saver
└── (ramp-motion-health)  # optional, skipped in v1
```

Single-container deployment. Pipeline, state machine, and disk I/O share one Python process. No Redis, no SQLite — local JSON manifests and JPEG files only.

### 3.2 GStreamer pipeline

```
nvurisrcbin × N ──► nvstreammux (batch=N, width=1920, height=1080, live-source=1)
        ──► videorate (target: 5 fps per source)
        ──► nvdspreprocess (custom_lib_path=libramp_motion_preproc.so,
                           ROIs per source from config)
        ──► pad probe (Python): state machines + frame saver
        ──► fakesink sync=0
```

### 3.3 Components

| Component | Language | Responsibility |
|---|---|---|
| `libramp_motion_preproc.so` | C++ / CUDA | Per-frame per-ROI CLAHE → MOG2 → attach motion metrics as `NvDsUserMeta`. No state, no I/O. |
| `ramp_motion/app.py` | Python | Build pipeline, attach probe, manage lifecycle, handle SIGTERM. |
| `ramp_motion/probe.py` | Python | Read `NvDsUserMeta`, call state machine, trigger frame saver. |
| `ramp_motion/state_machine.py` | Python (pure) | `advance(state, sample) → (state, action, cycle_stats)`. No I/O, no timers. Fully unit-testable. |
| `ramp_motion/frame_saver.py` | Python | `ThreadPoolExecutor`; NVMM→host copy in probe, JPEG encode + atomic write in worker. |
| `ramp_motion/manifest.py` | Python | Build cycle JSON, atomic write (`*.json.tmp` → `rename`). |
| `ramp_motion/config.py` | Python | Pydantic load of `config.yaml`. |

## 4. `nvdspreprocess` custom library

### 4.1 Entry points (DeepStream 6.3 SDK contract)

- `CustomTensorPreparation(CustomCtx*, NvDsPreProcessBatch*, NvDsPreProcessCustomBuf*&, CustomTensorParams&, NvDsPreProcessAcquirer*)` — called once per batch. We ignore the acquired tensor buffer (no downstream inference); we use this hook only to attach user-meta.
- `CustomTransformation(NvBufSurface*, NvBufSurface*, CustomTransformParams&)` — per-ROI pre-transform; we use it to extract the ROI crop.
- `initLib(CustomInitParams)` / `deInitLib(CustomCtx*)` — allocate/free per-source MOG2 state and CLAHE operator. (DS 6.3 signature, see `/opt/nvidia/deepstream/deepstream/sources/gst-plugins/gst-nvdspreprocess/nvdspreprocess_lib/nvdspreprocess_lib.cpp`.)

### 4.2 Processing per ROI per frame

```
roi_crop  = NvBufSurfaceFromBatchedROI(...)   // zero-copy NVMM crop (cv::cuda::GpuMat view)
clahe_out = cv::cuda::CLAHE(clip_limit=2.0, tile=(8,8)).apply(roi_crop_gray)
fg_mask   = cv::cuda::BackgroundSubtractorMOG2(
                history=500, varThreshold=16.0, detectShadows=true
            ).apply(clahe_out)

fg_pixels       = cv::cuda::countNonZero(fg_mask)
motion_fraction = fg_pixels / (roi.width * roi.height)
(labels, stats) = cv::cuda::connectedComponentsWithStats(fg_mask)    // → CPU stats
max_blob_area   = max(stats[:, CC_STAT_AREA])                        // excluding background label 0
```

### 4.3 Output: `NvDsUserMeta`

One `MotionMeta` per source per batch, attached to each source's `NvDsFrameMeta.frame_user_meta_list` (stream-level, not per-object):

```cpp
struct MotionMeta {
    uint32_t     source_id;
    uint32_t     roi_id;           // always 0 in v1 (one ROI per stream)
    uint64_t     ts_ns;            // monotonic ns from gst_clock_get_time
    uint64_t     frame_pts;
    uint32_t     fg_pixel_count;
    float        motion_fraction;
    uint32_t     max_blob_area_px;
};
```

Registered with a project-specific `user_meta_type` (e.g., `NVDS_USER_META_BASE + 0x4D4F`), freed by a `NvDsUserMetaReleaseFunc` registered via `nvds_add_user_meta_to_frame` — the probe reads but never frees the meta.

### 4.4 MOG2 state lifetime

One `cv::cuda::BackgroundSubtractorMOG2` per (source_id, roi_id) pair. Reset on explicit `ResetSourceState(source_id)` call, which the probe invokes when it detects disconnect-driven state-machine reset (over a C ABI shim). Also reset on pipeline EOS.

## 5. State machine

### 5.1 States

```
INIT                   — no ref yet, establishing baseline
WAITING_MOTION         — ref captured, awaiting motion start
IN_MOTION              — motion active, tracking peak
```

### 5.2 Transitions

```
INIT --[motion_fraction < thresh for T_quiet seconds]-->
    ACTION: SAVE_REF(current_frame)
    → WAITING_MOTION

WAITING_MOTION --[motion_fraction > thresh for T_enter seconds
                  AND max_blob_area_px >= min_blob_area_px]-->
    → IN_MOTION (reset cycle_stats = {peak: 0, calib_frames: []})

IN_MOTION --[motion_fraction > cycle_stats.peak_fraction
             AND time-since-last-calib >= calib_rate_limit_sec]-->
    ACTION: SAVE_CALIBRATION(current_frame, current_motion_fraction)
            cycle_stats.peak_fraction = motion_fraction
            cycle_stats.last_calib_ts = ts_ns
    → IN_MOTION

IN_MOTION --[motion_fraction > cycle_stats.peak_fraction
             AND rate-limit NOT satisfied]-->
    ACTION: update cycle_stats.peak_fraction only (no save)
    → IN_MOTION

IN_MOTION --[motion_fraction < thresh for T_quiet seconds]-->
    ACTION: SAVE_QUERY_AND_NEW_REF(current_frame)
            EMIT cycle_complete event
    → WAITING_MOTION

(any) --[no samples for stream_id for T_stale seconds]-->
    ACTION: RESET_ON_DISCONNECT
    → INIT
```

### 5.3 Pure-function contract

```python
@dataclass(frozen=True)
class MotionSample:
    stream_id: str
    ts_ns: int
    motion_fraction: float
    max_blob_area_px: int
    frame_pts: int
    disconnected: bool = False  # set by probe when T_stale tripped

class Action(Enum):
    NONE = 0
    SAVE_REF = 1
    SAVE_CALIBRATION = 2
    SAVE_QUERY_AND_NEW_REF = 3
    RESET = 4

def advance(state: StreamState, sample: MotionSample) -> tuple[StreamState, Action, CycleStats]:
    ...
```

`advance` is pure. All unit tests drive it with synthetic `MotionSample` sequences — no DeepStream, no GPU, no sleep.

### 5.4 Default parameters

| Parameter | Default | Tunable |
|---|---|---|
| `motion_thresh` (motion_fraction) | 0.02 | yes, per-stream |
| `T_quiet_sec` | 5.0 | yes |
| `T_enter_sec` (debounce) | 0.5 | yes |
| `min_blob_area_px` | 500 | yes |
| `calib_rate_limit_sec` | 1.0 | yes |
| `T_stale_sec` (disconnect detect) | 10.0 | yes |
| `pipeline_target_fps` | 5 | yes |

## 6. Frame I/O

### 6.1 Disk layout (mounted volume)

```
/data/
├── streams/
│   └── <stream_id>/
│       ├── ref/         <ISO8601>.jpg
│       ├── query/       <ISO8601>.jpg   (hard-linked to the corresponding new ref)
│       ├── calibration/ <ISO8601>_peak=<value>.jpg
│       └── cycles/      <ISO8601>.json  (cycle manifest)
└── events.log           JSONL: disconnects, reconnects, cycle_complete, resets
```

### 6.2 Cycle manifest schema

```json
{
  "stream_id": "cam-01",
  "cycle_id": "cam-01-2026-04-23T10-22-41Z",
  "ref_path": "streams/cam-01/ref/2026-04-23T10-15-03Z.jpg",
  "ref_at": "2026-04-23T10:15:03Z",
  "query_path": "streams/cam-01/query/2026-04-23T10-22-41Z.jpg",
  "query_at": "2026-04-23T10:22:41Z",
  "cycle_duration_sec": 458.3,
  "motion_peak_fraction": 0.34,
  "motion_peak_blob_area_px": 18420,
  "motion_samples_count": 1234,
  "calibration_frames": [
    "streams/cam-01/calibration/2026-04-23T10-19-12Z_peak=0.34.jpg"
  ],
  "new_ref_path": "streams/cam-01/ref/2026-04-23T10-22-41Z.jpg"
}
```

`query_path` and `new_ref_path` refer to the same physical JPEG (written once, hard-linked).

### 6.3 Save path (critical for correctness)

1. In probe: `pyds.get_nvds_buf_surface(buf, source_id)` → synchronous NVMM→host copy → numpy `uint8` HxWxC array (~2–5 ms). This happens inline so the NVMM surface is still valid.
2. Submit `(numpy_array, dest_path)` to `ThreadPoolExecutor(max_workers=4)`.
3. Worker: `cv2.imencode('.jpg', img, [IMWRITE_JPEG_QUALITY, 85])` → `Path(dest+'.tmp').write_bytes(data)` → `os.rename(dest+'.tmp', dest)`.
4. For `SAVE_QUERY_AND_NEW_REF`: write query JPEG first (atomically), hard-link to `new_ref` path, then write manifest JSON atomically. Manifest writes last so a partial cycle is always detectable (query exists without manifest → retry or discard).

### 6.4 Memory budget

- 1 probe copy per frame per stream saved = 1920×1080×3 bytes ≈ 6 MB copy buffer per worker slot. 4 workers × 6 MB = 24 MB peak. Negligible.
- Executor queue bounded at 16 items; overflow → log warning + drop calibration frames (never drop ref/query).

## 7. Configuration (`config.yaml`)

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
  rtsp_reconnect_attempts: -1   # infinite
  batch_size_auto: true          # == len(streams)

storage:
  root: /data
  jpeg_quality: 85

logging:
  level: INFO
  events_log: /data/events.log
```

## 8. Project structure

```
ramp-motion/
├── config.yaml
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── requirements.txt
├── README.md
├── docs/
│   └── superpowers/
│       └── specs/2026-04-23-ramp-motion-pipeline-design.md
├── preprocess/
│   ├── CMakeLists.txt
│   ├── include/ramp_motion_preproc.h
│   └── src/
│       ├── custom_lib.cpp
│       ├── clahe_mog2.cu
│       └── motion_meta.cpp
├── configs/
│   ├── deepstream_app.yaml
│   └── nvdspreprocess_config.txt
├── ramp_motion/
│   ├── __init__.py
│   ├── app.py
│   ├── probe.py
│   ├── state_machine.py
│   ├── frame_saver.py
│   ├── manifest.py
│   ├── health.py
│   └── config.py
├── tests/
│   ├── test_state_machine.py
│   ├── test_manifest.py
│   ├── test_frame_saver.py
│   └── test_preprocess_lib.py
└── tools/
    ├── calibrate_roi.py
    └── replay_events.py
```

## 9. Docker

### 9.1 Dockerfile (multi-stage, DeepStream 6.3)

The base image `nvcr.io/nvidia/deepstream:6.3-triton-multiarch` includes DeepStream SDK, TensorRT, CUDA, GStreamer, and a stock OpenCV — but **not** CUDA-enabled OpenCV modules (`cv::cuda::CLAHE`, `cv::cuda::BackgroundSubtractorMOG2`) and **not** the `pyds` Python bindings. Both are installed below.

```dockerfile
# =========================================================================
# Stage 1: build CUDA-enabled OpenCV (needed for cv::cuda::CLAHE and MOG2)
# =========================================================================
FROM nvcr.io/nvidia/deepstream:6.3-triton-multiarch AS opencv-build

ARG OPENCV_VERSION=4.8.0
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
    -D CUDA_ARCH_BIN=7.5,8.0,8.6,8.9 \
    -D BUILD_opencv_python2=OFF \
    -D BUILD_opencv_python3=OFF \
    -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_EXAMPLES=OFF \
    -D WITH_GSTREAMER=ON \
 && make -j"$(nproc)" && make install

# =========================================================================
# Stage 2: build the custom .so (links against CUDA OpenCV + DeepStream SDK)
# =========================================================================
FROM nvcr.io/nvidia/deepstream:6.3-triton-multiarch AS preproc-build
COPY --from=opencv-build /opt/opencv-cuda /opt/opencv-cuda
RUN apt-get update && apt-get install -y --no-install-recommends \
      cmake build-essential \
    && rm -rf /var/lib/apt/lists/*
ENV CMAKE_PREFIX_PATH=/opt/opencv-cuda
WORKDIR /build
COPY preprocess/ preprocess/
RUN cmake -S preprocess -B preprocess/build -DOpenCV_DIR=/opt/opencv-cuda/lib/cmake/opencv4 \
 && cmake --build preprocess/build -j"$(nproc)"

# =========================================================================
# Stage 3: runtime — DeepStream 6.3 + pyds + app
# =========================================================================
FROM nvcr.io/nvidia/deepstream:6.3-triton-multiarch

# OS packages + python runtime for gi/gst bindings
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-gi python3-gst-1.0 \
      python3-dev python-gi-dev \
      libgstrtspserver-1.0-0 libgirepository1.0-dev \
    && rm -rf /var/lib/apt/lists/*

# pyds wheel for DeepStream 6.3 (from NVIDIA-AI-IOT/deepstream_python_apps)
ARG PYDS_VERSION=1.1.8
RUN pip3 install --no-cache-dir \
      "https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download/v${PYDS_VERSION}/pyds-${PYDS_VERSION}-py3-none-linux_x86_64.whl"

# CUDA OpenCV runtime libs for the custom .so
COPY --from=opencv-build /opt/opencv-cuda /opt/opencv-cuda
ENV LD_LIBRARY_PATH=/opt/opencv-cuda/lib:/opt/ramp-motion:$LD_LIBRARY_PATH

# App runtime
WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
COPY --from=preproc-build /build/preprocess/build/libramp_motion_preproc.so \
     /opt/ramp-motion/libramp_motion_preproc.so
COPY ramp_motion/ ramp_motion/
COPY configs/ configs/
CMD ["python3", "-m", "ramp_motion.app"]
```

**Notes on DS 6.3 specifics:**
- Python bindings: `pyds 1.1.8` ships as a pre-built wheel for CUDA 12.1 / Ubuntu 20.04 (the DS 6.3 base). Installed via the GitHub release URL above — no local build required.
- OpenCV CUDA build is the longest step (~20–30 min on first build). Stage 1 is cached so rebuilds of stages 2/3 are fast.
- `CUDA_ARCH_BIN=7.5,8.0,8.6,8.9` covers T4, A100, A10/A30, L4/L40/4090. Prune to only `8.9` (RTX 4090) to shave 5 min off the build if targeting a single SKU.

### 9.2 docker-compose.yml

```yaml
services:
  ramp-motion:
    build: .
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,video,utility
    volumes:
      - ./config.yaml:/app/config.yaml:ro
      - ./data:/data
    restart: unless-stopped
```

GPU access via `runtime: nvidia`. Tested on Selectel (driver ≥ 525 + `nvidia-container-toolkit` required for CUDA 12.1 runtime inside DS 6.3 base image).

## 10. Testing strategy

| Level | What | Needs |
|---|---|---|
| Unit (pure) | `state_machine.advance` — ≥ 12 scenarios (init baseline, debounce reject, cycle complete, peak update, calib rate-limit, disconnect mid-cycle, reconnect, disconnect before first ref) | Python only |
| Unit | `manifest.write()` — atomic write, re-open, schema | Python only |
| Unit | `frame_saver` — mock executor, verify atomic rename order | Python only |
| Unit | `.so` via `ctypes` — feed numpy images (synthetic ramp-like scene), assert `motion_fraction` rises with injected motion | GPU required |
| Integration | Full pipeline with local RTSP simulator (`gst-rtsp-server` serving a looped mp4 with staged motion events) | Selectel GPU |
| Smoke | Real RTSP camera for 1 hour — expect ≥ 1 `cycle_complete` and no leaks | Selectel GPU + RTSP URL |

## 11. Implementation order (for writing-plans)

1. Scaffold project (pyproject, Dockerfile, docker-compose, directory layout, empty modules).
2. `state_machine.py` + full unit-test suite (TDD — pure logic before any native code).
3. `config.py` (Pydantic models matching `config.yaml`) + tests.
4. `preprocess/` — CMake skeleton, stub `.so` exporting entry points, `ctypes` smoke test from Python.
5. `.so` — CLAHE + MOG2 on CUDA via OpenCV, user-meta emission, ctypes test with synthetic images.
6. `app.py` — build pipeline for ONE RTSP (no probe yet), verify frames flow to `fakesink`.
7. `probe.py` — extract user-meta, drive state machine, log actions (no saving).
8. `frame_saver.py` + `manifest.py` — atomic JPEG + JSON writes, integrate into probe.
9. Disconnect detection + MOG2 state reset on reconnect + `events.log`.
10. Calibration frame saver (rate-limited).
11. ROI calibration CLI (`tools/calibrate_roi.py` using OpenCV `selectROI`).
12. Multi-RTSP support (`nvstreammux` batch > 1).
13. End-to-end smoke on Selectel with RTSP simulator.
14. End-to-end smoke on real RTSP camera.

## 12. Explicit non-goals (v1)

- **No object detection** — no YOLO, no `nvinfer`, no `nvtracker`.
- **No database** — no SQLite, no Redis. Manifests on disk only.
- **No REST API** — optional `/health` HTTP only; all data accessible via filesystem.
- **No S3** — local volume only. Add as a follow-up if the hypothesis validates.
- **No multi-ROI per stream** — single rectangular ROI per stream. Extension is straightforward (loop per ROI in state machine and .so).
- **No event streaming** — `events.log` is a terminal sink. No Kafka/NATS/Redis pub-sub.
- **No UI / dashboard** — operators open JPEGs directly from disk.

Each of these is a deliberate YAGNI choice to keep the hypothesis test cheap and fast.
