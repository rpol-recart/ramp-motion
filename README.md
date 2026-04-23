# ramp-motion  —  `ds6.3` branch

A DeepStream **6.3** (CUDA 12.1, Python 3.8) build of the ramp-motion
pipeline, targeted at production GPUs with **compute capability 8.6
(Ampere — A100, A40, A6000, RTX 3090)** and NVIDIA driver ≥ 525. The
`main` branch targets DeepStream 8.0 / Blackwell instead.

Same behaviour: watches N RTSP streams, runs CLAHE + MOG2 motion
detection in a custom `nvdspreprocess` `.so`, and a Python pad-probe
drives a pure-function state machine that saves a `ref` / `query` JPEG
pair per motion cycle — plus per-cycle peak calibration frames and a
JSON manifest. No object detection, no DB, no REST: local volume +
JSONL events only.

The sample `config.yaml` declares **5 streams** out of the box (a
common batched-capture configuration).

- [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) — **step-by-step install & run guide**
  (host prerequisites, Docker build with online/offline/air-gapped variants,
  smoke-tests, troubleshooting).
- [`docs/superpowers/specs/2026-04-23-ramp-motion-pipeline-design.md`](docs/superpowers/specs/2026-04-23-ramp-motion-pipeline-design.md)
  — full design spec.
- [`docs/superpowers/plans/2026-04-23-ramp-motion-pipeline.md`](docs/superpowers/plans/2026-04-23-ramp-motion-pipeline.md)
  — the step-by-step TDD plan that was used to build it.

## Requirements

- NVIDIA GPU (tested on Blackwell SM 12.0; Ada/Hopper supported via the same
  base image)
- NVIDIA driver ≥ 560 (Blackwell) or ≥ 525 (Ada/Hopper)
- `docker` + `nvidia-container-toolkit`

## Quickstart (host — unit tests only)

```bash
pip install -e ".[dev]"
pytest -m "not gpu and not integration"
```

## Quickstart (Docker + GPU)

```bash
# 1. Edit config.yaml — replace the placeholder RTSP URL with your own.
#    Use tools/calibrate_roi.py to pick an ROI interactively.
# 2. Build + run.
docker compose up --build
# ref/query JPEGs appear under ./data/streams/<stream_id>/
```

First build is long (~30–45 min) because it compiles OpenCV 4.10 with CUDA
from source. Subsequent builds are cached and finish in minutes.

## Configuration

Edit `config.yaml`. Key knobs:

| Key | Meaning |
|---|---|
| `streams[].rtsp_url` | RTSP URL (TCP forced at runtime via `select-rtp-protocol=4`) |
| `streams[].roi` | Rectangle in the muxer frame (default 1920×1080) |
| `motion.thresh` | Fraction of ROI pixels classified as foreground that counts as motion |
| `motion.t_quiet_sec` | Seconds of continuous low motion before `SAVE_REF` / `SAVE_QUERY` |
| `motion.t_enter_sec` | Debounce before entering `IN_MOTION` (rejects single-frame glitches) |
| `motion.min_blob_area_px` | Ignore noise blobs smaller than this |
| `mog2.*`, `clahe.*` | Tuning passed to `cv::cuda::BackgroundSubtractorMOG2` / `cv::cuda::CLAHE` |

## End-to-End Smoke Test

1. Put a real RTSP URL in `config.yaml` and set `roi` using
   `python tools/calibrate_roi.py --rtsp <url> --stream-id cam-01`.
2. `docker compose up --build`.
3. In another shell, `docker compose logs -f ramp-motion`. After ~5 seconds
   of a quiet scene you should see a `ref_saved` event in
   `./data/events.log` and a JPEG under `./data/streams/cam-01/ref/`.
4. Induce motion in the scene for ~10 s, then stop.
5. After another 5 s of quiet you should see `cycle_complete` in
   `./data/events.log` and a cycle manifest under
   `./data/streams/cam-01/cycles/`.

### Success criteria

- At least one complete cycle is recorded within a few minutes of real
  motion activity.
- `ref` and `query` JPEGs are openable and the scene in `query` differs
  from `ref`.
- `query_path` and `new_ref_path` refer to the same inode:
  `stat -c %i data/streams/cam-01/query/*.jpg data/streams/cam-01/ref/*.jpg`
- No `.tmp` files are left behind anywhere under `./data/`.

## Directory layout

```
ramp-motion/
├── preprocess/          # C++/CUDA custom library for nvdspreprocess
│   ├── CMakeLists.txt
│   ├── include/
│   └── src/
│       ├── custom_lib.cpp   # DS entry points (CustomTensorPreparation, initLib, …)
│       ├── clahe_mog2.cu    # CLAHE + MOG2 via cv::cuda
│       └── motion_meta.cpp  # NvDsUserMeta attachment
├── configs/
│   ├── deepstream_app.yaml
│   └── nvdspreprocess_config.txt    # template; ROIs are patched in at runtime
├── ramp_motion/         # Python pipeline + state machine
│   ├── app.py           # GStreamer pipeline builder
│   ├── probe.py         # pad-probe + dispatcher
│   ├── state_machine.py # pure advance(state, sample) → (state, action)
│   ├── frame_saver.py   # thread-pool JPEG writer with atomic rename + hardlink
│   ├── manifest.py      # cycle JSON writer (atomic)
│   ├── events_log.py    # thread-safe JSONL events
│   └── config.py        # Pydantic config models
├── tests/               # pure-Python unit tests + ctypes GPU tests
├── tools/
│   ├── calibrate_roi.py # CLI: OpenCV selectROI on first RTSP frame
│   └── …
├── Dockerfile           # 3-stage: OpenCV-CUDA → custom .so → runtime+pyds
└── docker-compose.yml
```

## License

See `LICENSE` (to be added).
