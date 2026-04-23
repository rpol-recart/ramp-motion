# Deployment Guide

Step-by-step instructions to deploy `ramp-motion` on a GPU host.

Tested target: **NVIDIA RTX PRO 6000 Blackwell** (SM 12.0) with Ubuntu 24.04,
driver 590.48, Docker 29.2.1. The same steps work on Ada / Hopper with driver
≥ 525.

---

## 0. Checklist

Before you start, confirm on the target host:

```bash
uname -a                                         # Linux kernel
nvidia-smi                                       # driver + GPU must be visible
docker --version                                 # >= 24
ls /etc/docker/daemon.json 2>/dev/null           # runtime config (optional)
docker info | grep -i runtime                    # must list 'nvidia'
df -h /var/lib/docker                            # need ≥ 60 GB free for the image
```

If any of the above is missing, fix it before continuing — see §1.

### 0.1 Environment snapshot (for bug reports)

Copy-paste this one-liner and include the output in any support ticket,
PR, or issue against the repo:

```bash
cat <<EOF
arch=$(uname -m)
driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
compute_cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
glibc=$(ldd --version | head -1)
docker=$(docker --version)
nvct=$(dpkg -l | grep nvidia-container-toolkit | awk '{print $3}')
disk_free=$(df -BG /var/lib/docker | tail -1 | awk '{print $4}')
ncpu=$(nproc)
EOF
```

Expected fields:

| Field | Example | Why it matters |
|---|---|---|
| `arch` | `x86_64` | The Dockerfile targets x86_64 |
| `driver` | `590.48.01` | Must match the "Minimum driver" table in §1.1 |
| `gpu` | `NVIDIA RTX PRO 6000 Blackwell Server Edition` | Determines whether DS 8.0 recognises the GPU |
| `compute_cap` | `12.0` | Pick `CUDA_ARCH_BIN` / `CUDA_ARCH_PTX` accordingly (§3.4) |
| `glibc` | `ldd (Ubuntu GLIBC 2.39-0ubuntu8.7) 2.39` | Must be ≥ 2.34 for pyds 1.2.2 |
| `docker` | `Docker version 29.2.1, build a5c7197` | ≥ 24 |
| `nvct` | `1.17.1-1` | nvidia-container-toolkit — absent → `--gpus all` silently fails |
| `disk_free` | `136G` | Need ≥ 60 GB free on `/var/lib/docker` |
| `ncpu` | `16` | Build parallelism; OpenCV compile scales with this |

---

## 1. Host prerequisites

### 1.1 NVIDIA driver

| GPU family | Minimum driver |
|---|---|
| Blackwell (RTX PRO 6000, B200) | 560+ (590+ recommended) |
| Hopper (H100) | 525+ |
| Ada (L40, L4, 4090) | 525+ |

Install the driver from NVIDIA and reboot. Verify `nvidia-smi` shows the GPU
and prints a "CUDA Version: X.Y" line (this is the *driver* CUDA API, not the
container-side toolkit).

### 1.2 Docker

Version ≥ 24. On Ubuntu 24.04:

```bash
sudo apt update
sudo apt install -y docker.io
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"        # log out / log back in
```

### 1.3 NVIDIA Container Toolkit

Enables `--gpus all` / `runtime: nvidia` inside Docker.

```bash
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
```

This should print the same GPU as the host `nvidia-smi`. If you see
`could not select device driver "" with capabilities: [[gpu]]`, the toolkit
is not wired in — re-run `nvidia-ctk runtime configure` and restart Docker.

### 1.4 Disk space

The first build downloads and compiles OpenCV with CUDA from source.
Expected usage:

| Layer | Size |
|---|---|
| DeepStream 8.0 base image | ~12 GB compressed, ~33 GB on disk |
| OpenCV-CUDA build stage | ~5 GB added |
| Final runtime image | ~37 GB on disk |
| Build cache (tmp during compile) | ~8 GB |

Provision at least **60 GB free** on `/var/lib/docker`.

---

## 2. Clone and configure

```bash
git clone https://github.com/rpol-recart/ramp-motion.git
cd ramp-motion
```

### 2.1 Edit `config.yaml`

Open `config.yaml` and set at least:

```yaml
streams:
  - stream_id: cam-01
    rtsp_url: rtsp://USER:PASS@CAMERA_HOST:554/stream1   # ← your URL
    roi: { x: 320, y: 180, width: 1280, height: 720 }    # ← your ROI
```

- `rtsp_url` can omit `USER:PASS@` if your camera does not require auth.
- `roi` is applied **after** `nvstreammux` rescales the input to 1920×1080
  (see §7 for changing the muxer resolution).

### 2.2 Pick a good ROI

Run the interactive calibration tool on any host that has Python + OpenCV
and a display (can be your laptop — does not need GPU):

```bash
pip install opencv-python-headless==4.11.* numpy PyYAML pydantic
python tools/calibrate_roi.py --rtsp "rtsp://USER:PASS@CAMERA_HOST/stream" \
                               --stream-id cam-01
```

The tool grabs one frame, opens an OpenCV window, waits for you to drag a
rectangle and press ENTER, then prints the YAML fragment to paste into
`config.yaml`.

If the host is headless, run the same command inside a container with
X-forwarding, or record a single frame with `ffmpeg -frames:v 1` and open
it on your workstation.

### 2.3 Tune thresholds (optional)

For static outdoor cameras, noise from wind/shadows is typically at
`motion_fraction ≈ 0.001–0.01`. Keep `motion.thresh` above your measured
noise floor to avoid false positives.

---

## 3. Build

### 3.1 Default path (download OpenCV as ZIP)

The default `OPENCV_SOURCE=zip` fetches OpenCV and opencv_contrib as ZIP
archives from GitHub over HTTPS. No `git clone` required at build time:

```bash
docker compose build
# equivalent: docker build -t ramp-motion:0.1.0 .
```

First run: **~30–45 min** on a 16-core machine (OpenCV compile dominates).

Subsequent rebuilds after edits to `ramp_motion/**` or `preprocess/**`
reuse the cached OpenCV layer and finish in 1–3 min.

### 3.2 Alternative: `git clone`

Useful if your firewall blocks GitHub's codeload ZIP endpoint but allows
the git smart protocol:

```bash
docker build --build-arg OPENCV_SOURCE=git -t ramp-motion:0.1.0 .
```

### 3.3 Alternative: fully offline build from pre-downloaded ZIPs

If the build host has **no outbound internet** but you can stage files into
the repo directory, pre-download the two ZIPs to `vendor/`:

```bash
# On a machine with internet:
mkdir -p vendor
VER=4.10.0
curl -fsSL -o "vendor/opencv-${VER}.zip"         "https://github.com/opencv/opencv/archive/refs/tags/${VER}.zip"
curl -fsSL -o "vendor/opencv_contrib-${VER}.zip" "https://github.com/opencv/opencv_contrib/archive/refs/tags/${VER}.zip"
sha256sum vendor/*.zip  > vendor/SHA256SUMS      # record hashes for provenance
```

Transfer the whole repo (including `vendor/*.zip`) to the air-gapped build
host, then:

```bash
docker build --build-arg OPENCV_SOURCE=/opt/vendor-zip -t ramp-motion:0.1.0 .
```

The Dockerfile `COPY vendor/ /opt/vendor/` stage will pick up the zips,
unzip them, and skip the network. The DeepStream base image still has to
be pulled once — if you also need that offline, see §8.

### 3.4 Build-time arguments

| ARG | Default | Meaning |
|---|---|---|
| `OPENCV_VERSION` | `4.10.0` | OpenCV tag; `opencv_contrib` is pinned to the same tag |
| `OPENCV_SOURCE` | `zip` | `zip`, `git`, or `/opt/vendor-zip` |
| `CUDA_ARCH_BIN` | `8.9;9.0` | CUDA architectures to compile native SASS for |
| `CUDA_ARCH_PTX` | `9.0` | CUDA architecture to emit PTX for (JIT fallback to newer cards) |
| `PYDS_VERSION` | `1.2.2` | pyds release tag on `NVIDIA-AI-IOT/deepstream_python_apps` |
| `PYDS_PY_TAG` | `cp312` | Python ABI tag of the pyds wheel (matches DS 8.0's Python 3.12) |

Blackwell SM 10/12 are reached at runtime via PTX JIT from `PTX=9.0`. If
you want native SASS instead (2–3 min faster cold start), add the card's
compute capability to `CUDA_ARCH_BIN`, e.g. `--build-arg CUDA_ARCH_BIN=8.9;9.0;10.0`.

### 3.5 Verify the image

```bash
docker images | grep ramp-motion                      # should list :0.1.0
docker run --rm --entrypoint '' --gpus all ramp-motion:0.1.0 bash -c '
  python3 -c "import pyds; print(\"pyds ok\")"
  python3 -c "import ctypes; ctypes.CDLL(\"/opt/ramp-motion/libramp_motion_preproc.so\"); print(\".so loads\")"
'
```

Expected output: `pyds ok` / `.so loads`.

---

## 4. Run

### 4.1 docker-compose (recommended)

```bash
mkdir -p data
docker compose up                      # foreground, Ctrl-C to stop
# OR
docker compose up -d                   # detached
docker compose logs -f ramp-motion     # follow logs
```

Artefacts appear under `./data/`:

```
data/
├── events.log                                   # JSONL: ref_saved, cycle_complete, ...
└── streams/cam-01/
    ├── ref/          2026-…-…Z.jpg              # reference frames
    ├── query/        2026-…-…Z.jpg              # query frames (cycle end)
    ├── calibration/  2026-…-…Z_peak=…jpg        # peak-motion frames
    └── cycles/       2026-…-…Z.json             # cycle manifests
```

### 4.2 docker run (manual)

```bash
docker run --rm --gpus all \
  --name ramp-motion-run \
  -e PYTHONUNBUFFERED=1 \
  -v "$(pwd)/config.yaml:/app/config.yaml:ro" \
  -v "$(pwd)/data:/data" \
  ramp-motion:0.1.0
```

### 4.3 Health check

After ~10 s of a quiet camera view, `data/events.log` should contain:

```json
{"event":"ref_saved","stream_id":"cam-01","ref_at":"2026-…"}
```

and `data/streams/cam-01/ref/` should contain a JPEG. If you see neither
after 30 s, jump to §6.

To validate a full cycle, wave something in front of the camera for
~10 s, then step away:

```bash
tail -f data/events.log
# wait for: {"event":"cycle_complete","stream_id":"cam-01", ... }
```

The manifest under `data/streams/cam-01/cycles/<timestamp>.json` describes
the cycle: `cycle_duration_sec`, `motion_peak_fraction`, paired
`ref_path` / `query_path`, and `calibration_frames[]`.

---

## 5. Smoke-test checklist

Run after every fresh deployment:

```bash
# 1. Image present
docker images ramp-motion:0.1.0 --format '{{.Repository}}:{{.Tag}} {{.Size}}'

# 2. GPU reachable from the container
docker run --rm --gpus all ramp-motion:0.1.0 -c 'exit 0' 2>&1 | head -3

# 3. GPU unit tests (.so CLAHE+MOG2 against synthetic frames)
docker run --rm --gpus all --entrypoint '' \
  -e FORCE_GPU_TESTS=1 \
  -e RAMP_MOTION_PREPROC_SO=/opt/ramp-motion/libramp_motion_preproc.so \
  -v "$(pwd)/tests:/app/tests:ro" \
  ramp-motion:0.1.0 bash -lc '
    pip install --break-system-packages -q pytest >/dev/null
    pytest /app/tests/test_preprocess_lib.py -v
  '
# Expected: 3 passed

# 4. Full pipeline end-to-end against your real RTSP
docker compose up -d
sleep 15
ls data/streams/*/ref/ || { echo "no ref frames yet — check logs"; docker compose logs --tail=40 ramp-motion; }
docker compose down
```

---

## 6. Troubleshooting

### 6.1 "not yet supported in this version of the container"

The DeepStream base image refuses to run on a GPU it does not know about.

- Symptom: `ERROR: No supported GPU(s) detected to run this container`
- Cause: DS 6.3 / 7.x on Blackwell
- Fix: ensure you built against `deepstream:8.0-triton-multiarch` (it is
  the default; if you changed it, revert)

### 6.2 `nvbufsurface: mapping of memory type (2) not supported`

Repeats forever, no frames saved.

- Cause: nvstreammux output is device-only memory, which pyds can't map
- Fix: already applied in `app.py` (`nvbuf-memory-type=1` on `nvstreammux`).
  If you subclassed the pipeline, propagate this property.

### 6.3 `get_nvds_buf_surface: Currently we only support RGBA/RGB color Format`

- Cause: raw NV12 reaches the probe
- Fix: already applied — `nvvideoconvert` + `capsfilter(format=RGBA)` is
  inserted between muxer and preprocess in `app.py`.

### 6.4 Container runs but no `ref_saved`

Follow the diagnostic ladder:

```bash
# Is GStreamer flowing?
docker logs ramp-motion-run 2>&1 | grep -E "Protocol set|rtspsrc|nvv4l2decoder" | head -5

# Is our custom .so being called?
docker logs ramp-motion-run 2>&1 | grep "custom_lib" | head -3

# Is our Python probe seeing meta?
docker logs ramp-motion-run 2>&1 | grep -E "\[motion\]|probe\] hit" | head -10

# What motion_fraction are we seeing?
docker logs ramp-motion-run 2>&1 | grep "\[motion\]" | tail -5
```

If `[motion] … motion_fraction=…` is consistently **above** `motion.thresh`,
the scene is too noisy — raise `motion.thresh` or increase `min_blob_area_px`.

If `[motion]` never appears, the .so is not publishing `NvDsUserMeta` —
double-check `custom-tensor-preparation-function=CustomTensorPreparation`
in `configs/nvdspreprocess_config.txt` matches an exported symbol in
the `.so`.

### 6.5 RTSP connects but no frames

Most public cameras require TCP. `select-rtp-protocol=4` is already set.
If your camera is firewalled from the Docker host, test reachability:

```bash
timeout 10 bash -c '</dev/tcp/CAMERA_HOST/554 && echo TCP_OK || echo TCP_FAIL'
ffprobe -rtsp_transport tcp -i "rtsp://CAMERA_HOST/stream1"
```

### 6.6 Build fails with `Unsupported gpu architecture 'compute_XX'`

`CUDA_ARCH_BIN` has a value newer than the container-side CUDA toolkit
supports. DS 8.0 ships CUDA 12.8, which knows SM ≤ 10.0 natively. Drop
the highest value from `CUDA_ARCH_BIN`; forward-compat for newer cards
is handled by `CUDA_ARCH_PTX`.

---

## 7. Production hardening

Things to change before putting this in front of real cameras:

- **Raise `motion.thresh`** to a value well above your measured noise floor
  (0.02–0.05 is typical for indoor/stable scenes).
- **Set `min_blob_area_px`** higher (500–2000) so shadow flicker and
  small animals do not trigger cycles.
- **Bound the data volume** — there is no log rotation for `events.log`
  and no cap on `streams/*/calibration/` growth. Either mount `data/` on
  a dedicated volume with `logrotate` and a cron cleaner, or add a reaper
  to the app.
- **Switch to `restart: always` / systemd** so the container survives
  reboots and the GStreamer pipeline restarts after a hard crash.
- **Lock `nvbuf-memory-type`** to pinned memory in every nvstream-* element
  you add (default on Blackwell is device-only).
- **Pin dependency hashes**: add `vendor/SHA256SUMS` and verify it in
  the Dockerfile before unzipping.

---

## 8. Fully air-gapped deployment

If the build host has **zero** outbound internet:

1. On a machine with internet, pull the DS 8.0 base image:
   ```bash
   docker pull nvcr.io/nvidia/deepstream:8.0-triton-multiarch
   docker save nvcr.io/nvidia/deepstream:8.0-triton-multiarch \
     | gzip > deepstream-8.0.tar.gz
   ```
2. Pre-download OpenCV zips into `vendor/` (see §3.3).
3. Download the pyds wheel:
   ```bash
   curl -fsSL -O \
     https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download/v1.2.2/pyds-1.2.2-cp312-cp312-linux_x86_64.whl
   mv pyds-*.whl vendor/
   ```
   Then edit `Dockerfile` to `pip install /opt/vendor/pyds-*.whl` instead
   of the GitHub URL.
4. Copy the repo + `vendor/*.zip` + `deepstream-8.0.tar.gz` to the target.
5. On the target:
   ```bash
   docker load -i deepstream-8.0.tar.gz
   docker build --build-arg OPENCV_SOURCE=/opt/vendor-zip -t ramp-motion:0.1.0 .
   ```

---

## 9. Upgrades

| Change | Rebuild stage 1? | Rebuild stage 2? | Rebuild stage 3? |
|---|---|---|---|
| Edit `ramp_motion/**.py` | no | no | **yes** |
| Edit `configs/**` | no | no | **yes** |
| Edit `preprocess/**.cpp/.cu/.h` | no | **yes** | **yes** |
| Bump `OPENCV_VERSION` | **yes** | **yes** | **yes** |
| Bump DS base image | **yes** | **yes** | **yes** |

Stage 1 (OpenCV) is the longest. To keep it cached across a `git pull`,
do not touch `Dockerfile` lines 1–65 unless you are actually changing
OpenCV or CUDA arch.
