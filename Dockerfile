# syntax=docker/dockerfile:1.6

# =========================================================================
# Stage 1: build CUDA-enabled OpenCV 4.10 (for cv::cuda::CLAHE and MOG2)
# Base is DeepStream 8.0 = CUDA 12.8 + Python 3.12; supports Blackwell SM 10/12.
# =========================================================================
FROM nvcr.io/nvidia/deepstream:8.0-triton-multiarch AS opencv-build
ARG OPENCV_VERSION=4.10.0
# SM 8.9 (Ada/L40/4090), 9.0 (Hopper). Blackwell (SM 10.0/12.0) is reached at
# runtime via PTX JIT from the highest bin arch.
ARG CUDA_ARCH_BIN=8.9;9.0
ARG CUDA_ARCH_PTX=9.0

# Source selector for the OpenCV tree:
#   zip (default)     — download https://github.com/opencv/opencv/archive/refs/tags/${OPENCV_VERSION}.zip
#                        (no git dependency at build time; works in networks
#                         that allow HTTPS but block the git smart protocol)
#   git               — `git clone --depth 1 --branch ${OPENCV_VERSION}`
#   /opt/vendor-zip   — use a pre-downloaded zip copied into the image
#                        (fully offline builds). See DEPLOYMENT.md §5.
ARG OPENCV_SOURCE=zip

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      cmake build-essential pkg-config ca-certificates \
      curl unzip git \
      libgtk-3-dev libavcodec-dev libavformat-dev libswscale-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy optional pre-downloaded vendor zips for offline builds. These files
# are ignored if the directory is empty — the build-arg OPENCV_SOURCE still
# decides which path runs.
COPY vendor/ /opt/vendor/

WORKDIR /opt
RUN set -eux; \
    case "${OPENCV_SOURCE}" in \
      git) \
        git clone --depth 1 --branch ${OPENCV_VERSION} https://github.com/opencv/opencv.git ; \
        git clone --depth 1 --branch ${OPENCV_VERSION} https://github.com/opencv/opencv_contrib.git ; \
        ;; \
      zip) \
        curl -fsSL -o opencv.zip          "https://github.com/opencv/opencv/archive/refs/tags/${OPENCV_VERSION}.zip" ; \
        curl -fsSL -o opencv_contrib.zip  "https://github.com/opencv/opencv_contrib/archive/refs/tags/${OPENCV_VERSION}.zip" ; \
        unzip -q opencv.zip          && mv "opencv-${OPENCV_VERSION}"         opencv ; \
        unzip -q opencv_contrib.zip  && mv "opencv_contrib-${OPENCV_VERSION}" opencv_contrib ; \
        rm -f opencv.zip opencv_contrib.zip ; \
        ;; \
      /opt/vendor-zip) \
        test -f /opt/vendor/opencv-${OPENCV_VERSION}.zip           || { echo "missing vendor/opencv-${OPENCV_VERSION}.zip" ; exit 1; } ; \
        test -f /opt/vendor/opencv_contrib-${OPENCV_VERSION}.zip   || { echo "missing vendor/opencv_contrib-${OPENCV_VERSION}.zip" ; exit 1; } ; \
        unzip -q /opt/vendor/opencv-${OPENCV_VERSION}.zip         && mv "opencv-${OPENCV_VERSION}"         opencv ; \
        unzip -q /opt/vendor/opencv_contrib-${OPENCV_VERSION}.zip && mv "opencv_contrib-${OPENCV_VERSION}" opencv_contrib ; \
        ;; \
      *) echo "unknown OPENCV_SOURCE=${OPENCV_SOURCE}"; exit 1 ;; \
    esac; \
    mkdir opencv/build

WORKDIR /opt/opencv/build
RUN cmake .. \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_INSTALL_PREFIX=/opt/opencv-cuda \
    -D OPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib/modules \
    -D WITH_CUDA=ON \
    -D CUDA_ARCH_BIN=${CUDA_ARCH_BIN} \
    -D CUDA_ARCH_PTX=${CUDA_ARCH_PTX} \
    -D BUILD_opencv_python2=OFF \
    -D BUILD_opencv_python3=OFF \
    -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_EXAMPLES=OFF \
    -D WITH_GSTREAMER=ON \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
 && make -j"$(nproc)" && make install

# =========================================================================
# Stage 2: build the custom .so
# =========================================================================
FROM nvcr.io/nvidia/deepstream:8.0-triton-multiarch AS preproc-build
COPY --from=opencv-build /opt/opencv-cuda /opt/opencv-cuda
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      cmake build-essential pkg-config \
      libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
      libglib2.0-dev \
    && rm -rf /var/lib/apt/lists/*
ENV CMAKE_PREFIX_PATH=/opt/opencv-cuda
WORKDIR /build
COPY preprocess/ preprocess/
RUN cmake -S preprocess -B preprocess/build \
      -DOpenCV_DIR=/opt/opencv-cuda/lib/cmake/opencv4 \
 && cmake --build preprocess/build -j"$(nproc)"

# =========================================================================
# Stage 3: runtime — DeepStream 8.0 + pyds 1.2.2 (Python 3.12) + app
# =========================================================================
FROM nvcr.io/nvidia/deepstream:8.0-triton-multiarch

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-gi python3-gst-1.0 \
      python3-dev python-gi-dev \
      libgstrtspserver-1.0-0 libgirepository1.0-dev \
    && rm -rf /var/lib/apt/lists/*

ARG PYDS_VERSION=1.2.2
ARG PYDS_PY_TAG=cp312
RUN pip3 install --no-cache-dir --break-system-packages \
      "https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download/v${PYDS_VERSION}/pyds-${PYDS_VERSION}-${PYDS_PY_TAG}-${PYDS_PY_TAG}-linux_x86_64.whl"

COPY --from=opencv-build /opt/opencv-cuda /opt/opencv-cuda
ENV LD_LIBRARY_PATH=/opt/opencv-cuda/lib:/opt/ramp-motion:/opt/nvidia/deepstream/deepstream/lib:$LD_LIBRARY_PATH

WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt

COPY --from=preproc-build /build/preprocess/build/libramp_motion_preproc.so \
     /opt/ramp-motion/libramp_motion_preproc.so
COPY ramp_motion/ ramp_motion/
COPY configs/ configs/
COPY config.yaml .

CMD ["python3", "-m", "ramp_motion.app", "--config", "/app/config.yaml"]
