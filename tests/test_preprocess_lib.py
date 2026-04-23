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
    bg = np.full((240, 320, 3), 128, dtype=np.uint8)
    for i in range(100):
        _process(lib, bg, source_id=1, ts_ns=i * 33_000_000)

    fg = bg.copy()
    fg[80:160, 120:200] = 255
    m = _process(lib, fg, source_id=1, ts_ns=100 * 33_000_000)

    assert m.motion_fraction > 0.01, f"expected motion, got {m.motion_fraction}"
    assert m.max_blob_area_px >= 1000


def test_per_source_state_isolated(lib):
    bg = np.full((240, 320, 3), 50, dtype=np.uint8)
    for i in range(100):
        _process(lib, bg, source_id=7, ts_ns=i * 33_000_000)
    m8 = _process(lib, bg, source_id=8, ts_ns=0)
    assert m8.source_id == 8
