import os
from pathlib import Path

import numpy as np
import cv2
import pytest

from ramp_motion.frame_saver import FrameSaver


def _img() -> np.ndarray:
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:32] = 255
    return img


def test_save_writes_jpeg_atomically(tmp_path: Path):
    saver = FrameSaver(root=tmp_path, jpeg_quality=85, max_workers=1)
    dest = tmp_path / "ref" / "a.jpg"
    future = saver.save(_img(), dest)
    future.result()
    assert dest.exists()
    decoded = cv2.imread(str(dest))
    assert decoded is not None
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
    assert os.stat(primary).st_ino == os.stat(linked).st_ino
    saver.shutdown()
