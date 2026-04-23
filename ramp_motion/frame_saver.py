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
