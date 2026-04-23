from __future__ import annotations
import ctypes
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Optional

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst  # type: ignore

import numpy as np
import pyds

from ramp_motion.state_machine import (
    Action, CycleStats, MotionSample, MotionThresholds, State, StreamState,
    advance, is_stale,
)
from ramp_motion.config import AppConfig
from ramp_motion.events_log import EventsLog
from ramp_motion.frame_saver import FrameSaver
from ramp_motion.manifest import CycleManifest, write_manifest

log = logging.getLogger(__name__)


def _frame_bgr_from_nvmm(buf, frame_meta) -> Optional[np.ndarray]:
    """Map NVMM buffer for one frame → host BGR numpy array (synchronous)."""
    try:
        frame = pyds.get_nvds_buf_surface(hash(buf), frame_meta.batch_id)
        rgba = np.array(frame, copy=True, order="C")
        bgr = rgba[:, :, [2, 1, 0]]
        return np.ascontiguousarray(bgr)
    except Exception as e:
        log.warning("get_nvds_buf_surface failed: %s", e)
        return None


def _iso_now(ts_ns: int) -> str:
    # ts_ns is GStreamer buf_pts (stream-relative). Live RTSP starts at ~0, so
    # using it directly yields 1970-01-01 timestamps. Use wall-clock for the
    # human-readable filename; keep ts_ns for internal ordering/events only.
    _ = ts_ns
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


# Must match kMotionUserMetaType in preprocess/include/ramp_motion_preproc.h.
# Falls in the NVDS_START_USER_META (0x1000+) range required by DS for
# third-party user meta.
MOTION_USER_META_TYPE = 0x1000 + 0x4D4F


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


ActionHandler = Callable[[str, Action, MotionSample, Optional[object], CycleStats], None]


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


_PROBE_HITS = 0
_PROBE_USER_META_HITS = 0


def preprocess_src_pad_probe(pad, info, ctx: ProbeContext):
    global _PROBE_HITS, _PROBE_USER_META_HITS
    _PROBE_HITS += 1
    if _PROBE_HITS in (1, 10, 50, 200) or _PROBE_HITS % 500 == 0:
        print(f"[probe] hit={_PROBE_HITS} usermeta={_PROBE_USER_META_HITS}",
              flush=True)

    buf = info.get_buffer()
    if buf is None:
        return Gst.PadProbeReturn.OK
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
    if batch_meta is None:
        if _PROBE_HITS <= 3:
            print(f"[probe] hit={_PROBE_HITS} NO BATCH META", flush=True)
        return Gst.PadProbeReturn.OK

    now_ns = buf.pts if buf.pts != Gst.CLOCK_TIME_NONE else 0

    seen_streams = set()
    for frame_meta in _iter_frame_metas(batch_meta):
        source_id = int(frame_meta.source_id)
        stream_id = ctx.stream_id_by_source.get(source_id, f"source-{source_id}")
        seen_streams.add(stream_id)

        image_cache: Optional[np.ndarray] = None
        for user_meta in _iter_user_metas(frame_meta, MOTION_USER_META_TYPE):
            _PROBE_USER_META_HITS += 1
            c_meta = _MotionMetaC.from_address(
                pyds.get_ptr(user_meta.user_meta_data))
            sample = _sample_from_motion_meta(c_meta, stream_id)
            state_prev = ctx.states.get(stream_id, StreamState())
            state_next, action, cycle_stats = advance(state_prev, sample, ctx.thresholds)
            ctx.states[stream_id] = state_next

            if _PROBE_USER_META_HITS % 50 == 1:
                print(
                    f"[motion] {stream_id} state={state_next.state.name} "
                    f"motion_fraction={sample.motion_fraction:.4f} "
                    f"blob={sample.max_blob_area_px} "
                    f"peak={state_next.cycle_stats.peak_fraction:.4f}",
                    flush=True,
                )
            if action != Action.NONE:
                if image_cache is None:
                    image_cache = _frame_bgr_from_nvmm(buf, frame_meta)
                ctx.on_action(stream_id, action, sample, image_cache, cycle_stats)

    # Detect stale streams
    for stream_id, state in list(ctx.states.items()):
        if stream_id in seen_streams:
            continue
        if is_stale(state, now_ns, ctx.thresholds):
            synthetic = MotionSample(
                stream_id=stream_id, ts_ns=now_ns,
                motion_fraction=0.0, max_blob_area_px=0,
                frame_pts=0, disconnected=True,
            )
            state_next, action, cycle_stats = advance(state, synthetic, ctx.thresholds)
            ctx.states[stream_id] = state_next
            if action != Action.NONE:
                ctx.on_action(stream_id, action, synthetic, None, cycle_stats)

    return Gst.PadProbeReturn.OK


def _log_action_handler(stream_id: str, action: Action, sample: MotionSample,
                         image: Optional[object], cycle_stats: CycleStats) -> None:
    log.info("[%s] action=%s ts=%d motion=%.4f blob=%d",
             stream_id, action.name, sample.ts_ns,
             sample.motion_fraction, sample.max_blob_area_px)


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
               sample: MotionSample, image_bgr, cycle_stats: CycleStats) -> None:
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
                motion_peak_fraction=cycle_stats.peak_fraction,
                motion_peak_blob_area_px=cycle_stats.peak_blob_area_px,
                motion_samples_count=cycle_stats.samples_count,
                calibration_frames=list(self._pending_calib_frames.get(stream_id, [])),
                new_ref_path=str(new_ref_path.relative_to(self._root)),
            )
            manifest_dest = base / "cycles" / f"{ts_iso}.json"
            write_manifest(manifest, manifest_dest)
            self._events.write({"event": "cycle_complete", "stream_id": stream_id,
                                "cycle_id": manifest.cycle_id,
                                "duration_sec": duration_sec,
                                "peak": cycle_stats.peak_fraction})

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


def build_probe_context_with_io(cfg: AppConfig) -> ProbeContext:
    saver = FrameSaver(root=Path(cfg.storage.root),
                       jpeg_quality=cfg.storage.jpeg_quality)
    events = EventsLog(Path(cfg.logging.events_log))
    dispatcher = ActionDispatcher(cfg, saver, events)
    return build_probe_context(cfg, on_action=dispatcher.handle)


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
