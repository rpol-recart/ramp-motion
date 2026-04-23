from __future__ import annotations
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Optional, Tuple


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


def _advance_waiting_motion(st: StreamState, s: MotionSample, th: MotionThresholds
                             ) -> Tuple[StreamState, Action, CycleStats]:
    motion_active = (
        s.motion_fraction >= th.motion_thresh
        and s.max_blob_area_px >= th.min_blob_area_px
    )
    if not motion_active:
        # Anchor debounce at this calm frame's timestamp so a brief motion-calm-motion
        # flicker still counts toward t_enter_ns when motion resumes (vs. a hard reset).
        return replace(st, motion_since_ts_ns=s.ts_ns, last_sample_ts_ns=s.ts_ns), \
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


def _advance_in_motion(st: StreamState, s: MotionSample, th: MotionThresholds
                        ) -> Tuple[StreamState, Action, CycleStats]:
    motion_active = (
        s.motion_fraction >= th.motion_thresh
        and s.max_blob_area_px >= th.min_blob_area_px
    )

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

    if motion_active:
        return (
            replace(st, cycle_stats=new_cs, quiet_since_ts_ns=None,
                    last_sample_ts_ns=s.ts_ns),
            action, new_cs,
        )

    if st.quiet_since_ts_ns is None:
        return (
            replace(st, cycle_stats=new_cs, quiet_since_ts_ns=s.ts_ns,
                    last_sample_ts_ns=s.ts_ns),
            action, new_cs,
        )

    if s.ts_ns - st.quiet_since_ts_ns >= th.t_quiet_ns:
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


def advance(st: StreamState, s: MotionSample, th: MotionThresholds
            ) -> Tuple[StreamState, Action, CycleStats]:
    if s.disconnected:
        return StreamState(), Action.RESET, CycleStats()
    if st.state == State.INIT:
        return _advance_init(st, s, th)
    if st.state == State.WAITING_MOTION:
        return _advance_waiting_motion(st, s, th)
    if st.state == State.IN_MOTION:
        return _advance_in_motion(st, s, th)
    raise AssertionError(f"unreachable state: {st.state}")


def is_stale(st: StreamState, now_ns: int, th: MotionThresholds) -> bool:
    if st.last_sample_ts_ns is None:
        return False
    return (now_ns - st.last_sample_ts_ns) >= th.t_stale_ns
