import pytest
from dataclasses import replace
from ramp_motion.state_machine import (
    State, Action, MotionSample, CycleStats, StreamState,
)
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
    with pytest.raises(AttributeError):
        s.motion_fraction = 0.2


def test_default_stream_state_is_init_no_quiet_timer():
    st = StreamState()
    assert st.state == State.INIT
    assert st.quiet_since_ts_ns is None
    assert st.motion_since_ts_ns is None
    assert st.cycle_stats.peak_fraction == 0.0


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
    st, _, _ = advance(st, _sample(2.1, motion=0.0), THRESH)
    st, action, _ = advance(st, _sample(5.0, motion=0.0), THRESH)
    assert st.state == State.INIT
    assert action == Action.NONE


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
    new_st, action, _ = advance(st, _sample(1.0, motion=0.5, blob=100), THRESH)
    assert new_st.state == State.WAITING_MOTION
    assert action == Action.NONE


def test_waiting_requires_debounce_to_enter_in_motion():
    st = _waiting_state(0.0)
    st, _, _ = advance(st, _sample(1.0, motion=0.5, blob=5000), THRESH)
    assert st.state == State.WAITING_MOTION
    st, _, _ = advance(st, _sample(1.2, motion=0.5, blob=5000), THRESH)
    assert st.state == State.WAITING_MOTION
    st, action, _ = advance(st, _sample(1.6, motion=0.5, blob=5000), THRESH)
    assert st.state == State.IN_MOTION
    assert action == Action.NONE


def test_waiting_debounce_reset_on_calm_frame():
    """A calm frame anchors the debounce at its own timestamp, so 0.5s later
    the motion resumes and crosses t_enter_ns (1.2 calm → 1.7 motion = 500ms)."""
    st = _waiting_state(0.0)
    st, _, _ = advance(st, _sample(1.0, motion=0.5, blob=5000), THRESH)
    st, _, _ = advance(st, _sample(1.2, motion=0.0), THRESH)
    st, _, _ = advance(st, _sample(1.3, motion=0.5, blob=5000), THRESH)
    st, action, _ = advance(st, _sample(1.7, motion=0.5, blob=5000), THRESH)
    assert st.state == State.IN_MOTION


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
    st = _in_motion_state(peak=0.1, last_calib_sec=0.0)
    st, action, _ = advance(st, _sample(0.2, motion=0.3, blob=5000), THRESH)
    assert action == Action.NONE
    assert st.cycle_stats.peak_fraction == pytest.approx(0.3)
    assert st.cycle_stats.last_calib_ts_ns == 0


def test_in_motion_no_peak_update_below_existing_peak():
    st = _in_motion_state(peak=0.5, last_calib_sec=-10.0)
    st, action, _ = advance(st, _sample(0.0, motion=0.3, blob=5000), THRESH)
    assert action == Action.NONE
    assert st.cycle_stats.peak_fraction == pytest.approx(0.5)


def test_in_motion_starts_quiet_timer_when_motion_drops():
    st = _in_motion_state(peak=0.3, last_calib_sec=-10.0)
    st, _, _ = advance(st, _sample(1.0, motion=0.0), THRESH)
    assert st.state == State.IN_MOTION
    assert st.quiet_since_ts_ns == int(1.0 * 1_000_000_000)


def test_in_motion_quiet_timer_resets_on_motion_return():
    st = _in_motion_state(peak=0.3, last_calib_sec=-10.0)
    st, _, _ = advance(st, _sample(1.0, motion=0.0), THRESH)
    st, _, _ = advance(st, _sample(2.0, motion=0.5, blob=5000), THRESH)
    assert st.state == State.IN_MOTION
    assert st.quiet_since_ts_ns is None


def test_in_motion_transitions_to_waiting_after_t_quiet_and_saves_query():
    st = _in_motion_state(peak=0.3, last_calib_sec=-10.0)
    st, _, _ = advance(st, _sample(1.0, motion=0.01), THRESH)
    st, _, _ = advance(st, _sample(3.0, motion=0.01), THRESH)
    st, action, cs = advance(st, _sample(6.0, motion=0.01), THRESH)
    assert st.state == State.WAITING_MOTION
    assert action == Action.SAVE_QUERY_AND_NEW_REF
    assert cs.peak_fraction == pytest.approx(0.3)
    assert st.ref_ts_ns == int(6.0 * 1_000_000_000)
    assert st.cycle_stats.peak_fraction == 0.0


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
    assert is_stale(st, now_ns=11 * 1_000_000_000, th=THRESH) is True
    assert is_stale(st, now_ns=5 * 1_000_000_000, th=THRESH) is False
    assert is_stale(StreamState(), now_ns=10**12, th=THRESH) is False


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
    assert s_a.quiet_since_ts_ns is None
    assert s_b.quiet_since_ts_ns == 0
