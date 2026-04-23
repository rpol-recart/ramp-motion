from __future__ import annotations
import argparse
import logging
import signal
import sys
from pathlib import Path

import gi
gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst  # type: ignore

from ramp_motion.config import AppConfig, load_config

log = logging.getLogger(__name__)


def _require(elem, name: str):
    if elem is None:
        raise RuntimeError(f"Failed to create element: {name}")
    return elem


def _set_property_if_exists(elem, prop: str, value) -> None:
    """Set a GObject property only if it exists on this element.
    DeepStream 6.3 and 8.0 share most nvurisrcbin properties but
    `rtsp-reconnect-attempts` was added after 6.3 — silently skip it
    there rather than raising TypeError.
    """
    try:
        if elem.find_property(prop) is not None:
            elem.set_property(prop, value)
    except TypeError:
        pass


def _patch_preprocess_config(cfg: AppConfig, template: Path, output: Path) -> None:
    """Rewrite nvdspreprocess config: keep [property]+[user-configs] from
    the template, then append one [group-N] section per stream with its ROI.
    """
    text = template.read_text()
    # Drop anything from a [group-* section onward — we regenerate.
    kept_lines = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("[group-"):
            break
        kept_lines.append(line)

    out = "\n".join(kept_lines).rstrip() + "\n\n"
    for idx, s in enumerate(cfg.streams):
        r = s.roi
        out += (
            f"[group-{idx}]\n"
            f"src-ids={idx}\n"
            f"custom-input-transformation-function=CustomTransformation\n"
            f"process-on-roi=1\n"
            f"roi-params-src-{idx}={r.x};{r.y};{r.width};{r.height}\n\n"
        )
    output.write_text(out)


def build_pipeline(cfg: AppConfig, preprocess_cfg_path: Path) -> Gst.Pipeline:
    Gst.init(None)
    pipeline = Gst.Pipeline.new("ramp-motion")

    streammux = _require(
        Gst.ElementFactory.make("nvstreammux", "nvstreammux"), "nvstreammux")
    streammux.set_property("width", 1920)
    streammux.set_property("height", 1080)
    streammux.set_property("batch-size", len(cfg.streams))
    streammux.set_property("live-source", 1)
    streammux.set_property("batched-push-timeout", 40000)
    # Host-pinned memory so our .so can NvBufSurfaceMap the input frame for
    # CPU-side OpenCV processing. Device-only memory (type=2, Blackwell default)
    # is not mappable and yields "mapping of memory type (2) not supported".
    streammux.set_property("nvbuf-memory-type", 1)
    pipeline.add(streammux)

    # Convert NV12 → RGBA so pyds.get_nvds_buf_surface can map frames for us.
    nvvidconv = _require(
        Gst.ElementFactory.make("nvvideoconvert", "nvvidconv_rgba"),
        "nvvideoconvert")
    nvvidconv.set_property("nvbuf-memory-type", 1)  # pinned, CPU-mappable
    pipeline.add(nvvidconv)

    capsfilter = _require(
        Gst.ElementFactory.make("capsfilter", "nvvidconv_capsfilter"),
        "capsfilter")
    capsfilter.set_property(
        "caps",
        Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA"),
    )
    pipeline.add(capsfilter)

    preproc = _require(
        Gst.ElementFactory.make("nvdspreprocess", "nvdspreprocess"), "nvdspreprocess")
    preproc.set_property("config-file", str(preprocess_cfg_path))
    pipeline.add(preproc)

    streammux.link(nvvidconv)
    nvvidconv.link(capsfilter)
    capsfilter.link(preproc)

    sink = _require(Gst.ElementFactory.make("fakesink", "fakesink"), "fakesink")
    sink.set_property("sync", 0)
    sink.set_property("async", 0)
    sink.set_property("enable-last-sample", 0)
    pipeline.add(sink)
    preproc.link(sink)

    from ramp_motion.probe import build_probe_context_with_io, preprocess_src_pad_probe
    ctx = build_probe_context_with_io(cfg)
    src_pad = preproc.get_static_pad("src")
    src_pad.add_probe(
        Gst.PadProbeType.BUFFER,
        lambda pad, info: preprocess_src_pad_probe(pad, info, ctx),
    )

    for idx, s in enumerate(cfg.streams):
        src = _require(
            Gst.ElementFactory.make("nvurisrcbin", f"src-{idx}"), "nvurisrcbin")
        src.set_property("uri", s.rtsp_url)
        _set_property_if_exists(src, "rtsp-reconnect-interval",
                                 cfg.pipeline.rtsp_reconnect_interval_sec)
        _set_property_if_exists(src, "rtsp-reconnect-attempts",
                                 cfg.pipeline.rtsp_reconnect_attempts)
        # Force TCP transport — many RTSP servers block UDP from containers.
        # GstRTSPLowerTrans.TCP == 0x4. Harmless for servers that accept UDP too.
        _set_property_if_exists(src, "select-rtp-protocol", 4)
        pipeline.add(src)
        # GStreamer 1.20+ (DS 8.0): request_pad_simple. GStreamer 1.16 (DS 6.3):
        # deprecated get_request_pad still exists and is the only option.
        if hasattr(streammux, "request_pad_simple"):
            pad = streammux.request_pad_simple(f"sink_{idx}")
        else:
            pad = streammux.get_request_pad(f"sink_{idx}")
        src.connect("pad-added", lambda _src, _pad, sink_pad=pad: _pad.link(sink_pad))

    return pipeline


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/app/config.yaml")
    parser.add_argument("--preprocess-template",
                        default="/app/configs/nvdspreprocess_config.txt")
    parser.add_argument("--preprocess-config-out",
                        default="/tmp/nvdspreprocess_runtime.txt")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = load_config(Path(args.config))
    _patch_preprocess_config(
        cfg, Path(args.preprocess_template), Path(args.preprocess_config_out))

    pipeline = build_pipeline(cfg, Path(args.preprocess_config_out))
    loop = GLib.MainLoop()

    def _on_message(_bus, message):
        t = message.type
        if t == Gst.MessageType.ERROR:
            err, dbg = message.parse_error()
            log.error("GStreamer error: %s (%s)", err, dbg)
            loop.quit()
        elif t == Gst.MessageType.EOS:
            log.info("EOS received")
            try:
                import ctypes as _ct
                lib = _ct.CDLL("libramp_motion_preproc.so")
                lib.ramp_motion_preproc_reset_source.argtypes = [_ct.c_uint32]
                for idx in range(len(cfg.streams)):
                    lib.ramp_motion_preproc_reset_source(idx)
            except OSError:
                pass
            loop.quit()
        return True

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", _on_message)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: loop.quit())

    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    finally:
        pipeline.set_state(Gst.State.NULL)
    return 0


if __name__ == "__main__":
    sys.exit(main())
