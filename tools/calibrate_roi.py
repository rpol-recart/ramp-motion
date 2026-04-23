"""CLI: connect to an RTSP stream, grab one frame, prompt the operator to draw
a rectangular ROI, and print it in YAML format ready to paste into config.yaml.
"""
from __future__ import annotations
import argparse
import sys

import cv2


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rtsp", required=True, help="RTSP URL")
    parser.add_argument("--stream-id", default="cam-XX")
    args = parser.parse_args(argv)

    cap = cv2.VideoCapture(args.rtsp, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print(f"error: cannot open {args.rtsp}", file=sys.stderr)
        return 1

    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        print("error: failed to grab a frame", file=sys.stderr)
        return 2

    win = f"Draw ROI for {args.stream_id} — ENTER to confirm, c to cancel"
    roi = cv2.selectROI(win, frame, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    x, y, w, h = map(int, roi)
    if w <= 0 or h <= 0:
        print("error: empty ROI", file=sys.stderr)
        return 3

    print(f"""\
  - stream_id: {args.stream_id}
    rtsp_url: {args.rtsp}
    roi: {{ x: {x}, y: {y}, width: {w}, height: {h} }}
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
