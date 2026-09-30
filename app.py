"""SnapGuard AI - local road-safety risk detection (image / video / webcam)."""
import argparse
import sys
import time
from pathlib import Path

import cv2

from detector import RoadObjectDetector
from overlay import draw
from risk import assess

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
OUT_DIR = Path("outputs")


def parse_args():
    p = argparse.ArgumentParser(description="SnapGuard AI road-safety risk detection")
    p.add_argument("--source", default="sample",
                   help="'sample', 'webcam', or a path to an image/video file")
    p.add_argument("--weights", default="yolov8n.pt", help="YOLO weights file")
    p.add_argument("--conf", type=float, default=0.35, help="detection confidence threshold")
    p.add_argument("--imgsz", type=int, default=640, help="inference size (lower = faster)")
    p.add_argument("--camera-index", type=int, default=0, help="webcam index")
    p.add_argument("--save", action="store_true", help="save video/webcam output to outputs/result.mp4")
    p.add_argument("--no-show", action="store_true", help="do not open a display window")
    return p.parse_args()


def process(frame, detector):
    t0 = time.perf_counter()
    dets = detector.detect(frame)
    infer_ms = (time.perf_counter() - t0) * 1000.0
    h, w = frame.shape[:2]
    result = assess(dets, w, h)
    return draw(frame, result, infer_ms), result


def run_image(path, detector, show):
    frame = cv2.imread(str(path))
    if frame is None:
        sys.exit(f"Could not read image: {path}")
    out, result = process(frame, detector)
    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / "result.jpg"
    cv2.imwrite(str(out_path), out)
    print(f"Risk level: {result.level}")
    for reason in result.reasons:
        print(f"  - {reason}")
    print(f"Saved: {out_path}")
    if show:
        cv2.imshow("SnapGuard AI", out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_stream(cap, detector, show, save):
    writer = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            out, _ = process(frame, detector)
            if save:
                if writer is None:
                    OUT_DIR.mkdir(exist_ok=True)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    if not fps or fps != fps or fps < 1:
                        fps = 20.0
                    hh, ww = out.shape[:2]
                    writer = cv2.VideoWriter(str(OUT_DIR / "result.mp4"),
                                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (ww, hh))
                writer.write(out)
            if show:
                cv2.imshow("SnapGuard AI (press q to quit)", out)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


def main():
    args = parse_args()
    show = not args.no_show
    detector = RoadObjectDetector(args.weights, args.conf, args.imgsz)

    if args.source == "sample":
        from ultralytics.utils import ASSETS
        run_image(ASSETS / "bus.jpg", detector, show)
    elif args.source == "webcam":
        cap = cv2.VideoCapture(args.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            sys.exit("Could not open webcam. Try --camera-index 1 and check Windows camera privacy settings.")
        run_stream(cap, detector, show, args.save)
    else:
        path = Path(args.source)
        if not path.exists():
            sys.exit(f"File not found: {path}")
        if path.suffix.lower() in IMAGE_EXTS:
            run_image(path, detector, show)
        else:
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                sys.exit(f"Could not open video: {path}")
            run_stream(cap, detector, show, args.save)


if __name__ == "__main__":
    main()
