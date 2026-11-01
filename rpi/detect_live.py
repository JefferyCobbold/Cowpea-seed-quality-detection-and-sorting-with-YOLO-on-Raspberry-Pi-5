#!/usr/bin/env python3
"""
Live seed detection on a Raspberry Pi 5 (or any PC) with FPS and object count overlay.

  python rpi/detect_live.py --model models/seed_model_ncnn_model            # Pi Camera
  python rpi/detect_live.py --model models/seed_model.pt --source usb0       # USB webcam
  python rpi/detect_live.py --model models/seed_model.pt --source clip.mp4   # video file

Press 'q' to quit, 's' to save a snapshot.
"""
import argparse
import time
from collections import deque

import cv2
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--source", default="picamera")
    ap.add_argument("--conf", type=float, default=0.56)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--resolution", default="640x480")
    args = ap.parse_args()

    w, h = map(int, args.resolution.lower().split("x"))
    model = YOLO(args.model, task="detect")

    if args.source == "picamera":
        from picamera2 import Picamera2
        cam = Picamera2()
        cam.configure(cam.create_preview_configuration(main={"format": "RGB888", "size": (w, h)}))
        cam.start()
        grab, release = cam.capture_array, cam.stop
    else:
        cap = cv2.VideoCapture(int(args.source[3:]) if args.source.startswith("usb") else args.source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        grab, release = (lambda: cap.read()[1]), cap.release

    fps_hist = deque(maxlen=30)
    snap = 0
    while True:
        t0 = time.perf_counter()
        frame = grab()
        if frame is None:
            break
        r = model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        out = r.plot()
        fps_hist.append(1 / (time.perf_counter() - t0))
        cv2.putText(out, f"FPS: {sum(fps_hist)/len(fps_hist):.2f}", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(out, f"Number of objects: {len(r.boxes)}", (10, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("Seed detection", out)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord("s"):
            cv2.imwrite(f"snapshot_{snap:03d}.jpg", out)
            snap += 1

    release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
