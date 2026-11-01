#!/usr/bin/env python3
"""
Automated cowpea seed sorter — Raspberry Pi 5 + Pi Camera + 2 servos + 16x2 LCD.

Cycle:
  1. Feeder servo pushes one seed into the camera's field of view.
  2. The custom YOLO model classifies the seed (Broken / Damaged / Intact / Perforated).
  3. The sorting servo swings the gate to the bin for that class.
  4. The class + confidence is shown on the LCD and logged to the terminal.
  5. Servos reset and the next seed is fed in.

Usage:
  python rpi/seed_sorter.py --model models/seed_model_ncnn_model
  python rpi/seed_sorter.py --model models/seed_model.pt --show    # with preview window
  python rpi/seed_sorter.py --model models/seed_model.pt --dry-run # no servos/LCD (test on a PC)

Edit the CONFIG block below to match your wiring and bin layout.
"""
import argparse
import csv
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

# ----------------------------- CONFIG ---------------------------------------
FEEDER_PIN = 18           # BCM pin for the feeder servo signal wire
SORTER_PIN = 13           # BCM pin for the sorting-gate servo signal wire

FEEDER_REST_DEG = 0       # feeder arm position while waiting
FEEDER_PUSH_DEG = 90      # feeder arm position that pushes a seed forward
SORTER_HOME_DEG = 0       # gate position between seeds

# Gate angle for each class (one bin per angle). Intact seeds go to 90°.
SORT_ANGLES = {
    "Intact_Seeds": 90,
    "Broken_Seeds": 30,
    "Damaged_Seeds": 150,
    "Perforated_Seeds": 180,
}
REJECT_ANGLE = 150        # where low-confidence / unknown detections go

CONF_THRESHOLD = 0.56     # best overall F1 (0.91) on the validation set was at conf ≈ 0.559
SETTLE_S = 0.6            # wait after feeding so the seed stops moving before capture
GATE_HOLD_S = 0.8         # how long the gate stays open
LCD_I2C_ADDR = 0x27       # common PCF8574 backpack address (try 0x3F if blank)
LOG_FILE = "sort_log.csv"
# ----------------------------------------------------------------------------


class Hardware:
    """Servos + LCD. In dry-run mode every call just prints."""

    def __init__(self, dry_run: bool):
        self.dry_run = dry_run
        self.lcd = None
        if dry_run:
            return
        from gpiozero import AngularServo
        from gpiozero.pins.lgpio import LGPIOFactory  # Pi 5 needs lgpio

        f = LGPIOFactory()
        kw = dict(min_angle=0, max_angle=180, min_pulse_width=0.0005,
                  max_pulse_width=0.0025, pin_factory=f)
        self.feeder = AngularServo(FEEDER_PIN, **kw)
        self.sorter = AngularServo(SORTER_PIN, **kw)
        try:
            from RPLCD.i2c import CharLCD
            self.lcd = CharLCD("PCF8574", LCD_I2C_ADDR, cols=16, rows=2)
            self.lcd.clear()
        except Exception as e:  # LCD is optional
            print(f"[warn] LCD not available: {e}")

    def feeder_to(self, deg):
        print(f"Feeder: moving to {deg}°")
        if not self.dry_run:
            self.feeder.angle = deg

    def sorter_to(self, deg):
        print(f"Setting sorting servo to {deg}°")
        if not self.dry_run:
            self.sorter.angle = deg

    def show(self, line1, line2=""):
        if self.lcd:
            self.lcd.clear()
            self.lcd.write_string(line1[:16])
            self.lcd.crlf()
            self.lcd.write_string(line2[:16])

    def close(self):
        if self.dry_run:
            return
        self.feeder.detach()
        self.sorter.detach()
        if self.lcd:
            self.lcd.clear()


def open_camera(source):
    """Pi Camera via Picamera2, or a USB cam / video file via OpenCV."""
    if source == "picamera":
        from picamera2 import Picamera2
        cam = Picamera2()
        cam.configure(cam.create_preview_configuration(
            main={"format": "RGB888", "size": (640, 480)}))
        cam.start()
        time.sleep(1)
        return lambda: cam.capture_array(), cam.stop
    cap = cv2.VideoCapture(int(source[3:]) if source.startswith("usb") else source)
    return (lambda: cap.read()[1]), cap.release


def classify(model, frame):
    """Return (label, conf, annotated_frame) for the most confident detection."""
    r = model.predict(frame, conf=CONF_THRESHOLD, imgsz=640, verbose=False)[0]
    if len(r.boxes) == 0:
        return None, 0.0, r.plot()
    i = int(r.boxes.conf.argmax())
    return model.names[int(r.boxes.cls[i])], float(r.boxes.conf[i]), r.plot()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help=".pt file or NCNN model folder")
    ap.add_argument("--source", default="picamera", help="picamera | usb0 | path/to/video")
    ap.add_argument("--show", action="store_true", help="show annotated preview window")
    ap.add_argument("--dry-run", action="store_true", help="no GPIO / LCD (test on a PC)")
    ap.add_argument("--max-seeds", type=int, default=0, help="stop after N seeds (0 = run forever)")
    args = ap.parse_args()

    model = YOLO(args.model, task="detect")
    hw = Hardware(args.dry_run)
    grab, release = open_camera(args.source)
    counts = Counter()

    new_log = not Path(LOG_FILE).exists()
    log = open(LOG_FILE, "a", newline="")
    writer = csv.writer(log)
    if new_log:
        writer.writerow(["timestamp", "label", "confidence", "gate_angle"])

    hw.sorter_to(SORTER_HOME_DEG)
    hw.feeder_to(FEEDER_REST_DEG)
    hw.show("Seed sorter", "ready")
    n = 0
    try:
        while not args.max_seeds or n < args.max_seeds:
            # 1. feed one seed
            print("Feeder: Moving object into view")
            hw.feeder_to(FEEDER_PUSH_DEG)
            time.sleep(0.4)
            hw.feeder_to(FEEDER_REST_DEG)
            time.sleep(SETTLE_S)

            # 2. classify
            frame = grab()
            if frame is None:
                print("[warn] no frame from camera")
                break
            t0 = time.perf_counter()
            label, conf, annotated = classify(model, frame)
            dt = (time.perf_counter() - t0) * 1000
            if args.show:
                cv2.imshow("YOLO detection results", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if label is None:
                print(f"No seed detected ({dt:.0f} ms) — skipping")
                hw.show("No seed", "detected")
                continue

            # 3. sort
            angle = SORT_ANGLES.get(label, REJECT_ANGLE)
            print(f"Detected '{label}' ({conf:.2f}, {dt:.0f} ms)")
            hw.show(label.replace("_Seeds", ""), f"conf {conf:.2f}")
            hw.sorter_to(angle)
            time.sleep(GATE_HOLD_S)
            print("Resetting sorting servo")
            hw.sorter_to(SORTER_HOME_DEG)

            # 4. log
            counts[label] += 1
            n += 1
            writer.writerow([datetime.now().isoformat(timespec="seconds"), label, f"{conf:.3f}", angle])
            log.flush()
    except KeyboardInterrupt:
        pass
    finally:
        print("\nSorted:", dict(counts))
        log.close()
        release()
        hw.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
