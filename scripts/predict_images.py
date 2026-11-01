#!/usr/bin/env python3
"""
Run the model on a folder of images and print a per-class tally.

  python scripts/predict_images.py --model models/seed_model.pt --source path/to/images
"""
import argparse
from collections import Counter

from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="models/seed_model.pt")
ap.add_argument("--source", required=True)
ap.add_argument("--conf", type=float, default=0.56)
args = ap.parse_args()

model = YOLO(args.model)
tally = Counter()
for r in model.predict(args.source, conf=args.conf, save=True, stream=True, verbose=False):
    for c in r.boxes.cls.tolist():
        tally[model.names[int(c)]] += 1

print("Detections per class:")
for name, n in sorted(tally.items()):
    print(f"  {name:<18} {n}")
print("Annotated images saved under runs/detect/predict*/")
