#!/usr/bin/env python3
"""
Export trained weights to NCNN — the fastest CPU format for Raspberry Pi 5.

  python scripts/export_ncnn.py --weights models/seed_model.pt --imgsz 640

Creates models/seed_model_ncnn_model/ which can be passed straight to --model.
"""
import argparse

from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--weights", default="models/seed_model.pt")
ap.add_argument("--imgsz", type=int, default=640)
args = ap.parse_args()

YOLO(args.weights).export(format="ncnn", imgsz=args.imgsz)
