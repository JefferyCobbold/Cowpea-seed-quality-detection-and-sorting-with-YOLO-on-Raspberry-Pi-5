# Models

Put the trained weights here as `seed_model.pt` (the `best.pt` from `runs/detect/train/weights/`).

Export to NCNN for the Raspberry Pi:

```bash
python scripts/export_ncnn.py --weights models/seed_model.pt
```

YOLO11s weights are ~19 MB, so they fit in a normal Git commit. For bigger models, attach them to a GitHub Release instead.
