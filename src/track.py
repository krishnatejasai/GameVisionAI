from pathlib import Path
import csv
from ultralytics import YOLO

MODEL_PATH = "models/gamevision_yolov8s_960.pt"
VIDEO_PATH = "data/sample_videos/match_clip.mp4"
OUTPUT_CSV = Path("outputs/metrics/match_clip_tracking_results.csv")

CLASS_NAMES = {
    0: "player",
    1: "referee",
    2: "ball",
}

OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

model = YOLO(MODEL_PATH)

results = model.track(
    source=VIDEO_PATH,
    conf=0.25,
    iou=0.5,
    tracker="bytetrack.yaml",
    persist=True,
    stream=True,
    save=True,
    project="outputs/videos",
    name="tracked_match_clip"
)

rows = []

for frame_idx, result in enumerate(results):
    boxes = result.boxes

    if boxes is None or boxes.id is None:
        continue

    for box, track_id, class_id, conf in zip(
        boxes.xyxy.cpu().numpy(),
        boxes.id.cpu().numpy(),
        boxes.cls.cpu().numpy(),
        boxes.conf.cpu().numpy(),
    ):
        x1, y1, x2, y2 = box
        class_id = int(class_id)

        rows.append({
            "frame": frame_idx,
            "track_id": int(track_id),
            "class_id": class_id,
            "class_name": CLASS_NAMES.get(class_id, str(class_id)),
            "confidence": float(conf),
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2),
            "center_x": float((x1 + x2) / 2),
            "center_y": float((y1 + y2) / 2),
        })

with OUTPUT_CSV.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"Tracking completed.")
print(f"Saved tracking CSV to: {OUTPUT_CSV}")
print(f"Total tracked detections: {len(rows)}")