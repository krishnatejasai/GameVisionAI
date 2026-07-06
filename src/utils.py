from pathlib import Path
import cv2

SEQUENCE_PREFIX = "frame_0_"
IMAGE_DIR = Path("data/raw/football_players/players/images/val")
OUTPUT_VIDEO = Path("data/sample_videos/frame0_sequence.mp4")

OUTPUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)

image_paths = sorted(
    [
        p for p in IMAGE_DIR.iterdir()
        if p.name.startswith(SEQUENCE_PREFIX)
        and p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ],
    key=lambda p: int(p.stem.split("_")[-1])
)

print(f"Frames found: {len(image_paths)}")

first_frame = cv2.imread(str(image_paths[0]))
height, width = first_frame.shape[:2]

writer = cv2.VideoWriter(
    str(OUTPUT_VIDEO),
    cv2.VideoWriter_fourcc(*"mp4v"),
    10,
    (width, height),
)

for image_path in image_paths:
    frame = cv2.imread(str(image_path))
    frame = cv2.resize(frame, (width, height))
    writer.write(frame)

writer.release()

print(f"Created video: {OUTPUT_VIDEO}")