import argparse
import csv
from pathlib import Path
from typing import Union

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "gamevision_yolov8s_960.pt"
)

DEFAULT_VIDEO_PATH = (
    PROJECT_ROOT
    / "data"
    / "sample_videos"
    / "match_clip.mp4"
)

DEFAULT_OUTPUT_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
    / "match_clip_tracking_results.csv"
)

DEFAULT_VIDEO_OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "videos"
)

CLASS_NAMES = {
    0: "player",
    1: "referee",
    2: "ball",
}


def resolve_path(path_value: Union[str, Path]) -> Path:
    path = Path(path_value)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run YOLO detection and ByteTrack multi-object "
            "tracking on a football video."
        )
    )

    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Path to the trained YOLO model weights.",
    )

    parser.add_argument(
        "--video",
        type=str,
        default=str(DEFAULT_VIDEO_PATH),
        help="Path to the input football video.",
    )

    parser.add_argument(
        "--output-csv",
        type=str,
        default=str(DEFAULT_OUTPUT_CSV),
        help="Path for the generated tracking CSV.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_VIDEO_OUTPUT_DIRECTORY),
        help="Directory for generated annotated video output.",
    )

    parser.add_argument(
        "--name",
        type=str,
        default="tracked_match_clip",
        help="Name of the Ultralytics output run.",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Detection confidence threshold.",
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.50,
        help="IoU threshold used by YOLO.",
    )

    parser.add_argument(
        "--tracker",
        type=str,
        default="bytetrack.yaml",
        help="Ultralytics tracker configuration.",
    )

    parser.add_argument(
        "--no-save-video",
        action="store_true",
        help="Do not save the annotated tracking video.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    model_path = resolve_path(args.model)
    video_path = resolve_path(args.video)
    output_csv = resolve_path(args.output_csv)
    output_directory = resolve_path(args.output_dir)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model weights not found: {model_path}"
        )

    if not video_path.exists():
        raise FileNotFoundError(
            f"Input video not found: {video_path}"
        )

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = YOLO(str(model_path))

    results = model.track(
        source=str(video_path),
        conf=args.confidence,
        iou=args.iou,
        tracker=args.tracker,
        persist=True,
        stream=True,
        save=not args.no_save_video,
        project=str(output_directory),
        name=args.name,
    )

    rows = []

    for frame_index, result in enumerate(results):
        boxes = result.boxes

        if boxes is None or boxes.id is None:
            continue

        for box, track_id, class_id, confidence in zip(
            boxes.xyxy.cpu().numpy(),
            boxes.id.cpu().numpy(),
            boxes.cls.cpu().numpy(),
            boxes.conf.cpu().numpy(),
        ):
            x1, y1, x2, y2 = box
            class_id = int(class_id)

            rows.append(
                {
                    "frame": frame_index,
                    "track_id": int(track_id),
                    "class_id": class_id,
                    "class_name": CLASS_NAMES.get(
                        class_id,
                        str(class_id),
                    ),
                    "confidence": float(confidence),
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "center_x": float((x1 + x2) / 2),
                    "center_y": float((y1 + y2) / 2),
                }
            )

    if not rows:
        raise RuntimeError(
            "Tracking completed without any tracked detections."
        )

    with output_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\n========== GAMEVISION AI TRACKING ==========")
    print(f"Model:              {model_path}")
    print(f"Video:              {video_path}")
    print(f"Confidence:         {args.confidence}")
    print(f"IoU threshold:      {args.iou}")
    print(f"Tracker:            {args.tracker}")
    print(f"Tracking CSV:       {output_csv}")
    print(f"Tracked detections: {len(rows)}")

    if args.no_save_video:
        print("Annotated video:    disabled")
    else:
        print(
            "Annotated video:    "
            f"{output_directory / args.name}"
        )


if __name__ == "__main__":
    main()