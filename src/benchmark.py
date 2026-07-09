import json
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


MODEL_PATH = "models/gamevision_yolov8s_960.pt"
VIDEO_PATH = "data/sample_videos/match_clip.mp4"
OUTPUT_PATH = Path("outputs/metrics/runtime_benchmark.json")
IMAGE_SIZE = 960


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {VIDEO_PATH}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    processed_frames = 0
    total_inference_time = 0.0

    print("\n========== GAMEVISION AI RUNTIME BENCHMARK ==========")
    print(f"Video: {VIDEO_PATH}")
    print(f"Resolution: {width}x{height}")
    print(f"Source FPS: {source_fps:.2f}")
    print(f"Total frames: {total_frames}")

    while True:
        success, frame = cap.read()

        if not success:
            break

        start = time.perf_counter()

        model.predict(
            source=frame,
            imgsz=IMAGE_SIZE,
            verbose=False,
        )

        elapsed = time.perf_counter() - start

        total_inference_time += elapsed
        processed_frames += 1

    cap.release()

    average_latency = total_inference_time / processed_frames
    effective_fps = processed_frames / total_inference_time

    results = {
        "model": MODEL_PATH,
        "video": VIDEO_PATH,
        "input_image_size": IMAGE_SIZE,
        "video_resolution": {
            "width": width,
            "height": height,
        },
        "source_fps": source_fps,
        "processed_frames": processed_frames,
        "total_processing_time_seconds": total_inference_time,
        "average_latency_ms_per_frame": average_latency * 1000,
        "effective_fps": effective_fps,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=4)

    print("\n========== BENCHMARK RESULTS ==========")
    print(f"Frames processed:       {processed_frames}")
    print(f"Processing time:        {total_inference_time:.2f} s")
    print(f"Average latency:        {average_latency * 1000:.2f} ms/frame")
    print(f"Effective throughput:   {effective_fps:.2f} FPS")
    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()