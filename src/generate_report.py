import json
from pathlib import Path
from datetime import datetime


METRICS_DIR = Path("outputs/metrics")

DETECTION_PATH = METRICS_DIR / "detection_metrics.json"
RUNTIME_PATH = METRICS_DIR / "runtime_benchmark.json"
CONTINUITY_PATH = METRICS_DIR / "tracking_continuity_metrics.json"
FRAGMENTATION_PATH = METRICS_DIR / "track_fragmentation_metrics.json"

OUTPUT_JSON = METRICS_DIR / "evaluation_summary.json"
OUTPUT_MD = METRICS_DIR / "evaluation_summary.md"


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"Required metrics file not found: {path}")

    with open(path, "r") as f:
        return json.load(f)


def main():
    detection = load_json(DETECTION_PATH)
    runtime = load_json(RUNTIME_PATH)
    continuity = load_json(CONTINUITY_PATH)
    fragmentation = load_json(FRAGMENTATION_PATH)

    report = {
        "project": "GameVision AI",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "evaluation_scope": {
            "detection": (
                "YOLO validation metrics computed on the held-out validation split."
            ),
            "tracking": (
                "Descriptive statistics computed from predicted ByteTrack trajectories. "
                "These are not ground-truth MOT metrics such as MOTA, HOTA, or IDF1."
            ),
            "runtime": (
                "End-to-end detector inference benchmark on the sample match clip "
                "using the local execution environment."
            ),
        },
        "detection": detection,
        "runtime": runtime,
        "tracking_continuity": continuity,
        "track_fragmentation": fragmentation,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(report, f, indent=4)

    overall = detection["overall"]
    class_wise = detection["class_wise"]

    runtime_fps = runtime["effective_fps"]
    latency = runtime["average_latency_ms_per_frame"]

    frag = fragmentation["class_wise"]

    markdown = f"""# GameVision AI Evaluation Summary

## Evaluation Methodology

GameVision AI was evaluated across three layers:

1. **Object detection** using a held-out validation split.
2. **Runtime performance** on the sample football match clip.
3. **Tracking behavior** using descriptive continuity and temporal coverage statistics derived from predicted ByteTrack trajectories.

Tracking statistics are descriptive metrics computed from predicted tracks. They are not ground-truth MOT metrics such as MOTA, HOTA, or IDF1.

---

## Overall Detection Performance

| Metric | Value |
|---|---:|
| Precision | {overall['precision']:.4f} |
| Recall | {overall['recall']:.4f} |
| mAP@0.5 | {overall['mAP50']:.4f} |
| mAP@0.5:0.95 | {overall['mAP50_95']:.4f} |

---

## Class-wise Detection Performance

| Class | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|
| Player | {class_wise['player']['precision']:.4f} | {class_wise['player']['recall']:.4f} | {class_wise['player']['mAP50']:.4f} | {class_wise['player']['mAP50_95']:.4f} |
| Referee | {class_wise['referee']['precision']:.4f} | {class_wise['referee']['recall']:.4f} | {class_wise['referee']['mAP50']:.4f} | {class_wise['referee']['mAP50_95']:.4f} |
| Ball | {class_wise['ball']['precision']:.4f} | {class_wise['ball']['recall']:.4f} | {class_wise['ball']['mAP50']:.4f} | {class_wise['ball']['mAP50_95']:.4f} |

---

## Runtime Benchmark

| Metric | Value |
|---|---:|
| Effective throughput | {runtime_fps:.2f} FPS |
| Average latency | {latency:.2f} ms/frame |
| Frames processed | {runtime['processed_frames']} |
| Source FPS | {runtime['source_fps']:.2f} |

The runtime benchmark measures detector inference performance in the local execution environment. Runtime results are hardware-dependent and should not be interpreted as universal deployment performance.

---

## Track Temporal Coverage

| Class | Unique Track IDs | Mean Coverage | Median Coverage | Mean Missing Frames | Longest Observed Track |
|---|---:|---:|---:|---:|---:|
| Player | {frag['player']['unique_track_ids']} | {frag['player']['mean_temporal_coverage']:.3f} | {frag['player']['median_temporal_coverage']:.3f} | {frag['player']['mean_missing_frames_inside_track']:.2f} | {frag['player']['longest_observed_track']} |
| Referee | {frag['referee']['unique_track_ids']} | {frag['referee']['mean_temporal_coverage']:.3f} | {frag['referee']['median_temporal_coverage']:.3f} | {frag['referee']['mean_missing_frames_inside_track']:.2f} | {frag['referee']['longest_observed_track']} |
| Ball | {frag['ball']['unique_track_ids']} | {frag['ball']['mean_temporal_coverage']:.3f} | {frag['ball']['median_temporal_coverage']:.3f} | {frag['ball']['mean_missing_frames_inside_track']:.2f} | {frag['ball']['longest_observed_track']} |

---

## Interpretation

The detector performs strongly on the player and referee classes, while ball detection remains the primary small-object detection challenge.

Player trajectories demonstrate strong temporal continuity across the evaluated clip. Referee trajectories exhibit greater fragmentation and more missing observations within track spans.

Ball tracks show high within-track temporal coverage but are distributed across multiple track IDs. This indicates that once the ball is successfully associated with a track, short-term continuity can be strong, while detection gaps and rapid motion can still cause identity fragmentation.

---

## Engineering Observations

- Player detection is the strongest component of the detector, with high precision, recall, and mAP.
- Referee detection is also strong on the held-out validation set, although tracking continuity is weaker than for players.
- Ball detection has substantially lower recall than player and referee detection, reflecting the difficulty of detecting small, fast-moving objects in compressed broadcast footage.
- ByteTrack provides persistent identities across frames, enabling trajectory generation and per-track motion analysis.
- Temporal coverage metrics provide descriptive insight into predicted track continuity without claiming unsupported ground-truth MOT accuracy.

---

## Limitations

- Tracking statistics are derived from predicted trajectories rather than ground-truth MOT annotations.
- Therefore, MOTA, HOTA, and IDF1 are not reported.
- Runtime measurements are hardware-dependent.
- Ball detection performance is constrained by small object size, motion blur, occlusion, and broadcast-video compression.
- Image-coordinate movement statistics are affected by camera motion and perspective.
- Pixel displacement and pixel speed should not be interpreted as real-world physical distance or velocity.
- Evaluation results are specific to the current held-out validation split and sample match footage.

---

## Generated Artifacts

This evaluation pipeline generates:

- Detection metrics in JSON format
- Runtime benchmark results
- Tracking continuity statistics
- Track fragmentation and temporal coverage statistics
- Consolidated JSON evaluation report
- Consolidated Markdown evaluation report
- Player trajectory visualization
- Player occupancy heatmap
- Ball trajectory visualization

These artifacts provide both quantitative and qualitative evidence for evaluating the GameVision AI pipeline.
"""

    with open(OUTPUT_MD, "w") as f:
        f.write(markdown)

    print("\n========== GAMEVISION AI CONSOLIDATED REPORT ==========")

    print("\nDETECTION")
    print(f"Overall Precision:      {overall['precision']:.4f}")
    print(f"Overall Recall:         {overall['recall']:.4f}")
    print(f"Overall mAP@0.5:        {overall['mAP50']:.4f}")
    print(f"Overall mAP@0.5:0.95:   {overall['mAP50_95']:.4f}")

    print("\nRUNTIME")
    print(f"Effective throughput:   {runtime_fps:.2f} FPS")
    print(f"Average latency:        {latency:.2f} ms/frame")
    print(f"Frames processed:       {runtime['processed_frames']}")

    print("\nTRACK COVERAGE")

    for class_name in ["player", "referee", "ball"]:
        values = frag[class_name]

        print(f"\n{class_name.upper()}")
        print(
            f"Mean temporal coverage:   "
            f"{values['mean_temporal_coverage']:.3f}"
        )
        print(
            f"Median temporal coverage: "
            f"{values['median_temporal_coverage']:.3f}"
        )
        print(
            f"Unique track IDs:         "
            f"{values['unique_track_ids']}"
        )
        print(
            f"Longest observed track:   "
            f"{values['longest_observed_track']} frames"
        )

    print("\n======================================================")
    print(f"JSON report:     {OUTPUT_JSON}")
    print(f"Markdown report: {OUTPUT_MD}")
    print("Consolidated evaluation report generated successfully.")


if __name__ == "__main__":
    main()