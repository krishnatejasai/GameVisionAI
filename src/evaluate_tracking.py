import json
from pathlib import Path

import pandas as pd


TRACKING_CSV = Path(
    "outputs/metrics/match_clip_tracking_results.csv"
)

OUTPUT_PATH = Path(
    "outputs/metrics/tracking_continuity_metrics.json"
)


def main():
    if not TRACKING_CSV.exists():
        raise FileNotFoundError(
            f"Tracking CSV not found: {TRACKING_CSV}"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(TRACKING_CSV)

    required_columns = {
        "frame",
        "track_id",
        "class_name",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    print("\n========== GAMEVISION AI TRACKING ANALYSIS ==========")

    results = {}

    for class_name in sorted(df["class_name"].dropna().unique()):
        class_df = df[df["class_name"] == class_name]

        track_lengths = (
            class_df.groupby("track_id")["frame"]
            .nunique()
            .sort_values(ascending=False)
        )

        class_result = {
            "detections": int(len(class_df)),
            "unique_track_ids": int(class_df["track_id"].nunique()),
            "mean_track_length_frames": float(track_lengths.mean()),
            "median_track_length_frames": float(track_lengths.median()),
            "longest_track_frames": int(track_lengths.max()),
            "tracks_at_least_10_frames": int(
                (track_lengths >= 10).sum()
            ),
            "tracks_at_least_30_frames": int(
                (track_lengths >= 30).sum()
            ),
        }

        results[class_name] = class_result

        print(f"\n{class_name.upper()}")
        print(f"Detections:                 {class_result['detections']}")
        print(f"Unique track IDs:           {class_result['unique_track_ids']}")
        print(
            "Mean track length:          "
            f"{class_result['mean_track_length_frames']:.2f} frames"
        )
        print(
            "Median track length:        "
            f"{class_result['median_track_length_frames']:.2f} frames"
        )
        print(
            "Longest track:              "
            f"{class_result['longest_track_frames']} frames"
        )
        print(
            "Tracks >= 10 frames:        "
            f"{class_result['tracks_at_least_10_frames']}"
        )
        print(
            "Tracks >= 30 frames:        "
            f"{class_result['tracks_at_least_30_frames']}"
        )

    summary = {
        "source_csv": str(TRACKING_CSV),
        "note": (
            "Descriptive continuity statistics computed from predicted "
            "tracks. These are not ground-truth MOT metrics."
        ),
        "class_wise": results,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"\nResults saved to: {OUTPUT_PATH}")
    print("Tracking analysis complete.")


if __name__ == "__main__":
    main()