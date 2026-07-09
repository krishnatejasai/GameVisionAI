import json
from pathlib import Path

import pandas as pd


TRACKING_CSV = Path("outputs/metrics/match_clip_tracking_results.csv")
OUTPUT_PATH = Path("outputs/metrics/track_fragmentation_metrics.json")


def summarize_track_coverage(df):
    results = {}

    total_frames = int(df["frame"].nunique())

    for class_name in sorted(df["class_name"].dropna().unique()):
        class_df = df[df["class_name"] == class_name]

        track_stats = []

        for track_id, group in class_df.groupby("track_id"):
            frames = sorted(group["frame"].unique())

            first_frame = min(frames)
            last_frame = max(frames)
            observed_frames = len(frames)
            frame_span = last_frame - first_frame + 1
            temporal_coverage = observed_frames / frame_span

            missing_frames_inside_track = frame_span - observed_frames

            track_stats.append({
                "track_id": int(track_id),
                "first_frame": int(first_frame),
                "last_frame": int(last_frame),
                "observed_frames": int(observed_frames),
                "frame_span": int(frame_span),
                "temporal_coverage": float(temporal_coverage),
                "missing_frames_inside_track": int(missing_frames_inside_track),
            })

        track_df = pd.DataFrame(track_stats)

        results[class_name] = {
            "total_frames_in_clip": total_frames,
            "unique_track_ids": int(class_df["track_id"].nunique()),
            "mean_temporal_coverage": float(track_df["temporal_coverage"].mean()),
            "median_temporal_coverage": float(track_df["temporal_coverage"].median()),
            "mean_missing_frames_inside_track": float(
                track_df["missing_frames_inside_track"].mean()
            ),
            "longest_frame_span": int(track_df["frame_span"].max()),
            "longest_observed_track": int(track_df["observed_frames"].max()),
        }

    return results


def main():
    if not TRACKING_CSV.exists():
        raise FileNotFoundError(f"Tracking CSV not found: {TRACKING_CSV}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(TRACKING_CSV)

    results = summarize_track_coverage(df)

    summary = {
        "source_csv": str(TRACKING_CSV),
        "note": (
            "Track fragmentation and temporal coverage are computed from "
            "predicted ByteTrack outputs, not ground-truth MOT annotations."
        ),
        "class_wise": results,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(summary, f, indent=4)

    print("\n========== TRACK FRAGMENTATION ANALYSIS ==========")

    for class_name, values in results.items():
        print(f"\n{class_name.upper()}")
        print(f"Unique track IDs:              {values['unique_track_ids']}")
        print(
            "Mean temporal coverage:        "
            f"{values['mean_temporal_coverage']:.3f}"
        )
        print(
            "Median temporal coverage:      "
            f"{values['median_temporal_coverage']:.3f}"
        )
        print(
            "Mean missing frames/track:     "
            f"{values['mean_missing_frames_inside_track']:.2f}"
        )
        print(f"Longest frame span:            {values['longest_frame_span']}")
        print(f"Longest observed track:        {values['longest_observed_track']}")

    print(f"\nResults saved to: {OUTPUT_PATH}")
    print("Track fragmentation analysis complete.")


if __name__ == "__main__":
    main()