from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = Path("outputs/metrics/match_clip_tracking_results.csv")
OUTPUT_METRICS = Path("outputs/metrics/match_clip_track_summary.csv")
OUTPUT_DIR = Path("outputs/plots/match_clip_analytics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_PATH)

def compute_distance(group):
    group = group.sort_values("frame")
    dx = group["center_x"].diff()
    dy = group["center_y"].diff()
    distances = np.sqrt(dx**2 + dy**2).fillna(0)
    return distances.sum()

summary = df.groupby(["track_id", "class_name"]).apply(
    lambda g: pd.Series({
        "frames_seen": g["frame"].nunique(),
        "start_frame": g["frame"].min(),
        "end_frame": g["frame"].max(),
        "avg_confidence": g["confidence"].mean(),
        "distance_pixels": compute_distance(g),
        "avg_speed_pixels_per_frame": compute_distance(g) / max(g["frame"].nunique() - 1, 1),
        "avg_x": g["center_x"].mean(),
        "avg_y": g["center_y"].mean(),
    })
).reset_index()

summary.to_csv(OUTPUT_METRICS, index=False)

print("Track summary saved to:", OUTPUT_METRICS)
print(summary)

players = df[df["class_name"] == "player"]

plt.figure(figsize=(12, 7))
for track_id, group in players.groupby("track_id"):
    group = group.sort_values("frame")
    plt.plot(group["center_x"], group["center_y"], marker="o", linewidth=1, label=f"ID {track_id}")

plt.gca().invert_yaxis()
plt.title("Player Trajectories")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.legend(fontsize=8)
plt.savefig(OUTPUT_DIR / "player_trajectories.png", dpi=200)
plt.show()

plt.figure(figsize=(10, 6))
plt.hist2d(players["center_x"], players["center_y"], bins=30)
plt.gca().invert_yaxis()
plt.title("Player Occupancy Heatmap")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.colorbar(label="Frequency")
plt.savefig(OUTPUT_DIR / "player_occupancy_heatmap.png", dpi=200)
plt.show()

ball = df[df["class_name"] == "ball"]

if not ball.empty:
    plt.figure(figsize=(12, 7))
    for track_id, group in ball.groupby("track_id"):
        group = group.sort_values("frame")
        plt.plot(group["center_x"], group["center_y"], marker="o", linewidth=2, label=f"Ball ID {track_id}")

    plt.gca().invert_yaxis()
    plt.title("Ball Trajectory")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.savefig(OUTPUT_DIR / "ball_trajectory.png", dpi=200)
    plt.show()
import numpy as np

# -----------------------------
# PROJECT-LEVEL METRICS
# -----------------------------

total_detections = len(df)
total_frames = df["frame"].nunique()

player_df = df[df["class_name"] == "player"].copy()

unique_player_tracks = player_df["track_id"].nunique()
avg_players_per_frame = (
    player_df.groupby("frame").size().mean()
    if not player_df.empty else 0
)

track_metrics = []

for track_id, group in player_df.groupby("track_id"):
    group = group.sort_values("frame").copy()

    dx = group["center_x"].diff()
    dy = group["center_y"].diff()

    step_distance = np.sqrt(dx**2 + dy**2)
    total_distance = step_distance.sum()

    frame_span = group["frame"].max() - group["frame"].min() + 1
    avg_speed = (
        total_distance / (frame_span - 1)
        if frame_span > 1 else 0
    )

    track_metrics.append({
        "track_id": track_id,
        "detections": len(group),
        "first_frame": group["frame"].min(),
        "last_frame": group["frame"].max(),
        "track_duration_frames": frame_span,
        "distance_pixels": total_distance,
        "avg_speed_pixels_per_frame": avg_speed,
    })

track_metrics_df = pd.DataFrame(track_metrics)

detailed_metrics_path = Path(
    "outputs/metrics/match_clip_player_metrics.csv"
)

track_metrics_df.to_csv(detailed_metrics_path, index=False)

if not track_metrics_df.empty:
    longest_track = int(
        track_metrics_df["track_duration_frames"].max()
    )

    max_distance = float(
        track_metrics_df["distance_pixels"].max()
    )

    mean_track_speed = float(
        track_metrics_df["avg_speed_pixels_per_frame"].mean()
    )
else:
    longest_track = 0
    max_distance = 0
    mean_track_speed = 0

print("\n========== GAMEVISION METRICS ==========")
print(f"Total tracked detections: {total_detections}")
print(f"Frames analyzed: {total_frames}")
print(f"Unique player track IDs: {unique_player_tracks}")
print(f"Average players detected per frame: {avg_players_per_frame:.2f}")
print(f"Longest player track: {longest_track} frames")
print(f"Maximum tracked distance: {max_distance:.2f} pixels")
print(f"Mean track speed: {mean_track_speed:.2f} pixels/frame")
print("========================================")
print(f"Detailed metrics saved to: {detailed_metrics_path}")
print("Analytics complete.")