import subprocess
import sys
from pathlib import Path


STEPS = [
    ("Detection evaluation", "src/evaluate.py"),
    ("Runtime benchmark", "src/benchmark.py"),
    ("Tracking continuity analysis", "src/evaluate_tracking.py"),
    ("Track fragmentation analysis", "src/analyze_tracks.py"),
    ("Consolidated report generation", "src/generate_report.py"),
]


def run_step(name, script):
    script_path = Path(script)

    if not script_path.exists():
        raise FileNotFoundError(f"Required script not found: {script}")

    print("\n" + "=" * 60)
    print(f"RUNNING: {name}")
    print("=" * 60)

    subprocess.run(
        [sys.executable, str(script_path)],
        check=True,
    )

    print(f"COMPLETED: {name}")


def main():
    print("\n" + "=" * 60)
    print("GAMEVISION AI — FULL EVALUATION PIPELINE")
    print("=" * 60)

    for name, script in STEPS:
        run_step(name, script)

    print("\n" + "=" * 60)
    print("EVALUATION PIPELINE COMPLETE")
    print("=" * 60)

    print("\nGenerated reports:")
    print("  outputs/metrics/detection_metrics.json")
    print("  outputs/metrics/runtime_benchmark.json")
    print("  outputs/metrics/tracking_continuity_metrics.json")
    print("  outputs/metrics/track_fragmentation_metrics.json")
    print("  outputs/metrics/evaluation_summary.json")
    print("  outputs/metrics/evaluation_summary.md")


if __name__ == "__main__":
    main()