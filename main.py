import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def run_script(script_name: str, extra_args: list[str]) -> None:
    script_path = PROJECT_ROOT / "src" / script_name

    if not script_path.exists():
        raise FileNotFoundError(
            f"Required script not found: {script_path}"
        )

    command = [
        sys.executable,
        str(script_path),
        *extra_args,
    ]

    subprocess.run(
        command,
        check=True,
        cwd=PROJECT_ROOT,
    )


def run_evaluation_pipeline() -> None:
    script_path = PROJECT_ROOT / "run_evaluation.py"

    if not script_path.exists():
        raise FileNotFoundError(
            f"Required script not found: {script_path}"
        )

    subprocess.run(
        [sys.executable, str(script_path)],
        check=True,
        cwd=PROJECT_ROOT,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Command-line interface for the GameVision AI "
            "football video analytics pipeline."
        )
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    track_parser = subparsers.add_parser(
        "track",
        help="Run object detection and multi-object tracking.",
    )
    track_parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Additional arguments passed to src/track.py.",
    )

    analytics_parser = subparsers.add_parser(
        "analytics",
        help="Generate football tracking analytics.",
    )
    analytics_parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Additional arguments passed to src/analytics.py.",
    )

    detect_parser = subparsers.add_parser(
        "detect",
        help="Run football object detection.",
    )
    detect_parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Additional arguments passed to src/detect.py.",
    )

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate the trained detection model.",
    )
    evaluate_parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Additional arguments passed to src/evaluate.py.",
    )

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Benchmark model inference performance.",
    )
    benchmark_parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Additional arguments passed to src/benchmark.py.",
    )

    tracking_analysis_parser = subparsers.add_parser(
        "tracking-analysis",
        help="Evaluate tracking continuity.",
    )
    tracking_analysis_parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Additional arguments passed to src/evaluate_tracking.py.",
    )

    fragmentation_parser = subparsers.add_parser(
        "fragmentation",
        help="Analyze track fragmentation and temporal coverage.",
    )
    fragmentation_parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Additional arguments passed to src/analyze_tracks.py.",
    )

    report_parser = subparsers.add_parser(
        "report",
        help="Generate the consolidated evaluation report.",
    )
    report_parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Additional arguments passed to src/generate_report.py.",
    )

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Inspect the football-player dataset.",
    )
    inspect_parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Additional arguments passed to src/inspect_dataset.py.",
    )

    subparsers.add_parser(
        "full-evaluation",
        help="Run the complete evaluation and reporting pipeline.",
    )

    args = parser.parse_args()

    command_to_script = {
        "track": "track.py",
        "analytics": "analytics.py",
        "detect": "detect.py",
        "evaluate": "evaluate.py",
        "benchmark": "benchmark.py",
        "tracking-analysis": "evaluate_tracking.py",
        "fragmentation": "analyze_tracks.py",
        "report": "generate_report.py",
        "inspect": "inspect_dataset.py",
    }

    if args.command == "full-evaluation":
        run_evaluation_pipeline()
        return

    run_script(
        script_name=command_to_script[args.command],
        extra_args=args.args,
    )


if __name__ == "__main__":
    main()