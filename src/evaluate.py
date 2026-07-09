import json
from pathlib import Path
import tempfile

import yaml
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "gamevision_yolov8s_960.pt"
DATA_CONFIG = PROJECT_ROOT / "configs" / "football_players.yaml"

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
    / "detection_metrics.json"
)

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "football_players"
    / "players"
).resolve()


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(
        "\n========== GAMEVISION AI DETECTION EVALUATION =========="
    )

    model = YOLO(str(MODEL_PATH))

    # Load the repository dataset configuration.
    with open(DATA_CONFIG, "r") as f:
        dataset_config = yaml.safe_load(f)

    # Resolve the dataset root dynamically so evaluation works
    # regardless of the machine-specific Ultralytics dataset directory.
    dataset_config["path"] = str(DATASET_ROOT)

    # Create a temporary resolved YAML configuration for Ultralytics.
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        delete=False,
    ) as temp_config:
        yaml.safe_dump(dataset_config, temp_config)
        resolved_config_path = temp_config.name

    metrics = model.val(
        data=resolved_config_path,
        split="val",
        imgsz=960,
        verbose=True,
    )

    class_names = model.names

    overall_metrics = {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
    }

    class_metrics = {}

    for class_id, class_name in class_names.items():
        result = metrics.box.class_result(class_id)

        class_metrics[class_name] = {
            "precision": float(result[0]),
            "recall": float(result[1]),
            "mAP50": float(result[2]),
            "mAP50_95": float(result[3]),
        }

    evaluation_results = {
        "model": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
        "dataset_config": str(
            DATA_CONFIG.relative_to(PROJECT_ROOT)
        ),
        "dataset_root": str(DATASET_ROOT),
        "validation_split": "val",
        "image_size": 960,
        "overall": overall_metrics,
        "class_wise": class_metrics,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(evaluation_results, f, indent=4)

    print("\n========== OVERALL RESULTS ==========")

    print(
        f"Precision:       "
        f"{overall_metrics['precision']:.4f}"
    )
    print(
        f"Recall:          "
        f"{overall_metrics['recall']:.4f}"
    )
    print(
        f"mAP@0.5:         "
        f"{overall_metrics['mAP50']:.4f}"
    )
    print(
        f"mAP@0.5:0.95:    "
        f"{overall_metrics['mAP50_95']:.4f}"
    )

    print("\n========== CLASS-WISE RESULTS ==========")

    for class_name, values in class_metrics.items():
        print(f"\n{class_name.upper()}")

        print(
            f"Precision:       "
            f"{values['precision']:.4f}"
        )
        print(
            f"Recall:          "
            f"{values['recall']:.4f}"
        )
        print(
            f"mAP@0.5:         "
            f"{values['mAP50']:.4f}"
        )
        print(
            f"mAP@0.5:0.95:    "
            f"{values['mAP50_95']:.4f}"
        )

    print(f"\nMetrics saved to: {OUTPUT_PATH}")
    print("Evaluation complete.")


if __name__ == "__main__":
    main()