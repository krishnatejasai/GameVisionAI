from pathlib import Path
from collections import Counter
import json

DATASET_ROOT = Path("data/raw/football_players/players")

splits = ["train", "val"]
summary = {}

for split in splits:
    image_dir = DATASET_ROOT / "images" / split
    label_dir = DATASET_ROOT / "labels" / split

    image_files = list(image_dir.glob("*"))
    label_files = list(label_dir.glob("*.txt"))

    class_counts = Counter()
    object_counts_per_image = []
    empty_labels = []

    for label_file in label_files:
        lines = [
            line.strip()
            for line in label_file.read_text().splitlines()
            if line.strip()
        ]

        object_counts_per_image.append(len(lines))

        if len(lines) == 0:
            empty_labels.append(label_file.name)

        for line in lines:
            parts = line.split()
            class_id = int(parts[0])
            class_counts[class_id] += 1

    total_objects = sum(class_counts.values())

    summary[split] = {
        "num_images": len(image_files),
        "num_label_files": len(label_files),
        "total_objects": total_objects,
        "class_distribution": dict(sorted(class_counts.items())),
        "empty_label_files": empty_labels,
        "average_objects_per_image": (
            total_objects / len(image_files) if image_files else 0
        ),
        "min_objects_per_image": (
            min(object_counts_per_image) if object_counts_per_image else 0
        ),
        "max_objects_per_image": (
            max(object_counts_per_image) if object_counts_per_image else 0
        ),
    }

output_path = Path("outputs/metrics/dataset_summary.json")
output_path.parent.mkdir(parents=True, exist_ok=True)

with output_path.open("w") as f:
    json.dump(summary, f, indent=4)

print(json.dumps(summary, indent=4))
print(f"\nSaved dataset summary to: {output_path}")