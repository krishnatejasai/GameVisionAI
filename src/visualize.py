from pathlib import Path
import random
import cv2
import matplotlib.pyplot as plt

IMAGE_DIR = Path("data/raw/football_players/players/images/train")
LABEL_DIR = Path("data/raw/football_players/players/labels/train")
OUTPUT_DIR = Path("outputs/plots/ground_truth_samples")

CLASS_NAMES = {
    0: "player",
    1: "referee",
    2: "ball",
}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

image_paths = [
    p for p in IMAGE_DIR.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
]

random.seed(42)
selected_images = random.sample(image_paths, min(6, len(image_paths)))

for image_path in selected_images:
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not read {image_path}")
        continue

    height, width = image.shape[:2]

    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    if label_path.exists():
        lines = [
            line.strip()
            for line in label_path.read_text().splitlines()
            if line.strip()
        ]

        for line in lines:
            class_id, x_center, y_center, box_width, box_height = map(
                float, line.split()
            )

            class_id = int(class_id)

            x_center *= width
            y_center *= height
            box_width *= width
            box_height *= height

            x1 = int(x_center - box_width / 2)
            y1 = int(y_center - box_height / 2)
            x2 = int(x_center + box_width / 2)
            y2 = int(y_center + box_height / 2)

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = CLASS_NAMES.get(class_id, str(class_id))

            cv2.putText(
                image,
                label,
                (x1, max(y1 - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

    output_path = OUTPUT_DIR / image_path.name
    cv2.imwrite(str(output_path), image)

    display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(14, 8))
    plt.imshow(display_image)
    plt.title(image_path.name)
    plt.axis("off")
    plt.show()

    print(f"Saved: {output_path}")