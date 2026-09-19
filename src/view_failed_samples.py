from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


FAILED_FILE = Path("data/processed/mediapipe_failed_images.csv")

SAMPLES_PER_CLASS = 8

df = pd.read_csv(FAILED_FILE)

classes_to_check = [
    "métro",
    "karhba",
    "bousta",
    "jadda",
    "5adamet",
    "bent",
    "mostawsaf"
]

for class_name in classes_to_check:

    class_df = df[df["class"] == class_name]

    if len(class_df) == 0:
        continue

    samples = class_df.sample(
        n=min(SAMPLES_PER_CLASS, len(class_df)),
        random_state=42
    )

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()

    for ax in axes:
        ax.axis("off")

    for ax, (_, row) in zip(axes, samples.iterrows()):

        image_path = Path(row["image_path"])

        try:
            image = Image.open(image_path)
            ax.imshow(image)
            ax.set_title(class_name)

        except Exception:
            ax.set_title("Could not open")

        ax.axis("off")

    plt.suptitle(
        f"Failed MediaPipe detections — {class_name}"
    )

    plt.tight_layout()
    plt.show()