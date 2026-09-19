from pathlib import Path
import random

import matplotlib.pyplot as plt
from PIL import Image


# ==============================
# Dataset location
# ==============================
DATASET_PATH = Path("data/raw/Dataset/Data")


# ==============================
# Check dataset path
# ==============================
if not DATASET_PATH.exists():
    print("ERROR: Dataset folder was not found.")
    print("Expected location:", DATASET_PATH)
    exit()


# ==============================
# Find all class folders
# ==============================
class_dirs = []

for category in DATASET_PATH.iterdir():

    if category.is_dir():

        for sign_class in category.iterdir():

            if sign_class.is_dir():
                class_dirs.append(sign_class)


print("===== TuniSign Sample Viewer =====")
print("Number of classes found:", len(class_dirs))
print()


# ==============================
# Choose random classes
# ==============================
sample_size = min(12, len(class_dirs))

selected_classes = random.sample(class_dirs, sample_size)


# ==============================
# Choose one random image
# from each selected class
# ==============================
samples = []

for class_dir in selected_classes:

    image_files = [
        file
        for file in class_dir.iterdir()
        if file.is_file()
        and file.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]

    if image_files:

        selected_image = random.choice(image_files)

        samples.append(selected_image)


# ==============================
# Display images
# ==============================
fig, axes = plt.subplots(3, 4, figsize=(12, 9))

axes = axes.flatten()


for ax, image_path in zip(axes, samples):

    image = Image.open(image_path)

    ax.imshow(image)

    category_name = image_path.parent.parent.name
    class_name = image_path.parent.name

    ax.set_title(
        f"{category_name} / {class_name}",
        fontsize=10
    )

    ax.axis("off")


# Hide unused plots
for ax in axes[len(samples):]:

    ax.axis("off")


plt.tight_layout()

plt.show()