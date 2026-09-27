from pathlib import Path
import random

import matplotlib.pyplot as plt
from PIL import Image


# ==============================
# Dataset location
# ==============================
DATASET_PATH = Path("data/raw/Dataset/Data")


# ==============================
# Which classes to inspect, and how many images per class.
# Grouped so related classes sit next to each other in the grid -
# makes it easy to eyeball "do these actually look different?"
# ==============================

TARGET_CLASSES = [
    # the se7a/car mix-up isn't a semantic minimal pair - check for
    # a labeling or extraction mistake
    "se7a",
    "car",

    # scattered ("magnet") confusions - check if samples within the
    # class itself look consistent, or if some are noisy/bad detections
    "radio",
    "siye7a",

    # genuine minimal pairs - check if these are actually
    # distinguishable by hand shape alone, or only by hand position
    "o5t",
    "5ou",
    "jad",
    "jadda",
    "bent",
    "eben",
]

SAMPLES_PER_CLASS = 4


# ==============================
# Check dataset path
# ==============================
if not DATASET_PATH.exists():
    print("ERROR: Dataset folder was not found.")
    print("Expected location:", DATASET_PATH)
    exit()


# ==============================
# Find all class folders, keyed by class name
# ==============================
class_dirs_by_name = {}

for category in DATASET_PATH.iterdir():

    if category.is_dir():

        for sign_class in category.iterdir():

            if sign_class.is_dir():
                class_dirs_by_name[sign_class.name] = sign_class


print("===== TuniSign Targeted Sample Viewer =====")
print("Number of classes found in dataset:", len(class_dirs_by_name))
print()


# ==============================
# Collect samples for each requested class
# ==============================
samples = []

for class_name in TARGET_CLASSES:

    class_dir = class_dirs_by_name.get(class_name)

    if class_dir is None:
        print(f"WARNING: class '{class_name}' not found in dataset, skipping.")
        continue

    image_files = [
        file
        for file in class_dir.iterdir()
        if file.is_file()
        and file.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]

    if not image_files:
        print(f"WARNING: class '{class_name}' has no images, skipping.")
        continue

    n_to_take = min(SAMPLES_PER_CLASS, len(image_files))
    chosen = random.sample(image_files, n_to_take)

    for image_path in chosen:
        samples.append((class_name, image_path))


# ==============================
# Display images
# ==============================
n_rows = len(TARGET_CLASSES)
n_cols = SAMPLES_PER_CLASS

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))

# Make axes always indexable as [row][col], even if n_rows == 1
if n_rows == 1:
    axes = [axes]


# Group the collected samples back by class, in the requested order
samples_by_class = {name: [] for name in TARGET_CLASSES}

for class_name, image_path in samples:
    samples_by_class[class_name].append(image_path)


for row, class_name in enumerate(TARGET_CLASSES):

    row_images = samples_by_class.get(class_name, [])

    for col in range(n_cols):

        ax = axes[row][col]

        if col < len(row_images):

            image = Image.open(row_images[col])
            ax.imshow(image)

        if col == 0:
            ax.set_ylabel(class_name, fontsize=12, rotation=0, labelpad=40)

        ax.set_xticks([])
        ax.set_yticks([])


plt.tight_layout()

plt.show()