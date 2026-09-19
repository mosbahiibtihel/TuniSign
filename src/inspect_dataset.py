from pathlib import Path


# ==============================
# 1. Dataset location
# ==============================

DATASET_PATH = Path("data/raw/Dataset/Data")


# ==============================
# 2. Check that the folder exists
# ==============================

if not DATASET_PATH.exists():
    print("ERROR: Dataset folder was not found.")
    print("Expected location:", DATASET_PATH)
    exit()


print("===== TuniSign Dataset Inspection =====")
print("Dataset path:", DATASET_PATH)
print()


# ==============================
# 3. Find the 5 main categories
# ==============================

categories = [
    folder for folder in DATASET_PATH.iterdir()
    if folder.is_dir()
]

print("Number of main categories:", len(categories))
print()


# ==============================
# 4. Find sign classes
# ==============================

total_images = 0
total_classes = 0

for category in sorted(categories):

    print(f"--- {category.name} ---")

    # Each folder inside the category represents one sign/class
    classes = [
        folder for folder in category.iterdir()
        if folder.is_dir()
    ]

    for sign_class in sorted(classes):

        images = [
            file for file in sign_class.iterdir()
            if file.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ]

        print(f"{sign_class.name}: {len(images)} images")

        total_images += len(images)
        total_classes += 1

    print()


# ==============================
# 5. Final statistics
# ==============================

print("===== Final Statistics =====")
print("Number of main categories:", len(categories))
print("Number of sign classes:", total_classes)
print("Total images:", total_images)