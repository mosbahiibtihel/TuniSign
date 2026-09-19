from pathlib import Path

import cv2
import mediapipe as mp
import pandas as pd


# ============================================
# Paths
# ============================================

DATASET_PATH = Path("data/raw/Dataset/Data")
RESULTS_DIR = Path("data/processed")


# ============================================
# Check dataset
# ============================================

if not DATASET_PATH.exists():
    print("ERROR: Dataset folder was not found.")
    print("Expected:", DATASET_PATH)
    exit()


if not Path("models/hand_landmarker.task").exists():
    print("ERROR: MediaPipe model was not found.")
    print("Expected: models/hand_landmarker.task")
    exit()


# ============================================
# Create results directory
# ============================================

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================
# Find all images
# ============================================

image_files = [
    file
    for file in DATASET_PATH.rglob("*")
    if file.is_file()
    and file.suffix.lower() in [".jpg", ".jpeg", ".png"]
]


print("============================================")
print("     TuniSign MediaPipe Quality Check")
print("============================================")
print()

print("MediaPipe version:", mp.__version__)
print("Total images:", len(image_files))
print()


# ============================================
# MediaPipe configuration
# ============================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="models/hand_landmarker.task"
    ),
    running_mode=RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3
)


# ============================================
# Store results
# ============================================

results = []


# ============================================
# Process all images
# ============================================

with HandLandmarker.create_from_options(options) as landmarker:

    for index, image_path in enumerate(image_files, start=1):

        # ----------------------------------------
        # Determine category and class
        # ----------------------------------------

        sign_class = image_path.parent.name
        category = image_path.parent.parent.name


        # ----------------------------------------
        # Read image
        # ----------------------------------------

        image = cv2.imread(str(image_path))


        if image is None:

            results.append({
                "image_path": str(image_path),
                "category": category,
                "class": sign_class,
                "hand_detected": False,
                "status": "read_error"
            })

            continue


        # ----------------------------------------
        # Convert BGR → RGB
        # ----------------------------------------

        image_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )


        # ----------------------------------------
        # Create MediaPipe image
        # ----------------------------------------

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=image_rgb
        )


        # ----------------------------------------
        # Detect hand
        # ----------------------------------------

        detection_result = landmarker.detect(mp_image)


        if detection_result.hand_landmarks:

            hand_detected = True
            status = "detected"

        else:

            hand_detected = False
            status = "failed"


        # ----------------------------------------
        # Save result
        # ----------------------------------------

        results.append({
            "image_path": str(image_path),
            "category": category,
            "class": sign_class,
            "hand_detected": hand_detected,
            "status": status
        })


        # ----------------------------------------
        # Progress
        # ----------------------------------------

        if index % 100 == 0 or index == len(image_files):

            print(
                f"Processed {index}/{len(image_files)} images..."
            )


# ============================================
# Create DataFrame
# ============================================

df = pd.DataFrame(results)


# ============================================
# Save complete results
# ============================================

all_results_path = (
    RESULTS_DIR / "mediapipe_detection_results.csv"
)

df.to_csv(
    all_results_path,
    index=False
)


# ============================================
# Failed images
# ============================================

failed_df = df[
    df["hand_detected"] == False
]


failed_results_path = (
    RESULTS_DIR / "mediapipe_failed_images.csv"
)

failed_df.to_csv(
    failed_results_path,
    index=False
)


# ============================================
# Overall statistics
# ============================================

total = len(df)

detected = df["hand_detected"].sum()

failed = total - detected

read_errors = len(
    df[df["status"] == "read_error"]
)


if total > 0:

    detection_rate = (
        detected / total
    ) * 100

    failure_rate = (
        failed / total
    ) * 100

else:

    detection_rate = 0
    failure_rate = 0


# ============================================
# Per-class statistics
# ============================================

class_statistics = (
    df
    .groupby(["category", "class"])
    .agg(
        images=("image_path", "count"),
        detected=("hand_detected", "sum")
    )
    .reset_index()
)


class_statistics["failed"] = (
    class_statistics["images"]
    - class_statistics["detected"]
)


class_statistics["detection_rate"] = (
    class_statistics["detected"]
    / class_statistics["images"]
    * 100
)


class_statistics = class_statistics.sort_values(
    "detection_rate"
)


class_results_path = (
    RESULTS_DIR / "mediapipe_class_statistics.csv"
)

class_statistics.to_csv(
    class_results_path,
    index=False
)


# ============================================
# Final report
# ============================================

print()
print("============================================")
print("          FINAL DETECTION REPORT")
print("============================================")

print(f"Total images:       {total}")
print(f"Hands detected:     {detected}")
print(f"Detection failures: {failed}")
print(f"Read errors:        {read_errors}")

print()
print(f"Detection rate: {detection_rate:.2f}%")
print(f"Failure rate:   {failure_rate:.2f}%")

print()
print("Saved files:")
print(all_results_path)
print(failed_results_path)
print(class_results_path)

print()
print("============================================")