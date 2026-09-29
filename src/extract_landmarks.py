from pathlib import Path

import cv2
import pandas as pd
import mediapipe as mp


# -----------------------------
# Paths
# -----------------------------

DATASET_DIR = Path("data/raw/Dataset/Data")
OUTPUT_DIR = Path("data/processed")

OUTPUT_FILE = OUTPUT_DIR / "landmarks_2hands.csv"


# -----------------------------
# Settings
# -----------------------------

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# métro has 0% MediaPipe detection
EXCLUDED_CLASS = "métro"


# -----------------------------
# MediaPipe
# -----------------------------

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions


options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="models/hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
)


# -----------------------------
# Main
# -----------------------------

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    total_images = 0
    detected_images = 0
    failed_images = 0

    print("Starting TWO-HAND landmark extraction...")
    print()

    with HandLandmarker.create_from_options(options) as landmarker:

        # Go through categories
        for category_folder in DATASET_DIR.iterdir():

            if not category_folder.is_dir():
                continue

            # Go through classes
            for class_folder in category_folder.iterdir():

                if not class_folder.is_dir():
                    continue

                class_name = class_folder.name

                # Skip métro
                if class_name == EXCLUDED_CLASS:
                    print("Skipping:", class_name)
                    continue

                print("Processing:", class_name)

                # Go through images
                for image_path in class_folder.iterdir():

                    if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                        continue

                    total_images += 1

                    # Read image
                    image = cv2.imread(str(image_path))

                    if image is None:
                        failed_images += 1
                        continue

                    # Convert BGR → RGB
                    image_rgb = cv2.cvtColor(
                        image,
                        cv2.COLOR_BGR2RGB
                    )

                    # Create MediaPipe image
                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=image_rgb
                    )

                    # Detect up to 2 hands
                    result = landmarker.detect(mp_image)

                    if not result.hand_landmarks:
                        failed_images += 1
                        continue

                    # ---------------------------------
                    # Create row
                    # ---------------------------------

                    row = {
                        "category": category_folder.name,
                        "class": class_name,
                        "image_path": str(image_path)
                    }

                    # ---------------------------------
                    # Save LEFT and RIGHT hand
                    # ---------------------------------

                    left_hand = None
                    right_hand = None

                    for hand_landmarks, handedness in zip(
                        result.hand_landmarks,
                        result.handedness
                    ):

                        label = handedness[0].category_name

                        if label == "Left":
                            left_hand = hand_landmarks

                        elif label == "Right":
                            right_hand = hand_landmarks

                    # ---------------------------------
                    # LEFT HAND
                    # ---------------------------------

                    for i in range(21):

                        if left_hand is not None:

                            landmark = left_hand[i]

                            row[f"left_x{i}"] = landmark.x
                            row[f"left_y{i}"] = landmark.y
                            row[f"left_z{i}"] = landmark.z

                        else:

                            row[f"left_x{i}"] = 0.0
                            row[f"left_y{i}"] = 0.0
                            row[f"left_z{i}"] = 0.0

                    # ---------------------------------
                    # RIGHT HAND
                    # ---------------------------------

                    for i in range(21):

                        if right_hand is not None:

                            landmark = right_hand[i]

                            row[f"right_x{i}"] = landmark.x
                            row[f"right_y{i}"] = landmark.y
                            row[f"right_z{i}"] = landmark.z

                        else:

                            row[f"right_x{i}"] = 0.0
                            row[f"right_y{i}"] = 0.0
                            row[f"right_z{i}"] = 0.0

                    rows.append(row)
                    detected_images += 1

    # -----------------------------
    # Save CSV
    # -----------------------------

    df = pd.DataFrame(rows)

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # -----------------------------
    # Final report
    # -----------------------------

    print()
    print("=" * 50)
    print("TWO-HAND LANDMARK EXTRACTION COMPLETE")
    print("=" * 50)

    print("Images processed:", total_images)
    print("Images with at least one hand:", detected_images)
    print("Detection failures:", failed_images)

    if total_images > 0:
        print(
            "Detection rate:",
            round(detected_images / total_images * 100, 2),
            "%"
        )

    print("Classes:", df["class"].nunique())

    print("Landmark features: 126")

    print()
    print("Saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()