import pandas as pd
import numpy as np

INPUT_FILE = "data/processed/landmarks.csv"
OUTPUT_FILE = "data/processed/landmarks_normalized.csv"


# Load the existing landmarks
df = pd.read_csv(INPUT_FILE)

# The 63 landmark columns
feature_cols = []

for i in range(21):
    feature_cols.extend([
        f"x{i}",
        f"y{i}",
        f"z{i}"
    ])


def normalize_landmarks(row):
    # Convert the 63 values into 21 points of (x, y, z)
    points = row[feature_cols].values.astype(float).reshape(21, 3)

    # 1. Put the wrist (landmark 0) at position (0, 0, 0)
    wrist = points[0].copy()
    points = points - wrist

    # 2. Calculate the size of the hand
    distances = np.linalg.norm(points[:, :2], axis=1)
    scale = np.max(distances)

    # Avoid division by zero
    if scale < 1e-6:
        scale = 1.0

    # 3. Make the hand size comparable
    points = points / scale

    return points.flatten()


# Normalize every image
normalized_data = df.apply(
    normalize_landmarks,
    axis=1,
    result_type="expand"
)

# Give the normalized columns their names
normalized_data.columns = feature_cols

# Keep the information about each image
metadata_cols = ["category", "class", "image_path"]

result = pd.concat(
    [df[metadata_cols], normalized_data],
    axis=1
)

# Save the new dataset
result.to_csv(OUTPUT_FILE, index=False)

print("Normalization complete!")
print(f"Original samples: {len(df)}")
print(f"Normalized samples: {len(result)}")
print(f"Saved to: {OUTPUT_FILE}")
print(f"Features: {len(feature_cols)}")