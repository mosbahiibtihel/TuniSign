import numpy as np
import pandas as pd

INPUT_FILE = "data/processed/landmarks_2hands.csv"
OUTPUT_FILE = "data/processed/landmarks_2hands_normalized.csv"


# ------------------------------------------------------------
# STEP 1: Load the raw two-hand file.
# Every row = one image. After the info columns, we expect
# 126 numbers: the first 63 belong to one hand, the next 63 to
# the other hand. A missing hand is expected to be all zeros
# (or empty).
# ------------------------------------------------------------
df = pd.read_csv(INPUT_FILE)

metadata_cols = [c for c in ["category", "class", "image_path"] if c in df.columns]
raw_cols = [c for c in df.columns if c not in metadata_cols]

print("Info columns      :", metadata_cols)
print("Number of features:", len(raw_cols))
print("First 3 feature columns :", raw_cols[:3])
print("Columns 64-66 (start of 2nd hand):", raw_cols[63:66])

if len(raw_cols) != 126:
    raise SystemExit(
        f"ERROR: expected 126 landmark columns, found {len(raw_cols)}. "
        "Check landmarks_2hands.csv before continuing."
    )


# ------------------------------------------------------------
# STEP 2: Small helper functions.
# ------------------------------------------------------------
def get_hand(values_63):
    """Return a (21, 3) array, or None if this hand is missing."""
    numbers = np.asarray(values_63, dtype=float)

    if np.isnan(numbers).any() or np.all(numbers == 0):
        return None

    return numbers.reshape(21, 3)


def normalize_one_hand(points):
    """
    Same recipe as before, for ONE hand:
    1. wrist becomes (0, 0, 0)
    2. divide by the biggest x/y distance from the wrist
    Also returns the wrist's ORIGINAL x and y (position in the image).
    """
    wrist = points[0].copy()
    relative = points - wrist

    scale = np.max(np.linalg.norm(relative[:, :2], axis=1))
    if scale < 1e-6:
        scale = 1.0

    relative = relative / scale

    return relative.flatten(), wrist[0], wrist[1]


def process_row(row):
    values = row[raw_cols].values.astype(float)

    hands = []
    for start in (0, 63):
        hand = get_hand(values[start:start + 63])
        if hand is not None:
            hands.append(hand)

    # Put the hands in a fixed order: leftmost wrist (in the image)
    # first. The browser does exactly the same, so the model always
    # sees the hands in the same order.
    hands.sort(key=lambda h: h[0, 0])

    output = []

    for slot in range(2):
        if slot < len(hands):
            flat, wrist_x, wrist_y = normalize_one_hand(hands[slot])
            output.extend(flat)
            output.extend([wrist_x, wrist_y, 1.0])   # 1.0 = hand present
        else:
            output.extend([0.0] * 66)                # slot empty

    return output


# ------------------------------------------------------------
# STEP 3: Sanity check on the raw positions.
# MediaPipe x/y values should be between 0 and 1. If you see
# big numbers here, the extraction saved pixels, not fractions -
# tell me before training.
# ------------------------------------------------------------
x_cols = [c for c in raw_cols if c.startswith("x") or "_x" in c]
if x_cols:
    x_values = df[x_cols].values.astype(float)
    x_values = x_values[~np.isnan(x_values)]
    x_values = x_values[x_values != 0]
    print(f"Raw x values range from {x_values.min():.3f} to {x_values.max():.3f}")


# ------------------------------------------------------------
# STEP 4: Normalize every row and save.
# ------------------------------------------------------------
rows = [process_row(row) for _, row in df.iterrows()]

feature_names = []
for hand_number in (1, 2):
    for i in range(21):
        feature_names.extend([
            f"h{hand_number}_x{i}",
            f"h{hand_number}_y{i}",
            f"h{hand_number}_z{i}",
        ])
    feature_names.extend([
        f"h{hand_number}_wrist_x",
        f"h{hand_number}_wrist_y",
        f"h{hand_number}_present",
    ])

normalized = pd.DataFrame(rows, columns=feature_names)

result = pd.concat(
    [df[metadata_cols].reset_index(drop=True), normalized],
    axis=1
)

result.to_csv(OUTPUT_FILE, index=False)

one_hand = int((normalized["h1_present"] == 1).sum() - (normalized["h2_present"] == 1).sum())
two_hands = int((normalized["h2_present"] == 1).sum())
no_hand = int((normalized["h1_present"] == 0).sum())

print()
print("Normalization complete!")
print(f"Samples          : {len(result)}")
print(f"One-hand samples : {one_hand}")
print(f"Two-hand samples : {two_hands}")
print(f"No hand found    : {no_hand}")
print(f"Features         : {len(feature_names)}")
print(f"Saved to         : {OUTPUT_FILE}")