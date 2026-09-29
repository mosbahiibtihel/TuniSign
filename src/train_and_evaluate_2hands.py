import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns


# -----------------------------
# Reproducibility
# -----------------------------

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


# -----------------------------
# Paths and settings
# -----------------------------
# New names everywhere, so your one-hand model and reports are NOT
# overwritten - you can compare the two versions in your report.

DATA_CSV = Path("data/processed/landmarks_2hands_normalized.csv")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports_2hands")

MODEL_FILE = MODELS_DIR / "tunisign_model_2hands.keras"
SAVED_MODEL_DIR = MODELS_DIR / "tunisign_saved_model_2hands"
CLASS_NAMES_FILE = MODELS_DIR / "class_names_2hands.json"

N_FOLDS = 5
EPOCHS = 200
BATCH_SIZE = 32

AUGMENT_TARGET = 80
MAX_CLASS_WEIGHT = 5.0

# How much we randomly shake the hand POSITION during augmentation.
# Without this, the model could memorize the exact position of each
# recording instead of learning "roughly where" the hand is.
POSITION_JITTER = 0.03

# One feature vector = 132 numbers = 2 slots x 66:
#   63 landmark numbers, wrist_x, wrist_y, present (1 or 0)
SLOT_SIZE = 66
N_SLOTS = 2

MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)


# -----------------------------
# Landmark augmentation (two-hand version)
# -----------------------------
# For each hand that is present: small rotation, small size change,
# small noise on the shape, and a small shake of the wrist position.
# Empty hand slots are left untouched (they stay all zeros).

def augment_sample(vec, rng):
    out = vec.copy()

    for slot in range(N_SLOTS):
        start = slot * SLOT_SIZE

        present = out[start + 65]
        if present < 0.5:
            continue

        coords = out[start:start + 63].reshape(21, 3).copy()

        angle = np.deg2rad(rng.uniform(-8, 8))
        cos_a, sin_a = np.cos(angle), np.sin(angle)

        x = coords[:, 0].copy()
        y = coords[:, 1].copy()

        coords[:, 0] = x * cos_a - y * sin_a
        coords[:, 1] = x * sin_a + y * cos_a

        scale = rng.uniform(0.93, 1.07)
        coords[:, :2] *= scale

        coords += rng.normal(0.0, 0.01, coords.shape)

        out[start:start + 63] = coords.reshape(-1)

        out[start + 63] += rng.normal(0.0, POSITION_JITTER)
        out[start + 64] += rng.normal(0.0, POSITION_JITTER)

    return out


def augment_class_balance(X, y, target_count, seed):
    rng = np.random.default_rng(seed)

    X_parts = [X]
    y_parts = [y]

    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        n_have = len(cls_idx)

        if n_have >= target_count:
            continue

        n_needed = target_count - n_have
        chosen = rng.choice(cls_idx, size=n_needed, replace=True)

        X_new = np.array([
            augment_sample(X[i], rng)
            for i in chosen
        ])

        y_new = np.full(n_needed, cls)

        X_parts.append(X_new)
        y_parts.append(y_new)

    return np.vstack(X_parts), np.concatenate(y_parts)


# -----------------------------
# Build neural network
# -----------------------------

def build_model(n_features, n_classes, norm_layer):

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(n_features,)),

        norm_layer,

        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(n_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def make_norm_layer(X):

    norm = tf.keras.layers.Normalization(axis=-1)
    norm.adapt(X)

    return norm


def make_class_weights(y_train):

    classes = np.unique(y_train)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )

    weights = np.clip(weights, None, MAX_CLASS_WEIGHT)

    return dict(zip(classes, weights))


# -----------------------------
# Main
# -----------------------------

def main():

    print("Loading landmarks...")

    df = pd.read_csv(DATA_CSV)

    feature_cols = [
        c for c in df.columns
        if c.startswith("h1_") or c.startswith("h2_")
    ]

    if len(feature_cols) != N_SLOTS * SLOT_SIZE:
        raise SystemExit(
            f"ERROR: expected {N_SLOTS * SLOT_SIZE} features, "
            f"found {len(feature_cols)}. Run normalize_landmarks_2hands.py first."
        )

    X = df[feature_cols].values.astype("float32")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["class"].values)

    class_names = list(label_encoder.classes_)
    n_classes = len(class_names)

    two_hand_share = float(df["h2_present"].mean()) * 100

    print()
    print("================================")
    print("DATASET")
    print("================================")
    print(f"Samples           : {len(X)}")
    print(f"Classes           : {n_classes}")
    print(f"Features          : {X.shape[1]}")
    print(f"Two-hand samples  : {two_hand_share:.1f} %")
    print()

    with open(CLASS_NAMES_FILE, "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    # -----------------------------
    # 5-fold cross validation
    # -----------------------------

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    oof_predictions = np.zeros(len(y), dtype=int)

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):

        print()
        print("--------------------------------")
        print(f"FOLD {fold}/{N_FOLDS}")
        print("--------------------------------")

        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        # Augment the TRAINING split only - never the test split.
        X_train, y_train = augment_class_balance(
            X_train, y_train, AUGMENT_TARGET, seed=SEED + fold
        )

        class_weights = make_class_weights(y_train)

        norm = make_norm_layer(X_train)

        model = build_model(X.shape[1], n_classes, norm)

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True
        )

        fold_history = model.fit(
            X_train,
            y_train,
            validation_split=0.15,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            class_weight=class_weights,
            callbacks=[early_stop],
            verbose=0
        )

        predictions = model.predict(X_test, verbose=0)
        predictions = np.argmax(predictions, axis=1)

        oof_predictions[test_idx] = predictions

        accuracy = np.mean(predictions == y_test)
        stopped_epoch = len(fold_history.history["loss"])
        print(f"Fold {fold} accuracy: {accuracy:.4f} (stopped at epoch {stopped_epoch}/{EPOCHS})")

    # -----------------------------
    # Overall evaluation
    # -----------------------------

    print()
    print("================================")
    print("FINAL 5-FOLD RESULTS")
    print("================================")

    overall_accuracy = np.mean(oof_predictions == y)
    print(f"Overall accuracy: {overall_accuracy:.4f}")

    report = classification_report(
        y, oof_predictions,
        target_names=class_names,
        digits=3,
        zero_division=0
    )

    print()
    print(report)

    with open(REPORTS_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    with open(REPORTS_DIR / "cv_summary.txt", "w", encoding="utf-8") as f:
        f.write("5-fold stratified cross-validation (two-hand model)\n")
        f.write(f"Samples: {len(X)}\n")
        f.write(f"Classes: {n_classes}\n")
        f.write(f"Features: {X.shape[1]}\n")
        f.write(f"Overall accuracy: {overall_accuracy:.4f}\n")

    # -----------------------------
    # Confusion matrix
    # -----------------------------

    cm = confusion_matrix(y, oof_predictions, normalize="true")

    plt.figure(figsize=(18, 16))

    sns.heatmap(
        cm,
        xticklabels=class_names,
        yticklabels=class_names,
        cmap="Blues",
        vmin=0,
        vmax=1
    )

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("TuniSign (2 hands) - 5-Fold Confusion Matrix (row-normalized)")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    print("Saved confusion matrix.")

    # -----------------------------
    # Top confused class pairs
    # -----------------------------

    raw_cm = confusion_matrix(y, oof_predictions)

    confused_pairs = []

    for true_idx in range(n_classes):
        for pred_idx in range(n_classes):
            if true_idx == pred_idx:
                continue

            count = raw_cm[true_idx, pred_idx]

            if count > 0:
                confused_pairs.append((
                    count,
                    class_names[true_idx],
                    class_names[pred_idx]
                ))

    confused_pairs.sort(reverse=True)

    print()
    print("Top confused pairs (true -> predicted, count):")
    for count, true_name, pred_name in confused_pairs[:20]:
        print(f"{count:>4}  true={true_name:<12} predicted={pred_name}")

    with open(REPORTS_DIR / "top_confusions.txt", "w", encoding="utf-8") as f:
        f.write("Top confused pairs (true -> predicted, count)\n")
        f.write("=" * 50 + "\n")
        for count, true_name, pred_name in confused_pairs:
            f.write(f"{count:>4}  true={true_name:<12} predicted={pred_name}\n")

    # -----------------------------
    # Train final model
    # -----------------------------

    print()
    print("================================")
    print("TRAINING FINAL MODEL")
    print("================================")

    X_bal, y_bal = augment_class_balance(X, y, AUGMENT_TARGET, seed=SEED)

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_bal, y_bal,
        test_size=0.1,
        stratify=y_bal,
        random_state=SEED
    )

    full_class_weights = make_class_weights(y_fit)

    norm = make_norm_layer(X_fit)

    final_model = build_model(X.shape[1], n_classes, norm)

    final_early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    history = final_model.fit(
        X_fit,
        y_fit,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=full_class_weights,
        callbacks=[final_early_stop],
        verbose=1
    )

    best_epoch = len(history.history["loss"])
    print(f"Final model stopped after {best_epoch} epochs.")

    if best_epoch >= EPOCHS:
        print(
            f"WARNING: hit the {EPOCHS}-epoch cap without early stopping "
            f"triggering - consider raising EPOCHS."
        )

    final_model.save(MODEL_FILE)

    # SavedModel folder = the format your tensorflowjs_converter
    # command needs (--input_format=tf_saved_model).
    try:
        final_model.export(str(SAVED_MODEL_DIR))
        print(f"Exported SavedModel to: {SAVED_MODEL_DIR}")
    except Exception as error:
        print("Could not export the SavedModel folder:", error)

    print()
    print("================================")
    print("DONE!")
    print("================================")
    print("Keras model :", MODEL_FILE)
    print("SavedModel  :", SAVED_MODEL_DIR)
    print("Class names :", CLASS_NAMES_FILE)


if __name__ == "__main__":
    main()