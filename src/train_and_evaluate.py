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

DATA_CSV = Path("data/processed/landmarks_normalized.csv")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")

N_FOLDS = 5
EPOCHS = 200  # was 100 - val_loss was still dropping at epoch 100
BATCH_SIZE = 32

# Every class is augmented up to this many training samples per fold.
# Classes already at or above this count are left alone.
AUGMENT_TARGET = 80

# Balanced class weights are clipped to this max so a 13-sample class
# doesn't get a weight so large the gradient becomes unstable.
MAX_CLASS_WEIGHT = 5.0

MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)


# -----------------------------
# Landmark augmentation
# -----------------------------
# Each sample is a flat 63-dim vector: 21 landmarks x (x, y, z),
# already normalized (wrist-relative, scale-normalized). We create
# synthetic variants with a small in-plane rotation, a small scale
# jitter, and a little coordinate noise. This only works because the
# features are raw normalized coordinates, not learned embeddings.

def augment_sample(vec, rng):
    coords = vec.reshape(21, 3).copy()

    # Small in-plane rotation (rotate x/y around the origin)
    angle = np.deg2rad(rng.uniform(-8, 8))
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    x = coords[:, 0].copy()
    y = coords[:, 1].copy()

    coords[:, 0] = x * cos_a - y * sin_a
    coords[:, 1] = x * sin_a + y * cos_a

    # Small uniform scale jitter
    scale = rng.uniform(0.93, 1.07)
    coords[:, :2] *= scale

    # Small per-coordinate noise
    coords += rng.normal(0.0, 0.01, coords.shape)

    return coords.reshape(-1)


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


# -----------------------------
# Normalization / class weights
# -----------------------------

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

    feature_cols = []

    for i in range(21):
        feature_cols.extend([f"x{i}", f"y{i}", f"z{i}"])

    X = df[feature_cols].values.astype("float32")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["class"].values)

    class_names = list(label_encoder.classes_)
    n_classes = len(class_names)

    print()
    print("================================")
    print("DATASET")
    print("================================")
    print(f"Samples  : {len(X)}")
    print(f"Classes  : {n_classes}")
    print(f"Features : {X.shape[1]}")
    print()

    with open(MODELS_DIR / "class_names.json", "w", encoding="utf-8") as f:
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

        # Augment small classes in the TRAINING split only.
        # Never touch X_test - that would leak information into
        # the fold's evaluation.
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
        f.write("5-fold stratified cross-validation\n")
        f.write(f"Samples: {len(X)}\n")
        f.write(f"Classes: {n_classes}\n")
        f.write(f"Features: {X.shape[1]}\n")
        f.write(f"Overall accuracy: {overall_accuracy:.4f}\n")

    # -----------------------------
    # Confusion matrix (row-normalized, easier to read with imbalance)
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
    plt.title("TuniSign - 5-Fold Confusion Matrix (row-normalized)")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    print("Saved confusion matrix.")

    # -----------------------------
    # Top confused class pairs
    # -----------------------------
    # A 56x56 heatmap is hard to read by eye. This pulls out the
    # off-diagonal cells with the most misclassifications - i.e. which
    # specific class gets mistaken for which other specific class, and
    # how often. Much faster way to see whether errors are "genuinely
    # similar signs" vs "just needs more data".

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

    top_confused_lines = [
        f"{count:>4}  true={true_name:<12} predicted={pred_name}"
        for count, true_name, pred_name in confused_pairs[:20]
    ]

    print()
    print("Top confused pairs (true -> predicted, count):")
    for line in top_confused_lines:
        print(line)

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

    # Hold out a small validation slice purely to know when to stop.
    # This model still trains on ~90% of the (augmented) data.
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
            f"triggering - the model may not have fully converged. "
            f"Consider raising EPOCHS further."
        )

    final_model.save(MODELS_DIR / "tunisign_model.keras")

    print()
    print("================================")
    print("DONE!")
    print("================================")
    print("Model saved to:")
    print("models/tunisign_model.keras")


if __name__ == "__main__":
    main()