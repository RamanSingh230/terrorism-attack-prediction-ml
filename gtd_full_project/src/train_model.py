"""
GTD Terrorist Group Prediction - Full Training Pipeline
=====================================================
Data cleaning -> Feature engineering -> Train/test split -> Baseline model
-> GridSearchCV hyperparameter tuning -> Cross-validation -> Final evaluation
-> Save model + encoders for deployment (Streamlit app uses these).

Predicts the perpetrating GROUP/ORGANIZATION from event characteristics
(attack type, weapon, target, region, casualties, etc). Attack Type is now
an input feature instead of the target.

Run in stages (each checkpoints to disk so you can resume):
    python src/train_model.py prep       # clean + engineer + split
    python src/train_model.py tune       # GridSearchCV hyperparameter search
    python src/train_model.py finalize   # CV + test evaluation + save model

Or run everything in one go:
    python src/train_model.py all
"""

import sys
import gc
import pandas as pd
import numpy as np
import joblib
import json
import time
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score

# -----------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "data" / "gtd_final.csv"
MODELS_DIR = ROOT / "models"
CKPT_DIR = ROOT / "checkpoints"
MODELS_DIR.mkdir(exist_ok=True)
CKPT_DIR.mkdir(exist_ok=True)

TARGET_COLUMN = "Terrorist Group"
CAT_COLS = ["country_txt", "region_txt", "Target", "Weapons", "Attack Type"]
RANDOM_STATE = 42

# "Unknown" means GTD analysts couldn't attribute the attack to a specific group.
# Per project owner's choice, this is kept as a valid, real output class - if the
# event characteristics look like the many historically-unattributed incidents,
# "Unknown" is the honest prediction rather than a forced guess at a named group.
DROP_UNKNOWN_GROUP = False

# Groups need to appear at least this many times or there's no realistic way for
# the model to learn a pattern for them (some groups have 1-2 incidents total).
# Testing showed 150 leaves ~89 sparse classes with mediocre accuracy and a huge
# (multi-hundred-MB) model, since each Random Forest node stores a per-class
# count array. Raising this to 500 keeps only the ~30 groups with enough history
# to actually learn a signature for, which happens to raise BOTH accuracy and
# macro-F1 substantially while shrinking the model to a deployable size - this
# isn't cherry-picking a metric, it's a more honest problem: predicting among
# a fixed set of well-documented groups instead of also guessing among dozens
# of one-off actors the model has almost no data on.
MIN_CLASS_COUNT = 500


# -----------------------------------------------------------------------
# STAGE 1: LOAD + CLEAN + FEATURE ENGINEER + SPLIT
# -----------------------------------------------------------------------
def stage_prep():
    print("[1/4] Loading data ...")
    df = pd.read_csv(DATA_PATH, encoding="latin1", low_memory=False)
    print(f"      Loaded {len(df):,} rows")

    print("[2/4] Cleaning ...")
    keep_cols = [
        "iyear", "imonth", "iday",
        "country_txt", "region_txt", "provstate", "City",
        "latitude", "longitude",
        "Attack Type", "Target", "Weapons", "Terrorist Group",
        "Fatalities", "Injured", "Success",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()
    df["Fatalities"] = pd.to_numeric(df["Fatalities"], errors="coerce").fillna(0)
    df["Injured"] = pd.to_numeric(df["Injured"], errors="coerce").fillna(0)
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["latitude"] = df["latitude"].fillna(df["latitude"].median())
    df["longitude"] = df["longitude"].fillna(df["longitude"].median())
    df = df.dropna(subset=[TARGET_COLUMN, "region_txt", "country_txt"])

    if DROP_UNKNOWN_GROUP:
        before = len(df)
        df = df[df[TARGET_COLUMN].astype(str).str.strip().str.lower() != "unknown"]
        print(f"      Dropped 'Unknown' group rows: {before - len(df):,} removed, {len(df):,} remain")

    counts = df[TARGET_COLUMN].value_counts()
    valid_classes = counts[counts >= MIN_CLASS_COUNT].index
    df = df[df[TARGET_COLUMN].isin(valid_classes)]
    print(f"      After cleaning: {len(df):,} rows, {df[TARGET_COLUMN].nunique()} classes")

    print("[3/4] Feature engineering ...")
    features = df.copy()
    encoders = {}
    for col in CAT_COLS:
        le = LabelEncoder()
        features[col + "_enc"] = le.fit_transform(features[col].astype(str))
        encoders[col] = le
    features["Success_enc"] = (
        features["Success"].astype(str).str.lower().str.contains("success").astype(int)
    )
    feature_cols = [c + "_enc" for c in CAT_COLS] + [
        "iyear", "imonth", "Fatalities", "Injured", "Success_enc"
    ]
    X = features[feature_cols]
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(features[TARGET_COLUMN])
    encoders["__target__"] = target_encoder

    print("[4/4] Splitting train/test (80/20, stratified) ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"      Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")

    joblib.dump(X_train, CKPT_DIR / "X_train.pkl")
    joblib.dump(X_test, CKPT_DIR / "X_test.pkl")
    joblib.dump(y_train, CKPT_DIR / "y_train.pkl")
    joblib.dump(y_test, CKPT_DIR / "y_test.pkl")
    joblib.dump(encoders, CKPT_DIR / "encoders.pkl")
    joblib.dump(feature_cols, CKPT_DIR / "feature_cols.pkl")
    print("Checkpoint saved. Run stage 'tune' next.")


# -----------------------------------------------------------------------
# STAGE 2: GRIDSEARCHCV HYPERPARAMETER TUNING (on a stratified subsample
# for speed; final model is refit on full training data in 'finalize')
# -----------------------------------------------------------------------
def stage_tune():
    X_train = joblib.load(CKPT_DIR / "X_train.pkl")
    y_train = joblib.load(CKPT_DIR / "y_train.pkl")

    # Use a stratified subsample just for the search to keep grid search fast;
    # the winning config gets refit on the FULL training set in stage_finalize.
    # (Group data is smaller after dropping "Unknown" + rare classes, so only
    # subsample if there's a decent surplus over the training set; otherwise
    # just tune on the full training set - stratified splitting needs the
    # leftover slice to still contain every class.)
    n_classes = len(np.unique(y_train))
    if len(X_train) > 40000 + n_classes * 5:
        X_sub, _, y_sub, _ = train_test_split(
            X_train, y_train, train_size=40000, random_state=RANDOM_STATE, stratify=y_train
        )
    else:
        X_sub, y_sub = X_train, y_train
    print(f"Running GridSearchCV on {X_sub.shape[0]:,}-row subsample ...")

    # Range validated empirically on the real GTD data with "Unknown" kept as a
    # real class and class_weight=None (see notes above best_model config below):
    # min_samples_leaf still controls saved-model size the most, since forests
    # over 30+ classes store a per-class count array at every node.
    param_grid = {
        "n_estimators": [80, 100],
        "max_depth": [15, 18],
        "min_samples_leaf": [5, 6],
    }
    base_model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1, class_weight=None)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    grid_search = GridSearchCV(
        base_model, param_grid, cv=cv, scoring="f1_macro", n_jobs=-1, verbose=1
    )

    start = time.time()
    grid_search.fit(X_sub, y_sub)
    elapsed = time.time() - start

    print(f"Done in {elapsed:.1f}s")
    print(f"Best params: {grid_search.best_params_}")
    print(f"Best CV macro-F1 (subsample): {grid_search.best_score_:.4f}")

    joblib.dump(grid_search.best_params_, CKPT_DIR / "best_params.pkl")
    with open(CKPT_DIR / "tune_summary.json", "w") as f:
        json.dump({
            "best_params": grid_search.best_params_,
            "best_cv_macro_f1_subsample": grid_search.best_score_,
        }, f, indent=2)
    print("Checkpoint saved. Run stage 'finalize' next.")


# -----------------------------------------------------------------------
# STAGE 3: REFIT ON FULL DATA + CROSS-VALIDATION + TEST EVALUATION + SAVE
# -----------------------------------------------------------------------
def stage_finalize():
    X_train = joblib.load(CKPT_DIR / "X_train.pkl")
    X_test = joblib.load(CKPT_DIR / "X_test.pkl")
    y_train = joblib.load(CKPT_DIR / "y_train.pkl")
    y_test = joblib.load(CKPT_DIR / "y_test.pkl")
    encoders = joblib.load(CKPT_DIR / "encoders.pkl")
    feature_cols = joblib.load(CKPT_DIR / "feature_cols.pkl")
    best_params = joblib.load(CKPT_DIR / "best_params.pkl")

    print(f"Best params: {best_params}")

    print("Running 3-fold cross-validation on training set ...")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    # n_jobs=1 here on purpose: cross_val_score with n_jobs=-1 clones and fits a
    # full forest per fold *in parallel*, which multiplies memory use by the
    # fold count. With ~90 group classes, each tree node stores a per-class
    # count array, so forests are already memory-heavy - safer to fit folds
    # one at a time even though it's slower. We also run CV *before* fitting
    # the final full model below and gc.collect() after, so we never hold two
    # full-size forests in memory at once.
    cv_model = RandomForestClassifier(
        **best_params, random_state=RANDOM_STATE, n_jobs=1, class_weight=None
    )
    cv_scores = cross_val_score(cv_model, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=1)
    print(f"CV macro-F1 scores: {np.round(cv_scores, 4)}  Mean: {cv_scores.mean():.4f}  Std: {cv_scores.std():.4f}")
    del cv_model
    gc.collect()

    print(f"Refitting best model on full training set with {best_params} ...")
    model = RandomForestClassifier(
        **best_params, random_state=RANDOM_STATE, n_jobs=1, class_weight=None
    )
    model.fit(X_train, y_train)

    print("Evaluating on held-out test set ...")
    y_pred = model.predict(X_test)
    target_encoder = encoders["__target__"]
    y_test_labels = target_encoder.inverse_transform(y_test)
    y_pred_labels = target_encoder.inverse_transform(y_pred)
    print(classification_report(y_test_labels, y_pred_labels))

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    print(f"Test Accuracy: {acc:.4f}  Test Macro-F1: {f1:.4f}")

    # Sanity check: model.pkl and encoders.pkl must always be a matched pair.
    # If these ever diverge (e.g. one file gets regenerated/committed without
    # the other, or MIN_CLASS_COUNT changes between runs), the deployed app
    # will silently predict class indices the target encoder can't decode,
    # producing a "y contains previously unseen labels" crash at inference
    # time instead of failing here where it's obvious what went wrong.
    n_model_classes = len(model.classes_)
    n_encoder_classes = len(encoders["__target__"].classes_)
    if n_model_classes != n_encoder_classes:
        raise RuntimeError(
            f"Model/encoder class count mismatch: model has {n_model_classes} "
            f"classes but target encoder has {n_encoder_classes} classes. "
            "These must be generated together in the same run - refusing to "
            "save a mismatched pair. Re-run 'prep' through 'finalize' fresh."
        )
    print(f"Sanity check passed: model and target encoder both have {n_model_classes} classes.")

    joblib.dump(model, MODELS_DIR / "model.pkl", compress=3)
    joblib.dump(encoders, MODELS_DIR / "encoders.pkl")
    joblib.dump(feature_cols, MODELS_DIR / "feature_cols.pkl")

    summary = {
        "best_params": best_params,
        "cv_scores": cv_scores.tolist(),
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "test_accuracy": acc,
        "test_macro_f1": f1,
    }
    with open(MODELS_DIR / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved model.pkl, encoders.pkl, feature_cols.pkl, training_summary.json to /models")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage == "prep":
        stage_prep()
    elif stage == "tune":
        stage_tune()
    elif stage == "finalize":
        stage_finalize()
    elif stage == "all":
        stage_prep()
        stage_tune()
        stage_finalize()
    else:
        print("Unknown stage. Use: prep | tune | finalize | all")

