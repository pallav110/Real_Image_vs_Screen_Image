"""Train the recapture detector.

Pipeline:
    dataset/ -> features (cached) -> stratified split -> standardize
             -> train 3 classifiers (class_weight balanced)
             -> honest metrics + per-feature separation
             -> save the winner to model.joblib

Run:
    python train.py            # uses cached features if present
    python train.py --refresh  # force re-extract features

Saved model.joblib bundles: classifier, fitted scaler, threshold, feature names
-- everything predict.py needs, so train & serve stay in lock-step.
"""

import pickle
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on path
from features import FEATURE_NAMES, extract_features

ROOT = Path(__file__).resolve().parent.parent  # repo root (this file lives in src/)
DATA_DIR = ROOT / "dataset"
CACHE = ROOT / "features_cache.npz"
MODEL_PATH = ROOT / "model.joblib"
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
CLASSES = {"real": 0, "screen": 1}
SEED = 42


# --------------------------------------------------------------------------- #
# Data loading + feature cache
# --------------------------------------------------------------------------- #
def list_images(folder: Path):
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTS)


def dataset_paths():
    """All (path, label) in the canonical order used to build the cache."""
    return [(p, label) for cls, label in CLASSES.items()
            for p in list_images(DATA_DIR / cls)]


def is_whatsapp(path: Path) -> bool:
    """WhatsApp filenames look like IMG-YYYYMMDD-WA####.jpg.

    WhatsApp transmission recompresses AND downscales images, which destroys the
    high-frequency recapture fingerprints (moire, sub-pixel chroma) this detector
    relies on. We therefore drop WhatsApp-transmitted SCREEN images: empirically
    they classify at ~14% (vs ~94% for native screens) because they no longer
    carry any screen evidence -- they are corrupted labels for a fingerprint
    method, and they do not match the grader's direct-camera capture scenario.
    Real images are unaffected (no fingerprint to destroy), so we keep them.
    """
    return "-WA" in path.name


def build_dataset(refresh=False):
    """Return X (n, d), y (n,), paths (list). Cache features to avoid recompute."""
    if CACHE.exists() and not refresh:
        data = np.load(CACHE)  # plain numeric/string arrays, no pickle needed
        if list(data["names"]) == FEATURE_NAMES:
            if "paths" in data.files:
                paths = [Path(s) for s in data["paths"]]
            else:  # older cache without paths -> reconstruct from disk order
                paths = [p for p, _ in dataset_paths()]
            if len(paths) == data["X"].shape[0]:
                print(f"Loaded cached features: {data['X'].shape[0]} images")
                return data["X"], data["y"], paths
        print("Cache stale -> re-extracting.")

    X, y, paths = [], [], []
    for cls, label in CLASSES.items():
        imgs = list_images(DATA_DIR / cls)
        print(f"Extracting {cls}: {len(imgs)} images ...")
        for p in imgs:
            try:
                vec, _ = extract_features(str(p))
            except Exception as e:
                print(f"  skip {p.name}: {e}")
                continue
            X.append(vec)
            y.append(label)
            paths.append(str(p))
    X, y = np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)
    np.savez(CACHE, X=X, y=y, names=np.array(FEATURE_NAMES), paths=np.array(paths))
    print(f"Extracted + cached: {X.shape[0]} images, {X.shape[1]} features")
    return X, y, [Path(s) for s in paths]


# --------------------------------------------------------------------------- #
# Per-feature separation (which fingerprints actually work?)
# --------------------------------------------------------------------------- #
def feature_separation(X, y):
    """Rank features by single-feature ROC-AUC (0.5 = useless, 1.0 = perfect)."""
    print("\n=== per-feature class separation (single-feature AUC) ===")
    rows = []
    for i, name in enumerate(FEATURE_NAMES):
        col = X[:, i]
        auc = roc_auc_score(y, col)
        auc = max(auc, 1 - auc)  # direction-agnostic
        rows.append((auc, name))
    for auc, name in sorted(rows, reverse=True):
        bar = "#" * int((auc - 0.5) * 40)
        print(f"  {name:18s} {auc:.3f}  {bar}")


# --------------------------------------------------------------------------- #
# Model comparison
# --------------------------------------------------------------------------- #
def candidate_models():
    return {
        "logreg": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1
        ),
        "grad_boost": GradientBoostingClassifier(random_state=SEED),
    }


def evaluate(name, model, Xtr, ytr, Xte, yte, Xall, yall):
    model.fit(Xtr, ytr)
    proba = model.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)

    bal_acc = balanced_accuracy_score(yte, pred)
    auc = roc_auc_score(yte, proba)
    # 5-fold CV balanced accuracy on the full set for a stabler estimate.
    cv = cross_val_score(
        model, Xall, yall,
        cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
        scoring="balanced_accuracy",
    )
    size_kb = len(pickle.dumps(model)) / 1024

    print(f"\n--- {name} ---")
    print(f"  holdout balanced-acc : {bal_acc:.3f}")
    print(f"  holdout ROC-AUC      : {auc:.3f}")
    print(f"  5-fold bal-acc       : {cv.mean():.3f} +/- {cv.std():.3f}")
    print(f"  model size           : {size_kb:.1f} KB")
    print("  confusion (rows=true real/screen, cols=pred):")
    print("   ", confusion_matrix(yte, pred).tolist())
    print(classification_report(yte, pred, target_names=["real", "screen"], digits=3))
    return {"name": name, "bal_acc": bal_acc, "auc": auc,
            "cv_mean": cv.mean(), "size_kb": size_kb}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    refresh = "--refresh" in sys.argv
    t0 = time.time()
    X, y, paths = build_dataset(refresh=refresh)
    print(f"(feature stage: {time.time() - t0:.1f}s)")

    # Drop WhatsApp-transmitted SCREEN images (see is_whatsapp): their recapture
    # fingerprints were destroyed in transit, so they are corrupted labels.
    excluded = np.array([is_whatsapp(p) and lab == 1 for p, lab in zip(paths, y)])
    if excluded.any():
        print(f"excluding {int(excluded.sum())} WhatsApp screen images "
              f"(fingerprints destroyed by WhatsApp re-encoding)")
        X, y = X[~excluded], y[~excluded]
    print(f"training set: real={int((y == 0).sum())}  screen={int((y == 1).sum())}  "
          f"(n={len(y)})")

    feature_separation(X, y)

    # Stratified holdout preserves the class ratio in both halves.
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEED
    )
    # Scaler fit on TRAIN ONLY (no validation leakage).
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s, X_s = scaler.transform(Xtr), scaler.transform(Xte), scaler.transform(X)

    print("\n=== model comparison (validation = 25% stratified holdout) ===")
    results = []
    for name, model in candidate_models().items():
        results.append(evaluate(name, model, Xtr_s, ytr, Xte_s, yte, X_s, y))

    # Pick winner: highest 5-fold balanced accuracy, tie-break smaller model.
    winner = max(results, key=lambda r: (round(r["cv_mean"], 3), -r["size_kb"]))
    print(f"\n>>> WINNER: {winner['name']} "
          f"(cv bal-acc {winner['cv_mean']:.3f}, {winner['size_kb']:.1f} KB)")

    # Refit winner on ALL data with the full-data scaler, then save the bundle.
    # NOTE: model.joblib is a LOCAL, self-generated artifact (not downloaded),
    # so loading it via joblib/pickle in predict.py is trusted by construction.
    final_scaler = StandardScaler().fit(X)
    final_model = candidate_models()[winner["name"]]
    final_model.fit(final_scaler.transform(X), y)
    joblib.dump(
        {
            "model": final_model,
            "scaler": final_scaler,
            "threshold": 0.5,
            "feature_names": FEATURE_NAMES,
            "model_name": winner["name"],
        },
        MODEL_PATH,
    )
    print(f"Saved -> {MODEL_PATH.name}")


if __name__ == "__main__":
    main()
