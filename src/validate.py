"""Reproducible PROOF that the detector generalizes (does not memorize the data).

Run:
    python validate.py

Three independent pieces of evidence, all on the shipped feature set + the same
WhatsApp-screen filter train.py uses:

  1. TRAIN vs UNSEEN HELD-OUT gap. A memorizing model scores ~100% on the images
     it trained on but much lower on unseen ones -> a large positive gap. A gap
     near zero means the model learned a general rule, not the dataset.
  2. 5-FOLD CROSS-VALIDATION. Every prediction is made on an image left out of
     training; a tight spread across folds => stable generalization.
  3. MODEL CAPACITY. A linear model with ~k coefficients cannot memorize n >> k
     images -- there is simply nowhere to store them.

Note: this certifies no *image-level* overfitting on this dataset. It cannot
certify robustness to unseen capture *devices* -- that is the honest limit of any
self-collected take-home, addressed by the physics-based (causal) features and
the confound checks in check_data.py.
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on path
from train import build_dataset, is_whatsapp, SEED


def main():
    X, y, paths = build_dataset()
    excl = np.array([is_whatsapp(p) and lab == 1 for p, lab in zip(paths, y)])
    keep = ~excl
    X, y = X[keep], y[keep]
    names = np.array([p.name for p, e in zip(paths, excl) if not e])
    print(f"dataset (WhatsApp-screens excluded): "
          f"real={int((y == 0).sum())} screen={int((y == 1).sum())} n={len(y)}\n")

    # 1) Train vs unseen held-out -------------------------------------------- #
    Xtr, Xte, ytr, yte, _, nte = train_test_split(
        X, y, names, test_size=0.25, stratify=y, random_state=SEED
    )
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(
        sc.transform(Xtr), ytr
    )
    tr = accuracy_score(ytr, clf.predict(sc.transform(Xtr)))
    pte = clf.predict_proba(sc.transform(Xte))[:, 1]
    te = accuracy_score(yte, (pte >= 0.5).astype(int))
    print("1) TRAIN vs UNSEEN HELD-OUT (25%)")
    print(f"   train acc {tr:.3f} | test acc {te:.3f} | "
          f"gap {tr - te:+.3f}   (near 0 => not memorizing)\n")

    # 2) Cross-validation ---------------------------------------------------- #
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    cv = cross_val_score(
        pipe, X, y,
        cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
        scoring="balanced_accuracy",
    )
    print("2) 5-FOLD CROSS-VALIDATION (every prediction on an unseen image)")
    print(f"   balanced accuracy {cv.mean():.3f} +/- {cv.std():.3f}   "
          f"per-fold {np.round(cv, 3)}\n")

    # 3) Capacity ------------------------------------------------------------ #
    k = X.shape[1] + 1  # weights + bias
    print("3) MODEL CAPACITY")
    print(f"   logistic-regression parameters: {k}   vs   training images: {len(y)}")
    print(f"   ~{len(y) / k:.0f} images per parameter -> far too few params to memorize.\n")

    # Concrete: scores on images the model never saw ------------------------- #
    print("Sample scores on UNSEEN held-out images (model never trained on these):")
    order = np.argsort(pte)
    for i in list(order[:4]) + list(order[-4:]):
        tag = "screen" if yte[i] == 1 else "real"
        ok = "OK " if (pte[i] >= 0.5) == (yte[i] == 1) else "MISS"
        print(f"  [{ok}] true={tag:6s} score={pte[i]:.3f}  {nte[i]}")


if __name__ == "__main__":
    main()
