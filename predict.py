"""The graded interface.

Usage:
    python predict.py some_image.jpg
Prints ONE number from 0 to 1:
    0 = real photo,  1 = photo of a screen (recapture / fraud)

Method: physics-driven features (moire/FFT, high-frequency residual, JPEG
double-compression blockiness) -> StandardScaler -> trained classifier.
Feature extraction is shared with train.py (features.py), so the model sees
identical numbers in training and inference.

Run `python train.py` first to produce model.joblib.
"""

import sys
from pathlib import Path

import joblib
import numpy as np

from features import extract_features

MODEL_PATH = Path(__file__).parent / "model.joblib"

# model.joblib is a LOCAL, self-generated artifact (produced by train.py on this
# machine), so loading it via joblib/pickle is trusted by construction.
_BUNDLE = None


def _load_bundle():
    global _BUNDLE
    if _BUNDLE is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"{MODEL_PATH.name} not found -- run `python train.py` first."
            )
        _BUNDLE = joblib.load(MODEL_PATH)
    return _BUNDLE


def predict(image_path: str) -> float:
    """Return P(photo-of-a-screen) in [0, 1] for one image."""
    bundle = _load_bundle()
    vec, names = extract_features(image_path)

    # Guard against train/serve feature drift.
    if names != bundle["feature_names"]:
        raise RuntimeError("feature mismatch between predict.py and the saved model")

    x = bundle["scaler"].transform(vec.reshape(1, -1))
    proba = bundle["model"].predict_proba(x)[0, 1]
    return float(np.clip(proba, 0.0, 1.0))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python predict.py image.jpg", file=sys.stderr)
        sys.exit(1)
    print(predict(sys.argv[1]))
