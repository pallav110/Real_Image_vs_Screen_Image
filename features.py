"""Physics-driven features for recapture (photo-of-a-screen) detection.

SHARED by train.py and predict.py -> identical numbers at train & inference
time (zero train/serve skew).

KEY DESIGN CHOICES
  1. Analyze NATIVE-RESOLUTION crops, never a downscaled whole image.
     Moire / screen pixel-grid structure live in the finest, highest-frequency
     detail; downsampling a multi-megapixel photo low-pass-filters those
     fingerprints away. We cut CROP x CROP windows from full resolution.
  2. Sample several crops (center + corners) and aggregate by BOTH mean and max
     -- max captures localized evidence, mean cuts single-window noise.

Fingerprint families (all interpretable):
  A. LUMA RESIDUAL SPECTRUM / MOIRE -- sharp, directional peaks from periodic
     screen structure beating against the sensor grid.
  B. HIGH-FREQUENCY RESIDUAL        -- magnitude & non-Gaussianity of fine noise.
  C. JPEG DOUBLE-COMPRESSION        -- stronger 8x8 block-boundary jumps.

(Color-opponent "chroma" features were prototyped and dropped: they tripled the
FFT cost while slightly *lowering* cross-validated accuracy -- the luma spectrum
already captures the recapture signal. A good reminder that more features != better.)

Light deps (numpy + Pillow only).

Public API:
    extract_features(path) -> (np.ndarray vector, list[str] names)
    FEATURE_NAMES          -> list[str]  (stable order)
    CROP                   -> int
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

# Native-resolution analysis window: big enough to show periodic structure,
# small enough to be fast. Cropped from the FULL-RES image (no downscaling).
CROP = 768


# --------------------------------------------------------------------------- #
# 0. Native-resolution crops
# --------------------------------------------------------------------------- #
def iter_crops(path: str):
    """Yield CROP x CROP RGB float32 windows at NATIVE resolution.

    Large image -> center + 4 corners (fine detail preserved).
    Small image -> the largest square, upscaled once (only ever upsample, so we
    never low-pass away existing high-frequency fingerprints)."""
    img = Image.open(path).convert("RGB")
    w, h = img.size

    if min(w, h) < CROP:
        side = min(w, h)
        left, top = (w - side) // 2, (h - side) // 2
        sq = img.crop((left, top, left + side, top + side))
        yield np.asarray(sq.resize((CROP, CROP), Image.Resampling.BICUBIC), np.float32)
        return

    inset = int(0.05 * min(w, h))  # so corners aren't pure image edges
    positions = [
        ((w - CROP) // 2, (h - CROP) // 2),
        (inset, inset),
        (w - CROP - inset, inset),
        (inset, h - CROP - inset),
        (w - CROP - inset, h - CROP - inset),
    ]
    for left, top in positions:
        left, top = max(0, left), max(0, top)
        yield np.asarray(img.crop((left, top, left + CROP, top + CROP)), np.float32)


# --------------------------------------------------------------------------- #
# Shared signal helpers
# --------------------------------------------------------------------------- #
def _highpass(plane, sigma=1.0):
    """Residual = detail above a Gaussian blur (where periodic structure lives)."""
    p_img = Image.fromarray(np.clip(plane, 0, 255).astype(np.uint8))
    blurred = np.asarray(p_img.filter(ImageFilter.GaussianBlur(sigma)), np.float32)
    return plane - blurred


# Geometry (window, radius mask, angle bins) depends only on crop shape, so we
# build it ONCE per shape and reuse across all crops/images -- this is identical
# math, just not recomputed 5x per image. Big speedup, zero change to values.
_GEOM = {}


def _geom(shape):
    g = _GEOM.get(shape)
    if g is None:
        cy, cx = shape[0] // 2, shape[1] // 2
        y, x = np.indices(shape)
        r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
        win = np.hanning(shape[0])[:, None] * np.hanning(shape[1])[None, :]
        outer = r > 0.10 * cy
        ang_outer = (np.arctan2(y - cy, x - cx)[outer] / np.pi * 6).astype(int) % 6
        g = {"win": win, "outer": outer, "ang_outer": ang_outer}
        _GEOM[shape] = g
    return g


# --------------------------------------------------------------------------- #
# A. Luma residual spectrum / moire (the heart)
# --------------------------------------------------------------------------- #
def spectrum_features(gray):
    g = _geom(gray.shape)
    hp = _highpass(gray, sigma=1.0)
    F = np.abs(np.fft.fftshift(np.fft.fft2(hp * g["win"])))
    spec = F[g["outer"]]
    mean = spec.mean() + 1e-6

    peak_ratio = float(spec.max() / mean)                  # peakiness
    thresh = spec.mean() + 4.0 * spec.std()
    peak_count = float((spec > thresh).mean())             # fraction strong peaks
    top_energy = float(np.sort(spec)[-50:].mean() / mean)  # top-50 peak height
    high_frac = float((spec ** 2).sum() / ((F ** 2).sum() + 1e-6))

    # Directional anisotropy: moire favors orientations -> uneven angular energy.
    ang = g["ang_outer"]
    bins = np.array([spec[ang == k].mean() for k in range(6)])
    anisotropy = float(bins.std() / (bins.mean() + 1e-6))

    return (
        [peak_ratio, peak_count, top_energy, high_frac, anisotropy],
        ["spec_peak_ratio", "spec_peak_count", "spec_top_energy",
         "spec_high_frac", "spec_anisotropy"],
    )


# --------------------------------------------------------------------------- #
# B. High-frequency residual statistics
# --------------------------------------------------------------------------- #
def residual_features(gray):
    res = _highpass(gray, sigma=2.0)
    res_std = float(res.std())
    r = res.ravel()
    r = (r - r.mean()) / (r.std() + 1e-6)
    res_kurtosis = float(np.mean(r ** 4))
    return ([res_std, res_kurtosis], ["hf_std", "hf_kurtosis"])


# --------------------------------------------------------------------------- #
# C. JPEG double-compression blockiness (8x8 grid)
# --------------------------------------------------------------------------- #
def blockiness_features(gray):
    dx = np.abs(np.diff(gray, axis=1))
    dy = np.abs(np.diff(gray, axis=0))
    cols = np.arange(dx.shape[1])
    rows = np.arange(dy.shape[0])
    on_x = dx[:, (cols % 8) == 7].mean()
    off_x = dx[:, (cols % 8) != 7].mean()
    on_y = dy[(rows % 8) == 7, :].mean()
    off_y = dy[(rows % 8) != 7, :].mean()
    block_ratio = float((on_x + on_y) / (off_x + off_y + 1e-6))
    return ([block_ratio], ["jpeg_block_ratio"])


# --------------------------------------------------------------------------- #
# Public assembly
# --------------------------------------------------------------------------- #
_BASE_NAMES = (
    ["spec_peak_ratio", "spec_peak_count", "spec_top_energy", "spec_high_frac",
     "spec_anisotropy"]
    + ["hf_std", "hf_kurtosis"]
    + ["jpeg_block_ratio"]
)


def _features_one_crop(rgb):
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    values = []
    for vals, _ in (
        spectrum_features(gray),
        residual_features(gray),
        blockiness_features(gray),
    ):
        values.extend(vals)
    return np.asarray(values, dtype=np.float32)


def extract_features(path: str):
    """Return (feature_vector float32, feature_names) for one image.

    Each base feature is computed per native-resolution crop, then aggregated
    across crops with BOTH mean and max (localized + global views)."""
    vecs = np.array([_features_one_crop(rgb) for rgb in iter_crops(path)])
    agg = np.concatenate([vecs.mean(axis=0), vecs.max(axis=0)])
    return agg.astype(np.float32), FEATURE_NAMES


FEATURE_NAMES = (
    [f"{n}_mean" for n in _BASE_NAMES] + [f"{n}_max" for n in _BASE_NAMES]
)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        vec, names = extract_features(sys.argv[1])
        for nm, v in zip(names, vec):
            print(f"{nm:22s} {v: .4f}")
    else:
        ds = Path(__file__).parent / "dataset"
        for cls in ("real", "screen"):
            folder = ds / cls
            imgs = sorted(folder.glob("*.jpg"))[:3] if folder.is_dir() else []
            print(f"\n--- {cls} (first {len(imgs)}) ---")
            for p in imgs:
                vec, _ = extract_features(str(p))
                print(p.name, np.round(vec, 3))
