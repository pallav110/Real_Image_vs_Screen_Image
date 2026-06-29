"""Dataset sanity checker for the recapture detector.

Run this WHILE collecting photos to catch problems early:
    python check_data.py

It looks for images in:
    data/real/    (label 0 = real photo)
    data/screen/  (label 1 = photo of a screen / recapture)

What it reports per class:
  - image count (and class balance)
  - unreadable / corrupt files (so you can delete them)
  - resolution summary (min / median / max longest-side)
  - brightness summary (mean luma)

Why brightness matters: if `real` and `screen` have very different
brightness distributions, the model can "cheat" by learning lighting
instead of the actual recapture physics, then fail on held-out photos.
This script flags that confound so you can fix it during collection.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

DATA_DIR = Path(__file__).parent / "data"
CLASSES = {"real": 0, "screen": 1}
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".tif", ".tiff"}


def list_images(folder: Path):
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTS)


def inspect_image(path: Path):
    """Return (longest_side_px, mean_luma_0_255) or None if unreadable."""
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return None
    # Downscale for a fast brightness estimate; resolution is read from full size.
    longest = max(img.size)
    small = img.copy()
    small.thumbnail((256, 256))
    arr = np.asarray(small, dtype=np.float32)
    # Rec. 601 luma
    luma = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    return longest, float(luma.mean())


def summarize(name: str, paths):
    print(f"\n=== {name}/  ({len(paths)} files) ===")
    if not paths:
        print("  (no images found)")
        return None

    res, lum, bad = [], [], []
    for p in paths:
        info = inspect_image(p)
        if info is None:
            bad.append(p.name)
            continue
        res.append(info[0])
        lum.append(info[1])

    if bad:
        print(f"  ! {len(bad)} unreadable file(s): {', '.join(bad[:8])}"
              + (" ..." if len(bad) > 8 else ""))

    if not res:
        print("  (no readable images)")
        return None

    res, lum = np.array(res), np.array(lum)
    print(f"  readable    : {len(res)}")
    print(f"  resolution  : min {res.min()}  median {int(np.median(res))}  max {res.max()}  (longest side, px)")
    print(f"  brightness  : mean {lum.mean():.1f}  std {lum.std():.1f}  (0-255 luma)")
    return {"count": len(res), "luma_mean": lum.mean(), "luma_std": lum.std()}


def main():
    if not DATA_DIR.is_dir():
        print(f"No data/ folder at: {DATA_DIR}")
        print("Create data/real/ and data/screen/ and add photos.")
        sys.exit(1)

    stats = {}
    for name in CLASSES:
        stats[name] = summarize(name, list_images(DATA_DIR / name))

    # --- Confound + balance warnings -------------------------------------
    print("\n=== checks ===")
    real, screen = stats.get("real"), stats.get("screen")
    if real and screen:
        # Class balance
        ratio = max(real["count"], screen["count"]) / max(1, min(real["count"], screen["count"]))
        if ratio > 1.5:
            print(f"  ! imbalance: {real['count']} real vs {screen['count']} screen "
                  f"({ratio:.1f}x). Aim for roughly even classes.")
        else:
            print(f"  ok balance: {real['count']} real vs {screen['count']} screen")

        # Brightness confound: how separated are the two brightness distributions?
        # A crude effect-size: difference of means in pooled-std units.
        pooled = np.sqrt((real["luma_std"] ** 2 + screen["luma_std"] ** 2) / 2) + 1e-6
        sep = abs(real["luma_mean"] - screen["luma_mean"]) / pooled
        print(f"  brightness separation (effect size): {sep:.2f}")
        if sep > 0.8:
            print("  ! WARNING: real vs screen differ a lot in brightness. The model may")
            print("    learn lighting instead of recapture. Shoot more overlapping examples")
            print("    (bright screens, dim real scenes) to break this confound.")
        else:
            print("  ok: brightness distributions overlap reasonably.")
    else:
        print("  (need both classes populated to run confound checks)")


if __name__ == "__main__":
    main()
