"""Measure the two REQUIRED numbers: latency and cost per image.

Run:
    python benchmark.py            # samples images from dataset/
    python benchmark.py img1 img2  # benchmark specific images

Reports:
  - model-load time (one-off)
  - per-image latency: median / p90 (ms), on this device
  - cost per image: on-device (free) + a cloud CPU estimate ($/1k, $/1M)

It prints a copy-paste-ready block for note.md.
"""

import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on path
import predict as P

ROOT = Path(__file__).resolve().parent.parent  # repo root (this file lives in src/)
DATA_DIR = ROOT / "dataset"
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# --- Cloud cost assumption (edit to taste) -------------------------------- #
# A small commodity CPU box, e.g. ~2 vCPU on-demand. Adjust to your provider.
CLOUD_USD_PER_HOUR = 0.05
WARMUP = 3


def sample_images(n=60):
    imgs = []
    for cls in ("real", "screen"):
        folder = DATA_DIR / cls
        if folder.is_dir():
            imgs += [p for p in sorted(folder.iterdir()) if p.suffix.lower() in EXTS]
    return imgs[:: max(1, len(imgs) // n)][:n] if imgs else []


def main():
    paths = [Path(a) for a in sys.argv[1:]] or sample_images()
    if not paths:
        print("No images found. Put photos in dataset/ or pass paths.")
        sys.exit(1)

    # One-off model load (separated from per-image timing).
    t0 = time.perf_counter()
    P._load_bundle()
    load_ms = (time.perf_counter() - t0) * 1000

    # Warm up (JIT caches, file system, numpy) so timings are steady-state.
    for p in paths[:WARMUP]:
        P.predict(str(p))

    times = []
    for p in paths:
        t = time.perf_counter()
        P.predict(str(p))
        times.append((time.perf_counter() - t) * 1000)
    times = np.array(times)

    median = float(np.median(times))
    p90 = float(np.percentile(times, 90))
    imgs_per_hour = 3600.0 / (median / 1000.0)
    usd_per_1k = CLOUD_USD_PER_HOUR / imgs_per_hour * 1000
    usd_per_1m = usd_per_1k * 1000

    device = f"{platform.system()} {platform.machine()}, CPU only"

    print("\n================ BENCHMARK ================")
    print(f"images timed     : {len(times)}")
    print(f"device           : {device}")
    print(f"model load (1x)  : {load_ms:.0f} ms")
    print(f"latency median   : {median:.1f} ms/image")
    print(f"latency p90      : {p90:.1f} ms/image")
    print("\n--- cost per image ---")
    print("on-device        : ~free (no API, no GPU)")
    print(f"cloud CPU est.   : ${usd_per_1k:.4f} / 1,000   (${usd_per_1m:.2f} / 1,000,000)")
    print(f"  assumes ${CLOUD_USD_PER_HOUR:.3f}/hr CPU, {imgs_per_hour:,.0f} images/hr")

    print("\n--- paste into note.md ---")
    print(f"Latency: ~{median:.0f} ms/image (median) on {device}.")
    print(f"Cost: ~free on-device; ~${usd_per_1k:.3f} per 1,000 images on a "
          f"${CLOUD_USD_PER_HOUR:.3f}/hr CPU instance.")


if __name__ == "__main__":
    main()
