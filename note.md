# Spot the Fake Photo — Approach Note

**Task:** given one image, output a score in [0,1] (1 = photo of a screen / recapture).

## Approach (no deep learning)

A photo *of a screen* carries physical fingerprints a real first-capture photo
cannot. I measure those directly with ~10 interpretable features, then classify
with a tiny **logistic regression** (0.8 KB on disk):

1. **Moiré / residual spectrum (the core).** A screen's pixel grid beats against
   the camera sensor grid, producing periodic interference. I high-pass each crop,
   FFT it, and measure how sharply energy concentrates into peaks (peak ratio,
   top-peak height, peak count, high-frequency fraction, directional anisotropy).
2. **High-frequency residual.** Magnitude and non-Gaussianity (kurtosis) of the
   fine-detail noise, which recapture alters.
3. **JPEG double-compression blockiness.** A screenshot is JPEG-encoded once by
   the source device and again by my camera, strengthening 8×8 block-edge jumps.

**Two design choices that mattered most**

- **Analyze native-resolution crops, never a downscaled image.** Moiré lives in the
  finest detail; resizing a multi-megapixel photo low-pass-filters the fingerprints
  away. Switching from a 512px downscale to native-resolution 768px crops lifted
  balanced accuracy from **70% → 91%** with the same model. I sample 5 crops
  (center + corners) and aggregate each feature by **mean and max** (max catches
  locally-strong moiré; mean cuts single-window noise).
- **Data quality.** A source-stratified error analysis showed 14 of my "screen"
  images had been sent via **WhatsApp**, which recompresses + downscales and
  *destroys* the recapture fingerprints — they classified at 14% vs 94% for native
  screen captures (WhatsApp *real* images were unaffected at 98%, confirming the
  cause). They are corrupted labels for a fingerprint method and do not match the
  grader's direct-camera scenario, so I exclude WhatsApp-transmitted screens.

## Accuracy (5-fold stratified cross-validation, balanced accuracy)

| Setting | Accuracy | Balanced acc |
|---|---|---|
| **Shipped model** (direct-camera captures) | **96.9%** | **0.963 ± 0.017** |
| Native-only, both classes | 94.6% | 0.946 |
| All images incl. WhatsApp screens (disclosed) | 92.7% | 0.911 |

Held-out split (25%): screen recall **100%** (29/29), real recall 96.7%, ROC-AUC 0.997.
Dataset: 242 real + 117 native screen images. Logistic regression beat random
forest and gradient boosting on accuracy *and* size — the features are nearly
linearly separable, so the simplest model won.

## The two required numbers

- **Latency:** ~**494 ms / image** (median) on a Windows laptop, **CPU only, no GPU**.
  One-off model load ~0.9 s. (numpy + Pillow + scikit-learn only.)
- **Cost:** **~free on-device.** On a commodity ~$0.05/hr CPU instance: **~$0.007
  per 1,000 images** (~$7 per million).

## What I'd improve

- **Latency knob:** dropping from 5 crops to a single center crop cuts latency to
  ~150 ms at a small accuracy cost — a tunable speed/accuracy trade-off.
- **Device diversity:** my photos come from a few phones/screens; I'd collect more
  capture devices and display types (OLED/LCD/e-ink, high-DPI "retina") to harden
  generalization, since high-DPI screens at distance show the weakest moiré.
- **Robustness:** add the WhatsApp/recompression domain back as an explicit
  *augmentation-aware* case if the deployment must survive forwarded images.
- **Threshold tuning:** currently 0.5; I'd set it from the ROC to match the real
  cost trade-off between missed fraud and false alarms.

## Files

`predict.py` (interface) · `features.py` (shared feature extraction) ·
`train.py` (trains + compares 3 models, saves winner) · `benchmark.py` (latency/cost)
· `check_data.py` (dataset/confound checks) · `model.joblib` (shipped) ·
`requirements.txt`.
