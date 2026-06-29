"""Generate research-paper-style figures into figures/.

Run (needs matplotlib, a figures-only extra dependency):
    python src/make_figures.py

Produces:
  fig1_feature_separation.png  single-feature class separation (AUC)
  fig2_score_distribution.png  out-of-fold score histogram, real vs screen
  fig3_performance.png         ROC curve + confusion matrix
  fig4_whatsapp_confound.png   accuracy by source x class (the data-quality finding)
  fig5_spectrum_moire.png      residual FFT of a real vs a screen crop (the physics)
  fig6_examples.png            example images with predicted scores

All figures use the cached features + the same WhatsApp-screen filter as train.py.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on path
import predict as P
from features import FEATURE_NAMES, _highpass, iter_crops
from train import build_dataset, is_whatsapp, SEED

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "figures"
FIGS.mkdir(exist_ok=True)
REAL, SCREEN = "#34a853", "#ea4335"
plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.axisbelow": True})


def pipe():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=1000, class_weight="balanced"))


def oof_proba(X, y):
    return cross_val_predict(pipe(), X, y,
                             cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                             method="predict_proba")[:, 1]


# --------------------------------------------------------------------------- #
def fig_feature_separation(Xk, yk):
    aucs = []
    for i, n in enumerate(FEATURE_NAMES):
        a = roc_auc_score(yk, Xk[:, i]); aucs.append((max(a, 1 - a), n))
    aucs.sort()
    vals = [a for a, _ in aucs]; labs = [n for _, n in aucs]
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["#2a6df4" if v >= 0.85 else "#9aa0a6" if v >= 0.7 else "#c8ccd1" for v in vals]
    ax.barh(labs, vals, color=colors)
    ax.axvline(0.5, color="k", ls="--", lw=1, label="chance (0.5)")
    ax.set_xlim(0.5, 1.0); ax.set_xlabel("single-feature ROC-AUC (class separation)")
    ax.set_title("Figure 1 — Which physical fingerprints separate the classes")
    ax.legend(loc="lower right")
    for v, l in zip(vals, labs):
        ax.text(v + 0.004, l, f"{v:.2f}", va="center", fontsize=8)
    fig.tight_layout(); fig.savefig(FIGS / "fig1_feature_separation.png"); plt.close(fig)


def fig_score_distribution(proba, yk):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bins = np.linspace(0, 1, 26)
    ax.hist(proba[yk == 0], bins, color=REAL, alpha=0.75, label="real photo")
    ax.hist(proba[yk == 1], bins, color=SCREEN, alpha=0.75, label="photo of a screen")
    ax.axvline(0.5, color="k", ls="--", lw=1, label="threshold 0.5")
    ax.set_xlabel("predicted P(photo-of-a-screen)  —  out-of-fold")
    ax.set_ylabel("# images")
    ax.set_title("Figure 2 — Out-of-fold score distribution (unseen images)")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIGS / "fig2_score_distribution.png"); plt.close(fig)


def fig_performance(proba, yk):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    RocCurveDisplay.from_predictions(yk, proba, ax=a1, name="logreg")
    a1.plot([0, 1], [0, 1], "k--", lw=1)
    a1.set_title(f"Figure 3a — ROC (AUC = {roc_auc_score(yk, proba):.3f})")
    cm = confusion_matrix(yk, (proba >= 0.5).astype(int))
    im = a2.imshow(cm, cmap="Blues")
    a2.set_xticks([0, 1], ["pred real", "pred screen"])
    a2.set_yticks([0, 1], ["true real", "true screen"])
    for (r, c), v in np.ndenumerate(cm):
        a2.text(c, r, str(v), ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black", fontsize=14)
    a2.set_title("Figure 3b — Confusion matrix (out-of-fold)")
    a2.grid(False)
    fig.colorbar(im, ax=a2, fraction=0.046)
    fig.tight_layout(); fig.savefig(FIGS / "fig3_performance.png"); plt.close(fig)


def fig_whatsapp(X, y, paths):
    proba = oof_proba(X, y)  # OOF on ALL data (incl. WhatsApp screens)
    pred = (proba >= 0.5).astype(int)
    wa = np.array([is_whatsapp(p) for p in paths])
    groups = [("real\nnative", (y == 0) & ~wa), ("real\nWhatsApp", (y == 0) & wa),
              ("screen\nnative", (y == 1) & ~wa), ("screen\nWhatsApp", (y == 1) & wa)]
    labs, accs, ns = [], [], []
    for name, m in groups:
        labs.append(name); accs.append((pred[m] == y[m]).mean()); ns.append(int(m.sum()))
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    colors = [REAL, REAL, SCREEN, SCREEN]
    bars = ax.bar(labs, accs, color=colors, alpha=0.85)
    ax.axhline(0.95, color="k", ls="--", lw=1, label="95% target")
    ax.set_ylim(0, 1.05); ax.set_ylabel("out-of-fold accuracy")
    ax.set_title("Figure 4 — WhatsApp recompression destroys the screen fingerprint")
    for b, a, n in zip(bars, accs, ns):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.02, f"{a*100:.0f}%\n(n={n})",
                ha="center", fontsize=9)
    ax.legend(loc="lower left")
    fig.tight_layout(); fig.savefig(FIGS / "fig4_whatsapp_confound.png"); plt.close(fig)


def _radial(F):
    cy, cx = F.shape[0] // 2, F.shape[1] // 2
    y, x = np.indices(F.shape)
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2).astype(int)
    rmax = min(cy, cx)
    tot = np.bincount(r.ravel(), F.ravel()); cnt = np.bincount(r.ravel())
    return tot[:rmax] / np.maximum(cnt[:rmax], 1)


def fig_spectrum(real_path, screen_path):
    def spec(path):
        gray = next(iter_crops(path))
        gray = 0.299 * gray[..., 0] + 0.587 * gray[..., 1] + 0.114 * gray[..., 2]
        hp = _highpass(gray, 1.0)
        win = np.hanning(hp.shape[0])[:, None] * np.hanning(hp.shape[1])[None, :]
        F = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(hp * win))))
        return gray, F

    gr, Fr = spec(real_path); gs, Fs = spec(screen_path)
    fig, ax = plt.subplots(2, 3, figsize=(12, 7.6))
    for row, (g, F, name, col) in enumerate(
        [(gr, Fr, "REAL photo", REAL), (gs, Fs, "PHOTO OF A SCREEN", SCREEN)]):
        ax[row, 0].imshow(g, cmap="gray"); ax[row, 0].set_title(f"{name}\n(native crop)")
        ax[row, 0].axis("off")
        ax[row, 1].imshow(F, cmap="magma")
        ax[row, 1].set_title("residual FFT (log-magnitude)\nbright off-center dots = moiré")
        ax[row, 1].axis("off")
        prof = _radial(np.expm1(F))
        ax[row, 2].plot(prof, color=col)
        ax[row, 2].set_title("radial spectrum\n(smooth=real, peaks=screen)")
        ax[row, 2].set_xlabel("spatial frequency"); ax[row, 2].set_ylabel("energy")
    fig.suptitle("Figure 5 — The physics: a screen photo concentrates energy into "
                 "periodic spectral peaks (moiré)", fontsize=11)
    fig.tight_layout(); fig.savefig(FIGS / "fig5_spectrum_moire.png"); plt.close(fig)


def fig_examples(paths, y):
    # pick a few of each class
    reals = [p for p, l in zip(paths, y) if l == 0 and "IMG_2026" in p.name][:3]
    screens = [p for p, l in zip(paths, y) if l == 1 and "IMG_2026" in p.name][:3]
    picks = reals + screens
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.4))
    for ax, p in zip(axes.ravel(), picks):
        from PIL import Image
        im = Image.open(p).convert("RGB"); im.thumbnail((360, 360))
        s = P.predict(str(p))
        verdict = "SCREEN" if s >= 0.5 else "REAL"
        col = SCREEN if s >= 0.5 else REAL
        ax.imshow(im); ax.axis("off")
        ax.set_title(f"score={s:.2f} -> {verdict}", color=col, fontsize=11)
    fig.suptitle("Figure 6 — Example predictions (score = P(photo-of-a-screen))",
                 fontsize=12)
    fig.tight_layout(); fig.savefig(FIGS / "fig6_examples.png"); plt.close(fig)


def main():
    X, y, paths = build_dataset()
    excl = np.array([is_whatsapp(p) and lab == 1 for p, lab in zip(paths, y)])
    keep = ~excl
    Xk, yk = X[keep], y[keep]
    paths_k = [p for p, e in zip(paths, excl) if not e]
    proba_k = oof_proba(Xk, yk)

    print("rendering figures ...")
    fig_feature_separation(Xk, yk)
    fig_score_distribution(proba_k, yk)
    fig_performance(proba_k, yk)
    fig_whatsapp(X, y, paths)
    # representative crisp screenshot + real for the physics figure
    screen = next(p for p in paths_k if "IMG_2026" in p.name and p.parent.name == "screen")
    real = next(p for p in paths_k if "IMG_2026" in p.name and p.parent.name == "real")
    fig_spectrum(real, screen)
    fig_examples(paths_k, yk)
    print(f"done -> {FIGS}")
    for f in sorted(FIGS.glob("*.png")):
        print("  ", f.name)


if __name__ == "__main__":
    main()
