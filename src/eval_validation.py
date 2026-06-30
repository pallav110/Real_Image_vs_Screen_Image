"""Evaluate predict.py on the held-out validation set (brand-new images).

Run:
    python src/eval_validation.py

dataset/validation/ holds same-scene pairs shot AFTER the model was frozen:
    real/test_NN.jpeg   <->   screen/fake_NN.jpeg   (same scene, real vs off-screen)
None of these were used in training or cross-validation, so this is a genuine
blind test -- handy to run live in an interview.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on path
import predict as P

V = Path(__file__).resolve().parent.parent / "dataset" / "validation"


def pid(p):  # the NN in test_NN / fake_NN
    return p.stem.split("_")[-1]


def main():
    if not (V / "real").is_dir():
        print(f"No validation set at {V}")
        sys.exit(1)

    reals = {pid(p): p for p in sorted((V / "real").glob("*"))}
    screens = {pid(p): p for p in sorted((V / "screen").glob("*"))}

    print(f"{'pair':>4}  {'REAL (test_)':>16}  {'SCREEN (fake_)':>16}")
    print("-" * 44)
    rc = sc = rt = st = 0
    for n in sorted(set(reals) | set(screens)):
        rs = P.predict(str(reals[n])) if n in reals else None
        ss = P.predict(str(screens[n])) if n in screens else None
        if rs is not None:
            rt += 1; rc += rs < 0.5
        if ss is not None:
            st += 1; sc += ss >= 0.5
        rtxt = (f"{rs:.3f} {'ok' if rs < 0.5 else 'MISS'}") if rs is not None else "--"
        stxt = (f"{ss:.3f} {'ok' if ss >= 0.5 else 'MISS'}") if ss is not None else "--"
        print(f"{n:>4}  {rtxt:>16}  {stxt:>16}")

    print("-" * 44)
    tot, cor = rt + st, rc + sc
    print(f"real   {rc}/{rt}   screen {sc}/{st}")
    print(f"BLIND-TEST ACCURACY: {cor}/{tot} = {cor / tot * 100:.1f}%")


if __name__ == "__main__":
    main()
