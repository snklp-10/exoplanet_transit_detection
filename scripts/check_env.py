"""Verify the environment: imports, versions, device, a synthetic BLS recovery, optional archive access.

Usage:
    python scripts/check_env.py            # offline checks
    python scripts/check_env.py --network  # also test MAST / NASA Exoplanet Archive reachability
"""

from __future__ import annotations

import argparse
import importlib
import sys

REQUIRED = [
    "numpy", "scipy", "pandas", "pyarrow", "astropy", "astroquery",
    "lightkurve", "torch", "sklearn", "matplotlib", "yaml", "tqdm",
]
OPTIONAL = ["xgboost", "wotan", "tensorboard", "pytest"]


def check_imports() -> bool:
    ok = True
    print("== Imports ==")
    for name in REQUIRED:
        try:
            m = importlib.import_module(name)
            print(f"  ok      {name:<12} {getattr(m, '__version__', '')}")
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"  MISSING {name:<12} ({type(e).__name__}: {e})")
    for name in OPTIONAL:
        try:
            m = importlib.import_module(name)
            print(f"  ok      {name:<12} {getattr(m, '__version__', '')} (optional)")
        except Exception:  # noqa: BLE001
            print(f"  absent  {name:<12} (optional)")
    return ok


def check_device() -> None:
    print("== Device ==")
    try:
        import torch

        print(f"  torch {torch.__version__}, cuda available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  gpu: {torch.cuda.get_device_name(0)}")
        from exotransit.utils import get_device

        print(f"  selected device: {get_device()}")
    except Exception as e:  # noqa: BLE001
        print(f"  device check failed: {e}")


def check_bls_recovery() -> bool:
    """Inject a transit into noise and confirm astropy BLS recovers the period."""
    print("== Synthetic BLS recovery ==")
    import numpy as np
    from astropy.timeseries import BoxLeastSquares

    rng = np.random.default_rng(0)
    # ~90 days at Kepler long cadence (29.4 min)
    t = np.arange(0, 90, 29.4 / 60 / 24)
    true_period, true_t0, duration, depth = 7.3, 2.1, 0.15, 0.002
    flux = 1 + rng.normal(0, 3e-4, t.size)
    in_transit = ((t - true_t0 + 0.5 * true_period) % true_period - 0.5 * true_period)
    flux[np.abs(in_transit) < duration / 2] -= depth

    bls = BoxLeastSquares(t, flux)
    periods = np.linspace(1.0, 20.0, 20000)
    res = bls.power(periods, duration)
    best = periods[np.argmax(res.power)]
    ok = abs(best - true_period) / true_period < 0.01
    print(f"  true period {true_period} d, recovered {best:.3f} d -> {'ok' if ok else 'FAIL'}")
    return ok


def check_cnn_forward() -> bool:
    """Tiny two-branch 1D CNN forward/backward pass at the configured view sizes."""
    print("== 1D CNN forward/backward ==")
    import torch
    from torch import nn

    from exotransit.utils import load_config, seed_everything

    cfg = load_config()
    seed_everything(cfg["seed"])
    g, l = cfg["preprocess"]["global_bins"], cfg["preprocess"]["local_bins"]

    def branch() -> nn.Sequential:
        return nn.Sequential(
            nn.Conv1d(1, 8, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(8, 16, 5, padding=2), nn.ReLU(), nn.AdaptiveAvgPool1d(4), nn.Flatten(),
        )

    gb, lb, head = branch(), branch(), nn.Linear(16 * 4 * 2, 1)
    xg, xl = torch.randn(4, 1, g), torch.randn(4, 1, l)
    logits = head(torch.cat([gb(xg), lb(xl)], dim=1))
    logits.sum().backward()
    ok = logits.shape == (4, 1)
    print(f"  global {g} + local {l} -> logits {tuple(logits.shape)} -> {'ok' if ok else 'FAIL'}")
    return ok


def check_network() -> bool:
    print("== Network (MAST / NASA Exoplanet Archive) ==")
    import urllib.request

    ok = True
    for name, url in [
        ("NASA Exoplanet Archive", "https://exoplanetarchive.ipac.caltech.edu/TAP/capabilities"),
        ("MAST", "https://mast.stsci.edu/api/v0/"),
    ]:
        try:
            with urllib.request.urlopen(url, timeout=15) as r:
                print(f"  ok      {name} (HTTP {r.status})")
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"  BLOCKED {name}: {type(e).__name__}: {e}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", action="store_true", help="also test archive reachability")
    args = ap.parse_args()

    print(f"Python {sys.version.split()[0]}\n")
    results = {"imports": check_imports()}
    if results["imports"]:
        check_device()
        results["bls"] = check_bls_recovery()
        results["cnn"] = check_cnn_forward()
    if args.network:
        results["network"] = check_network()

    print("\n== Summary ==")
    for k, v in results.items():
        print(f"  {k:<8} {'PASS' if v else 'FAIL'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
