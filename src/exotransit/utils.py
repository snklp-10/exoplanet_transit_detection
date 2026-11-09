"""Small shared helpers: config loading, seeding, device selection."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "config.yaml"


def load_config(path: str | os.PathLike | None = None) -> dict:
    """Load the YAML config. Relative paths inside it are relative to the repo root."""
    with open(path or DEFAULT_CONFIG) as f:
        return yaml.safe_load(f)


def resolve(cfg_path: str) -> Path:
    """Turn a config-relative path into an absolute path under the repo root."""
    p = Path(cfg_path)
    return p if p.is_absolute() else REPO_ROOT / p


def seed_everything(seed: int = 42, deterministic: bool = True) -> None:
    """Seed python, numpy and torch (if installed) for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> str:
    """Best available torch device: cuda, then mps (Apple silicon), then cpu."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
