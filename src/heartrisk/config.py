"""Loads config/params.yaml and resolves paths relative to the project root."""

import os
from functools import lru_cache
from pathlib import Path

import yaml

# Source checkout by default; containers set HEARTRISK_HOME because the package
# is installed into site-packages there.
PROJECT_ROOT = Path(os.getenv("HEARTRISK_HOME", Path(__file__).resolve().parents[2]))
CONFIG_PATH = PROJECT_ROOT / "config" / "params.yaml"


@lru_cache(maxsize=1)
def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def resolve(relative: str) -> Path:
    """Turn a path from the config into an absolute path under the project root."""
    return PROJECT_ROOT / relative
