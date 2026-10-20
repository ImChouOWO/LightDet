"""Explicit configuration for the gRefCOCO-only branch."""
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[3]


def resolve(path):
    path=Path(path)
    return path if path.is_absolute() else ROOT/path


def load_config(path):
    with resolve(path).open() as stream:
        return yaml.safe_load(stream)
