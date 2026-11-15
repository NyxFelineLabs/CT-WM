"""Portable run-local copies of versioned, validation-selected configurations."""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]

def seed_config(relative, destination):
    destination = Path(destination)
    if not destination.exists():
        source = ROOT / 'configs' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
