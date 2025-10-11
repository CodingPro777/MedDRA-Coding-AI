import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import yaml


def load_yaml(path: Path) -> Dict:
    """Load YAML configuration from disk."""
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def dump_json(path: Path, payload: Dict) -> None:
    """Write a JSON document with UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def read_json(path: Path) -> Dict:
    """Read JSON document from disk."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_version_folder(folder_name: str) -> Dict[str, str]:
    """Split a MedDRA directory name into language and version components."""
    if "__" in folder_name:
        # Already formatted as language__version
        parts = folder_name.split("__", 1)
        return {"language": parts[0], "version": parts[1]}

    if "_" not in folder_name:
        return {"language": folder_name, "version": "unknown"}

    language, version = folder_name.split("_", 1)
    return {"language": language, "version": version}


def clean_field(value: Optional[str]) -> str:
    """Normalize raw ASC field strings."""
    if value is None:
        return ""
    cleaned = value.strip().strip('"').strip()
    return cleaned


def ensure_required_files(version_dir: Path, filenames: Iterable[str]) -> None:
    """Validate that the required MedDRA ASCII files exist."""
    missing = [fname for fname in filenames if not (version_dir / fname).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required MedDRA files in {version_dir}: {', '.join(missing)}"
        )
