"""Utility for loading project configuration from YAML."""

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load the project configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file, relative
            to the project root or as an absolute path.

    Returns:
        A dictionary containing the parsed configuration, with
        top-level keys such as ``data``, ``model``, ``training``,
        ``paths``, and ``hardware``.

    Raises:
        FileNotFoundError: If the config file does not exist at
            the given path.
        yaml.YAMLError: If the file is not valid YAML.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path.resolve()}")

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)