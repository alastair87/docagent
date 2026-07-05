from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from autoscribe.models import AppConfig


def load_config(config_path: Path | None = None, overrides: dict[str, Any] | None = None) -> AppConfig:
    load_dotenv()
    data: dict[str, Any] = {}
    if config_path and config_path.exists():
        data.update(json.loads(config_path.read_text(encoding="utf-8")))

    env_map = {
        "AUTOSCRIBE_BACKEND": "llm_backend",
        "AUTOSCRIBE_MODEL": "model",
        "AUTOSCRIBE_BASE_URL": "openai_base_url",
        "AUTOSCRIBE_DATA_DIR": "data_dir",
    }
    for env_name, key in env_map.items():
        value = os.getenv(env_name)
        if value:
            data[key] = value

    if overrides:
        data.update({key: value for key, value in overrides.items() if value is not None})
    return AppConfig(**data)
