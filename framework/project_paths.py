"""Repository-wide paths and local configuration helpers."""
from __future__ import annotations

import json
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("HTP266_DATA_DIR", REPO_ROOT / "data")).expanduser()
IMAGE_DIR = Path(os.environ.get("HTP266_IMG_DIR", DATA_DIR / "01.원천데이터")).expanduser()
LABEL_DIR = Path(os.environ.get("HTP266_LBL_DIR", DATA_DIR / "02.라벨링데이터")).expanduser()
RESULTS_DIR = REPO_ROOT / "results"
OUTPUT_DIR = Path(os.environ.get("HTP_OUTPUT_DIR", REPO_ROOT / "outputs")).expanduser()


def ensure_output_dir() -> Path:
    """Create and return the directory used for new experiment outputs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def get_openrouter_api_key() -> str:
    """Read the OpenRouter key from the environment or an ignored config file."""
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if key:
        return key

    config_path = REPO_ROOT / "config.json"
    if config_path.exists():
        with config_path.open(encoding="utf-8") as f:
            key = str(json.load(f).get("OPENROUTER_API_KEY", "")).strip()
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY를 환경변수로 설정하거나 저장소 루트의 "
            "config.json에 입력하세요."
        )
    return key
