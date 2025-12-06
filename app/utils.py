# app/utils.py
import json
import os
import logging
import torch

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: str):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def get_device():
    """Return torch device ('cuda' if available else 'cpu')."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
