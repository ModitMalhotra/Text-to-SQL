import hashlib
import json
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "llm"


def make_key(model, prompt, run):
    raw = json.dumps({"model": model, "prompt": prompt, "run": run}, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def load(key):
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def save(key, value):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / f"{key}.json").write_text(json.dumps(value))
