"""Project paths and environment. All code written during the sprint (6-8 Nov 2026)."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

# Large files live off OneDrive, on the D: drive, as requested.
DOWNLOAD_DIR = Path(os.environ.get("APART_DOWNLOAD_DIR", r"D:\APART Downloads"))
DATA_DIR = DOWNLOAD_DIR / "data"
RUNS_DIR = Path(os.environ.get("APART_RUNS_DIR", DOWNLOAD_DIR / "runs"))
HF_CACHE = DOWNLOAD_DIR / "hf_cache"
RESULTS_DIR = ROOT / "results"

for _d in (DATA_DIR, RUNS_DIR, HF_CACHE, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_HOME", str(HF_CACHE))

MAIN_MODEL = os.environ.get("MAIN_MODEL", "deepseek-v4-flash")
SECOND_MODEL = os.environ.get("SECOND_MODEL", "qwen3.8-27b")
THIRD_MODEL = os.environ.get("THIRD_MODEL", "gemma-4-31b-it")
STRONG_MODEL = os.environ.get("STRONG_MODEL", "glm-5.2")

# Credits per million tokens, from docs.adaptionlabs.ai/inference-stacks/models (6 Nov 2026).
PRICES = {
    "deepseek-v4-flash": (0.04444, 0.001414, 0.13332),
    "qwen3.8-27b": (0.039, 0.0039, 1.25),
    "gemma-4-31b-it": (0.09, 0.05, 0.34),
    "gemma-4-26b-a4b-it": (0.0765, 0.0425, 0.255),
    "glm-53-flash": (0.15, 0.03, 0.50),
    "qwen3.5-397b": (0.55, 0.225, 3.50),
    "glm-5.2": (1.40, 0.26, 4.40),
}


def cost(model, input_tokens, output_tokens, cached_tokens=0):
    inp, cached, out = PRICES.get(model, (0.0, 0.0, 0.0))
    return ((input_tokens - cached_tokens) * inp + cached_tokens * cached + output_tokens * out) / 1e6


def adaption_client(timeout=180.0):
    """Adaption SDK client. Default read timeout raised to 180 s: large reasoning models
    (glm-5.2) can exceed the SDK default on a cold first call."""
    from adaption import Adaption
    key = os.environ.get("ADAPTION_API_KEY")
    if not key:
        raise RuntimeError("ADAPTION_API_KEY missing: put it in .env")
    return Adaption(api_key=key, timeout=timeout)
