from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
CACHE_DIR = PROJECT_ROOT / "cache"
MODELS_CONFIG_PATH = CONFIG_DIR / "models.json"
GUIDELINE_PATH = CONFIG_DIR / "TG263_nomenclature.xlsx"
PROMPT_GLOB = "prompt*.txt"

APP_NAME = "rt-rename"
APP_TITLE = "RT-Rename"
APP_VERSION = "0.3"
DEFAULT_HOST = os.environ.get("HOST", "0.0.0.0")
DEFAULT_PORT = int(os.environ.get("PORT", "8055"))
DEFAULT_GUIDELINE = "TG263"
DEFAULT_MODEL_DISPLAY = "Llama 3.1 | 70B | local"
DEFAULT_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "ollama:11434")
STATUS_IDLE = "idle"

ROW_FIELDS = (
    "local name",
    "TG263 name",
    "confidence",
    "verify",
    "accept",
    "comment",
    "raw output",
    "timestamp",
)

TARGET_VOLUME_MARKERS = ("PTV", "GTV", "CTV", "ITV")
EXCLUDED_NRRD_SUFFIXES = (
    "_stitched.nrrd",
    "_s2_def.nrrd",
    "_s2.nrrd",
)

MODEL_SYSTEM_PROMPT = (
    "You are a radiation oncology professional with vast experience in naming "
    "structures for radiotherapy treatment planning. You understand English, "
    "German and Dutch. You are tasked with renaming structures based on a "
    "standardized nomenclature list. Follow the prompts strictly and do not "
    "provide any additional information."
)
