import os
import tomllib
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parents[1]   # .../backend
PROJECT_DIR = BACKEND_DIR.parent                    # project root

load_dotenv(PROJECT_DIR / ".env")                   # secrets (project root)
load_dotenv(BACKEND_DIR / ".env")                   # or backend/.env

SETTINGS = tomllib.loads((BACKEND_DIR / "settings.toml").read_text(encoding="utf-8"))

DATA_DIR = Path(os.getenv("DATA_DIR", SETTINGS.get("data_dir", "data")))
MAX_PAGES = SETTINGS["limits"]["max_pages"]
MAX_UPLOAD_MB = SETTINGS["limits"]["max_upload_mb"]
TOP_K = SETTINGS["limits"]["top_k"]

VISION_MODEL = SETTINGS["ingestion"]["vision_model"]
EMBEDDING_MODEL = SETTINGS["ingestion"]["embedding_model"]
PINECONE_INDEX_NAME = SETTINGS["ingestion"]["pinecone_index_name"]

MODELS = SETTINGS["models"]
DEFAULT_MODEL = MODELS[0]["key"]

REQUIRED_SECRETS = ["GROQ_API_KEY", "PINECONE_API_KEY"]


def require_secrets():
    missing = [k for k in REQUIRED_SECRETS if not os.getenv(k)]
    if missing:
        raise RuntimeError(
            f"Missing secrets: {', '.join(missing)}. "
            "Copy .env.example to .env and fill them in."
        )


def get_model(key: str | None) -> dict:
    key = key or DEFAULT_MODEL
    for m in MODELS:
        if m["key"] == key:
            return m
    raise KeyError(key)


def doc_dir(doc_id: str) -> Path:
    p = DATA_DIR / "docs" / doc_id
    (p / "images").mkdir(parents=True, exist_ok=True)
    return p