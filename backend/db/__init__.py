# backend/db/__init__.py

import os
from dotenv import load_dotenv
from pathlib import Path
from db.postgres_repo import PostgresMetadataRepository

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH)

_repo = None

def get_repo():
    global _repo
    if _repo is None:
        dsn = os.getenv("POSTGRES_DSN")
        if not dsn:
            raise RuntimeError("POSTGRES_DSN not set")
        _repo = PostgresMetadataRepository(dsn)
    return _repo

