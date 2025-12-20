from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    if not value:
        return default
    return Path(value).expanduser()


@dataclass(frozen=True)
class Settings:
    backend_db_path: Path
    discovery_db_path: Path
    market_db_path: Path

    worker_poll_seconds: float
    worker_id: str

    use_spark: bool
    spark_master: str | None


def get_settings() -> Settings:
    repo_root = _repo_root()
    data_dir = repo_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    return Settings(
        backend_db_path=_env_path("BACKEND_DB_PATH", data_dir / "backend.db"),
        discovery_db_path=_env_path("DISCOVERY_DB_PATH", data_dir / "discovery.db"),
        market_db_path=_env_path("MARKET_DB_PATH", data_dir / "market_data.db"),
        worker_poll_seconds=float(os.getenv("WORKER_POLL_SECONDS", "1.0")),
        worker_id=os.getenv("WORKER_ID", uuid.uuid4().hex[:10]),
        use_spark=_env_bool("USE_SPARK", False),
        spark_master=os.getenv("SPARK_MASTER"),
    )

