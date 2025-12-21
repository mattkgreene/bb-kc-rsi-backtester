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
    database_url: str | None
    redis_url: str | None

    backend_db_path: str | Path
    discovery_db_path: str | Path
    market_db_path: str | Path

    worker_poll_seconds: float
    worker_id: str

    use_spark: bool
    spark_master: str | None


def get_settings() -> Settings:
    repo_root = _repo_root()
    database_url = os.getenv("DATABASE_URL")
    redis_url = os.getenv("REDIS_URL")

    data_dir = repo_root / "data"
    if not database_url:
        data_dir.mkdir(parents=True, exist_ok=True)

    if database_url:
        backend_db_path: str | Path = os.getenv("JOBS_DATABASE_URL") or database_url
        discovery_db_path: str | Path = os.getenv("DISCOVERY_DATABASE_URL") or database_url
        market_db_path: str | Path = os.getenv("MARKET_DATABASE_URL") or database_url
    else:
        backend_db_path = _env_path("BACKEND_DB_PATH", data_dir / "backend.db")
        discovery_db_path = _env_path("DISCOVERY_DB_PATH", data_dir / "discovery.db")
        market_db_path = _env_path("MARKET_DB_PATH", data_dir / "market_data.db")

    return Settings(
        database_url=database_url,
        redis_url=redis_url,
        backend_db_path=backend_db_path,
        discovery_db_path=discovery_db_path,
        market_db_path=market_db_path,
        worker_poll_seconds=float(os.getenv("WORKER_POLL_SECONDS", "1.0")),
        worker_id=os.getenv("WORKER_ID", uuid.uuid4().hex[:10]),
        use_spark=_env_bool("USE_SPARK", False),
        spark_master=os.getenv("SPARK_MASTER"),
    )
