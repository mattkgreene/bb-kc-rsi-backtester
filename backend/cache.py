from __future__ import annotations

import json
from typing import Any, Optional


def get_redis_client(redis_url: str | None):
    if not redis_url:
        return None
    try:
        import redis  # type: ignore
    except Exception:  # pragma: no cover
        return None
    try:
        return redis.Redis.from_url(str(redis_url), decode_responses=True)
    except Exception:  # pragma: no cover
        return None


def redis_get_json(client, key: str) -> Optional[Any]:
    if client is None:
        return None
    try:
        raw = client.get(key)
        if not raw:
            return None
        return json.loads(raw)
    except Exception:
        return None


def redis_set_json(client, key: str, value: Any, *, ttl_seconds: int) -> None:
    if client is None:
        return
    try:
        client.setex(key, int(ttl_seconds), json.dumps(value, default=str))
    except Exception:
        return
