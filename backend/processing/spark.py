from __future__ import annotations

from typing import Optional


def get_spark_session(*, app_name: str, master: str | None = None):
    """
    Best-effort SparkSession factory.

    This keeps Spark optional: if pyspark isn't installed (or Java isn't
    configured), callers receive None and should fall back to pandas/numpy.
    """
    try:
        from pyspark.sql import SparkSession  # type: ignore
    except Exception:
        return None

    builder = SparkSession.builder.appName(app_name)
    if master:
        builder = builder.master(master)
    try:
        return builder.getOrCreate()
    except Exception:
        return None


def spark_available(*, app_name: str = "bbkc-backend", master: str | None = None) -> bool:
    return get_spark_session(app_name=app_name, master=master) is not None

