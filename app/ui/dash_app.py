from __future__ import annotations

import sys
from pathlib import Path


def _ensure_repo_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


_ensure_repo_on_path()

from frontend.ui.dash_app import *  # noqa: F403


if __name__ == "__main__":
    from frontend.ui.dash_app import main

    main()
