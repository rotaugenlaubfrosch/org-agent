from __future__ import annotations

from collections.abc import Callable

ProgressCallback = Callable[[str, str], None]


def report(progress: ProgressCallback | None, step: str, message: str) -> None:
    if progress is not None:
        progress(step, message)
