from __future__ import annotations

from pathlib import Path


def find_project_root(start: Path | None = None, max_levels: int = 5) -> Path:
    """
    Walk up directories until finding the project root.

    Root detection rules (in order):
    - directory contains README.md, or
    - directory contains .git, or
    - running from src/notebooks and the grandparent contains README.md
    """
    current = (start or Path.cwd()).resolve()

    for _ in range(max_levels):
        if (current / "README.md").exists() or (current / ".git").exists():
            return current

        # If running from .../src/notebooks, the project root is two levels up.
        if current.name == "notebooks" and (current.parent.parent / "README.md").exists():
            return current.parent.parent

        if current.parent == current:
            break
        current = current.parent

    # Fallback: preserve previous behavior
    return (start or Path.cwd()).parent.parent

