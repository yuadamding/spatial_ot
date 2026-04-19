from __future__ import annotations

from pathlib import Path


def test_gitignore_covers_generated_artifacts() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    gitignore = (repo_root / ".gitignore").read_text()
    required_patterns = [
        ".DS_Store",
        "__pycache__/",
        "*.pyc",
        ".pytest_cache/",
        "*.h5ad",
        "*.pt",
        "*.npz",
    ]
    for pattern in required_patterns:
        assert pattern in gitignore
