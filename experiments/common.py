"""Shared helpers: where results go and how tables are written."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

# Korean Windows consoles default to cp949, which cannot print ≈, α, ×, …
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "results"
FIGURES = ROOT / "docs" / "figures"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)


def md_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def write_result(name: str, title: str, body: str) -> None:
    text = f"# {title}\n\n{body}\n"
    (RESULTS / f"{name}.md").write_text(text, encoding="utf-8")
    print(text)
