from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PARTS = {"tests", ".venv", "__pycache__", "frontend_v2", ".git"}
FORBIDDEN_PATTERNS = [
    re.compile(r"return\s+True\s*#\s*mock", re.IGNORECASE),
    re.compile(r"#\s*TODO:\s*implement", re.IGNORECASE),
    re.compile(r"pass\s*#\s*placeholder", re.IGNORECASE),
    re.compile(r"mock_status\s*=", re.IGNORECASE),
    re.compile(r"force\s+True", re.IGNORECASE),
]


def should_scan(path: Path) -> bool:
    return path.suffix == ".py" and not any(part in EXCLUDED_PARTS for part in path.parts)


def main() -> int:
    failures: list[str] = []
    for path in ROOT.rglob("*.py"):
        rel = path.relative_to(ROOT)
        if not should_scan(rel):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if any(pattern.search(line) for pattern in FORBIDDEN_PATTERNS):
                failures.append(f"{rel}:{lineno}: {line.strip()}")

    if failures:
        print("Production mock/placeholder patterns found:", file=sys.stderr)
        print("\n".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
