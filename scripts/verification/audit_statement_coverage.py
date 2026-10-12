"""Inventory all owned Python statements or enforce their execution gate."""

from pathlib import Path
import sys

ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / ".social-navigation-root").is_file()
)
sys.path.insert(0, str(ROOT / "src"))

from social_navigation.coverage_audit import main  # noqa: E402 - initialize checkout imports first

if __name__ == "__main__":
    raise SystemExit(main())
