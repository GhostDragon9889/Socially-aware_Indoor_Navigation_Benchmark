"""CLI integrity gate; real physical training is driven by the runtime runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .manifests import validate_training_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--mode", choices=("paper_method_reconstruction", "strict_author_experiment"), default="paper_method_reconstruction")
    parser.add_argument("--evaluation-manifest", type=Path)
    parser.add_argument("--verify-package-hashes", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate_training_manifest(args.manifest, mode=args.mode,
        evaluation_manifest=args.evaluation_manifest, verify_package_hashes=args.verify_package_hashes)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
