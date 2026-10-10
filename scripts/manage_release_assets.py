"""CLI for the checked private pedestrian/robot/four-scene release."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from social_navigation.asset_release import main

if __name__ == '__main__':
    raise SystemExit(main(default_root=ROOT))
