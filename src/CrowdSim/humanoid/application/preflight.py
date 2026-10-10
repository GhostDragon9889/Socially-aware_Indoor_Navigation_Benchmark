"""Run the AMP entry's local-file/configuration gate without starting Kit.

Usage: python -m CrowdSim.humanoid.application.preflight <AMP CLI arguments>
The same flags and output-collision checks as the real runner are applied.
"""
from __future__ import annotations

import json
import sys

from .configuration import ConfigurationError, load_amp_profile, parse_amp_args
from .resources import ResourceError, preflight_amp_resources


def main(argv=None):
    args = parse_amp_args(argv)
    try:
        profile = load_amp_profile(args.config)
        report = preflight_amp_resources(args, profile)
    except (ConfigurationError, ResourceError, OSError) as error:
        print(f"AMP resource preflight failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"arguments": vars(args), "profile_mode": profile["mode"],
                      "resource_preflight": report.as_dict()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
