"""SDK-free application configuration and local resource checks.

CrowdSim remains a source-checkout application; this package does not bundle
external weights, licensed body models, datasets, or an Isaac Sim installation.
"""

from .configuration import ConfigurationError, load_amp_profile, parse_amp_args

__all__ = ["ConfigurationError", "load_amp_profile", "parse_amp_args"]
