"""Rebase only Maps source/default reader from the CPU-tested union."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "maps_interact_mixed_training_staging_v1/mixed_corpus.py"
source = BASE.read_text()
text = source
replacements = [
    ('from interaction_corpus import InteractionORCACorpus  # noqa: E402',
     'from interaction_corpus import InteractionORCACorpus  # noqa: E402\nfrom official_maps_corpus import OfficialMapsORCACorpus  # noqa: E402'),
    ('DEFAULT_MAPS = ROOT / "data/diffusion_amp/orca_maps_expanded_1000_v1/manifest.json"',
     'DEFAULT_MAPS = ROOT / "data/diffusion_amp/orca_maps_official_usable_1000_v1/manifest.json"'),
    ('"maps": DomainCorpus(ORCACorpus(maps_manifest, split, device=device, seed=seed,',
     '"maps": DomainCorpus(OfficialMapsORCACorpus(maps_manifest, split, device=device, seed=seed,'),
    ('"schema": "local_orca.maps_interact_union.v1"', '"schema": "local_orca.official_maps_interact_union.v2"')
]
for old, new in replacements:
    if text.count(old) != 1:
        raise ValueError("Expected one source occurrence")
    text = text.replace(old, new)
(HERE / "mixed_corpus.py").write_text(text)
reverse = text
for old, new in reversed(replacements):
    reverse = reverse.replace(new, old)
assert reverse == source and ast.dump(ast.parse(reverse)) == ast.dump(ast.parse(source))
record = {"base": str(BASE), "base_sha256": hashlib.sha256(BASE.read_bytes()).hexdigest(),
          "derived": str(HERE / "mixed_corpus.py"), "derived_sha256": hashlib.sha256(text.encode()).hexdigest(),
          "inverse_bytes_exact": True, "inverse_AST_exact": True,
          "NN_or_training_math_changed": False,
          "replacements": [{"old": old, "new": new} for old, new in replacements]}
(HERE / "mixed_source_derivation.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps({key: value for key, value in record.items() if key != "replacements"}))

