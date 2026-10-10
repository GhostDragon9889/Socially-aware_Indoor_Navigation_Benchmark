"""SDK-free, read-only verification against the frozen actual v7 train bank."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import torch

from CrowdSim.humanoid.amass_corpus import AMASSCorpus
from CrowdSim.humanoid.training.expert_sampling import ExpertWindowSampler


ROOT = Path(__file__).resolve().parents[2]


def main():
    torch.set_num_threads(4)
    started = time.monotonic()
    manifest = ROOT/"data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json"
    corpus = AMASSCorpus(manifest, device="cpu", dof_velocity_frame="child_joint")
    corpus_rng_before = corpus.generator.get_state().clone()
    targets = {"low_speed": .10, "fast": .05, "sideways": .10, "turning": .25, "moving": .50}
    sampler = ExpertWindowSampler.from_corpus(corpus, target_weights=targets,
        generator=torch.Generator(device="cpu").manual_seed(20261002))
    before = sampler.state_dict()
    draw = sampler.sample(32768)
    starts = corpus.data["length_starts"][draw.motion_ids]
    stops = starts+corpus.data["motion_num_frames"][draw.motion_ids]
    assert torch.equal(draw.absolute_indices, starts+draw.frame_indices)
    assert (draw.history_indices[:, -1] >= starts).all()
    assert (draw.next_indices < stops).all()
    assert torch.isfinite(draw.probabilities).all() and (draw.probabilities > 0).all()
    assert torch.equal(corpus.generator.get_state(), corpus_rng_before)
    description = sampler.describe()
    sampler.load_state_dict(before)
    repeat = sampler.sample(32768)
    for field in ("motion_ids", "frame_indices", "absolute_indices", "history_indices", "next_indices", "probabilities"):
        assert torch.equal(getattr(draw, field), getattr(repeat, field))
    assert draw.strata == repeat.strata and draw.parent_sources == repeat.parent_sources
    result = {"status": "passed", "cpu_only": True, "kit_started": False, "gpu_used": False,
        "elapsed_s": time.monotonic()-started, "sampler": description,
        "actual_draws": len(draw.motion_ids),
        "actual_stratum_probability": {key: value/len(draw.motion_ids) for key, value in sorted(Counter(draw.strata).items())},
        "actual_dataset_probability": {key: value/len(draw.motion_ids) for key, value in sorted(Counter(draw.datasets).items())},
        "actual_unique_parents": len(set(draw.parent_sources)),
        "history_and_next_frame_boundaries_verified": True,
        "checkpoint_draws_exactly_reproduced": True, "existing_corpus_rng_unchanged": True,
        "source_hashes": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in (
            "CrowdSim/humanoid/training/expert_sampling.py", "tests/test_expert_window_sampling.py")},
        "limitations": ["Not connected to a trainer, reset sampler or runner.",
            "No training/physical style improvement claim.", "Target probabilities are declared engineering choices, not author hyperparameters.",
            "Full bank SHA is reused from frozen v7 integrity proof, not rehashed in this index verification."]}
    output = Path(__file__).with_name("expert_window_sampling_v7_verification.json")
    output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({key: value for key, value in result.items() if key != "sampler"}, indent=2))


if __name__ == "__main__":
    main()
