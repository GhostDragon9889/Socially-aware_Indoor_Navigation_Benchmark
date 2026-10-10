"""CPU-only proof on the immutable v7 bank; never writes corpus or model files."""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import torch
import yaml

from CrowdSim.humanoid.amass_corpus import AMASSCorpus
from CrowdSim.humanoid.training.expert_sampling import ExpertReferenceSampler, STRATA


ROOT = Path(__file__).resolve().parents[2]
DOCS = Path(__file__).resolve().parent
MANIFEST = ROOT/"data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json"
PROFILE = ROOT/"configs/diffusion_amp/amp_style_amass_stratified_experts.yaml"


def record(path):
    path = Path(path)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def corpus(seed=7):
    return AMASSCorpus(MANIFEST, "train", device="cpu", seed=seed, dof_velocity_frame="child_joint")


def same_reference(left, right):
    for a, b in zip(left[:2], right[:2], strict=True):
        for field in vars(a):
            assert torch.equal(getattr(a, field), getattr(b, field)), field
    for a, b in zip(left[2:], right[2:], strict=True):
        assert torch.equal(a, b)


def main():
    torch.set_num_threads(4)
    assert not torch.cuda.is_initialized()
    profile = yaml.safe_load(PROFILE.read_text())
    config = profile["training"]["expert_sampling"]
    legacy, replacement = corpus(), corpus()
    global_before = torch.get_rng_state().clone()
    started = time.perf_counter()
    bridge = ExpertReferenceSampler(replacement, config)
    build_s = time.perf_counter()-started
    assert torch.equal(legacy.generator.get_state(), replacement.generator.get_state())
    assert torch.equal(global_before, torch.get_rng_state())
    reset_pairs = []
    for reset_batch, expert_batch in ((4, 64), (31, 512), (1, 4096), (19, 7), (5, 123)):
        left = legacy.sample_reference(reset_batch)
        right = replacement.sample_reference(reset_batch)
        same_reference(left, right)
        legacy.sample_reference(expert_batch)
        actual = bridge.sample_reference(expert_batch)
        assert torch.equal(legacy.generator.get_state(), replacement.generator.get_state())
        starts = replacement.data["length_starts"][actual[3]]
        counts = replacement.data["motion_num_frames"][actual[3]]
        assert torch.all(actual[4] >= 9) and torch.all(actual[4]+1 < counts)
        assert torch.equal(actual[2], replacement.data["dps"][starts+actual[4]+1])
        reset_pairs.append({"reset_batch": reset_batch, "expert_batch": expert_batch,
                            "reset_motion_ids": left[3].tolist(), "reset_frames": left[4].tolist(),
                            "all_reset_payloads_and_shared_rng_equal": True})
    assert torch.equal(global_before, torch.get_rng_state())
    state, corpus_rng = bridge.state_dict(), replacement.generator.get_state().clone()
    expected = bridge.sample_reference(4096)
    expected_reset = replacement.sample_reference(127)
    recovered_corpus = corpus(seed=200)
    recovered = ExpertReferenceSampler(recovered_corpus, config)
    recovered_corpus.generator.set_state(corpus_rng)
    recovered.load_state_dict(state)
    same_reference(expected, recovered.sample_reference(4096))
    same_reference(expected_reset, recovered_corpus.sample_reference(127))
    assert torch.equal(replacement.generator.get_state(), recovered_corpus.generator.get_state())
    assert torch.equal(bridge.generator.get_state(), recovered.generator.get_state())

    classes, datasets, parents = Counter(), Counter(), Counter()
    sample_started = time.perf_counter()
    # Same production batch ceiling; avoid one unnecessarily huge history gather.
    for _ in range(8):
        draws = bridge.sampler.sample(4096)
        classes.update(draws.strata)
        datasets.update(draws.datasets)
        parents.update(draws.parent_sources)
        starts = replacement.data["length_starts"][draws.motion_ids]
        stops = starts+replacement.data["motion_num_frames"][draws.motion_ids]
        assert torch.all(draws.history_indices[:, -1] >= starts)
        assert torch.all(draws.next_indices < stops)
    sample_s = time.perf_counter()-sample_started
    description = bridge.describe()
    target = description["sampler"]["config"]["normalized_target_probabilities"]
    assert sum(classes.values()) == 32768
    assert all(abs(classes[name]/32768-target[name]) < .015 for name in STRATA)
    assert torch.equal(global_before, torch.get_rng_state()) and not torch.cuda.is_initialized()
    result = {"schema": "navisaaclab.expert_sampling_integration_v7_verification.v1", "status": "passed",
              "generated_at_utc": datetime.now(timezone.utc).isoformat(), "CPU_only": True, "Kit_started": False,
              "manifest": record(MANIFEST), "profile": record(PROFILE), "implementation": record(ROOT/"CrowdSim/humanoid/training/expert_sampling.py"),
              "bank_hash_scope": "Stat bytes and declared frozen v7 SHA; this proof does not repeat the1.1GB bank hash",
              "description": description, "actual_reset_expert_pair_checks": reset_pairs,
              "build_index_wall_s": build_s, "private_32768_window_sample_wall_s": sample_s,
              "actual_private_sample_counts": {"windows":32768, "strata":dict(classes), "datasets":dict(datasets),
                                              "unique_parents":len(parents), "maximum_parent_draws":max(parents.values())},
              "resume_expert_history_next_and_following_reset_equal": True,
              "global_Torch_CPU_rng_unchanged": True, "independent_generator_identity": bridge.generator is not replacement.generator,
              "global_Torch_CUDA_rng_observed": False, "GPU_used": False,
              "limitations": ["Only actual CPU corpus/reference/expert windows; no new training or physical reset was executed",
                  "The32768 prior probe calls the isolated sampler directly; paired bridge checks separately prove legacy shared RNG compensation",
                  "Maintained shared RNG equality applies only to matched call schedules; changed policies may change actual fall/reset times",
                  "This explicit prior is not a paper hyperparameter and has not demonstrated motion-quality improvement"]}
    output = DOCS/"expert_sampling_integration_v7_verification.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print("PASS", output)
    print(json.dumps({"build_wall_s":build_s, "registry":description["sampler"]["registry_sha256"],
                      "probabilities":description["sampler"]["probabilities_sha256"], "legal_centers":description["sampler"]["legal_centers"],
                      "actual_sample":result["actual_private_sample_counts"]}, indent=2))


if __name__ == "__main__":
    main()
