"""Real hierarchy, boundary and checkpoint contracts for future AMP sampling."""

from collections import Counter
from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from CrowdSim.humanoid.training.expert_sampling import (
    ExpertWindowSampler, STRATA, WindowThresholds, classify_windows,
)


def make_corpus(tmp_path, specifications):
    velocity, rotation, rows, sources, counts, starts = [], [], [], [], [], []
    cursor = 0
    for index, spec in enumerate(specifications):
        count = spec.get("frames", 30)
        speed = spec.get("speed", 1.)
        angle = spec.get("initial_yaw", 0.)+torch.arange(count)*spec.get("yaw_rate", 0.)/30
        movement = angle+spec.get("movement_angle", 0.)
        v = torch.zeros(count, 24, 3)
        v[:, 0, 0], v[:, 0, 1] = speed*movement.cos(), speed*movement.sin()
        q = torch.zeros(count, 24, 4)
        q[..., 3] = 1
        q[:, 0, 2], q[:, 0, 3] = (angle/2).sin(), (angle/2).cos()
        dataset = spec.get("dataset", "A")
        parent_name = spec.get("parent", f"parent{index}")
        parent = f"/processed/smpl/{dataset}/subject/{parent_name}.motion"
        row = {"motion_id": index, "source_path": parent, "frames": count, "dt": 1/30,
               "dataset": dataset, "subject_namespace": f"{dataset}/subject"}
        if "segment_start" in spec:
            start = spec["segment_start"]
            row.update(source_path=f"{parent}#frames={start}:{start+count}", parent_source_path=parent,
                       parent_control_frame_start=start, parent_control_frame_stop=start+count,
                       parent_source_frame_start=start, parent_source_frame_stop=start+count)
        rows.append(row)
        velocity.append(v)
        rotation.append(q)
        sources.append(row["source_path"])
        counts.append(count)
        starts.append(cursor)
        cursor += count
    bank = {"gvs": torch.cat(velocity), "grs": torch.cat(rotation),
            "length_starts": torch.tensor(starts), "motion_num_frames": torch.tensor(counts),
            "motion_dt": torch.full((len(rows),), 1/30), "motion_files": tuple(sources)}
    bank_path = tmp_path/"bank.pt"
    torch.save(bank, bank_path)
    entry = {"path": str(bank_path), "bytes": bank_path.stat().st_size,
             "sha256": hashlib.sha256(bank_path.read_bytes()).hexdigest(), "clips": rows}
    manifest = {"schema": "navisaaclab.amass_reconstruction.v1", "splits": {"train": entry, "validation": {"clips": []}, "test": {"clips": []}}}
    manifest_path = tmp_path/"manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return SimpleNamespace(manifest_path=manifest_path, manifest=manifest, entry=entry,
                           data=torch.load(bank_path, weights_only=True, map_location="cpu"), split="train")


def rewrite_manifest(corpus):
    corpus.manifest_path.write_text(json.dumps(corpus.manifest))


def build(corpus, *, weights=None, seed=17, **kwargs):
    return ExpertWindowSampler.from_corpus(corpus, target_weights=weights or {"moving": 1.},
        generator=torch.Generator().manual_seed(seed), minimum_parents_per_bucket=1,
        minimum_windows_per_bucket=1, **kwargs)


@pytest.mark.parametrize("spec,expected", [
    ({"speed": 0., "yaw_rate": 2.}, "low_speed"),
    ({"speed": 2.3, "yaw_rate": 2.}, "fast"),
    ({"speed": 1., "movement_angle": torch.pi/2, "yaw_rate": 2.}, "sideways"),
    ({"speed": 1., "yaw_rate": 1., "initial_yaw": 3.05}, "turning"),
    ({"speed": 1., "yaw_rate": .1}, "moving"),
])
def test_real_short_windows_and_yaw_wrap(tmp_path, spec, expected):
    corpus = make_corpus(tmp_path, [spec])
    labels = classify_windows(corpus.data["gvs"][:, 0, :2], corpus.data["grs"][:, 0])
    assert len(labels) == 20
    assert labels.unique().tolist() == [STRATA.index(expected)]


def test_whole_clip_turn_does_not_label_every_short_history(tmp_path):
    corpus = make_corpus(tmp_path, [{"frames": 100, "yaw_rate": .3}])
    # Complete clip rotates .99rad, but every 0.3s window changes only .09rad.
    sampler = build(corpus)
    assert sampler.support["turning"]["windows"] == 0
    assert sampler.support["moving"]["windows"] == 90


def test_history_and_next_frame_never_cross_bank_motion_boundaries(tmp_path):
    corpus = make_corpus(tmp_path, [{"frames": 11}, {"frames": 15}])
    sampler = build(corpus)
    before = {k: v.clone() for k, v in corpus.data.items() if isinstance(v, torch.Tensor)}
    draws = sampler.sample(4096)
    starts = corpus.data["length_starts"][draws.motion_ids]
    stops = starts+corpus.data["motion_num_frames"][draws.motion_ids]
    assert torch.equal(draws.absolute_indices, starts+draws.frame_indices)
    assert torch.equal(draws.history_indices[:, 0], draws.absolute_indices)
    assert torch.all(draws.history_indices[:, -1] >= starts)
    assert torch.equal(draws.next_indices, draws.absolute_indices+1)
    assert torch.all(draws.next_indices < stops)
    assert (draws.motion_ids == 0).any()
    assert torch.all(draws.frame_indices[draws.motion_ids == 0] == 9)
    for key, value in before.items():
        assert torch.equal(corpus.data[key], value)


def test_uniform_parent_eliminates_four_segment_multiplicity(tmp_path):
    specifications = [{"parent": "four", "frames": 20, "segment_start": k*20} for k in range(4)]
    specifications.append({"parent": "one", "frames": 40})
    sampler = build(make_corpus(tmp_path, specifications))
    expected = sampler.describe()["expected_parent_probability"]
    assert sorted(expected.values()) == pytest.approx([.5, .5])
    assert torch.all(sampler.expected_probabilities[:40] == .5/40)
    assert torch.all(sampler.expected_probabilities[40:] == .5/30)
    draws = sampler.sample(20000)
    counts = Counter(draws.parent_sources)
    assert all(abs(count/20000-.5) < .015 for count in counts.values())
    assert sum(sampler.expected_probabilities).item() == pytest.approx(1.)


def test_exact_stratum_dataset_parent_center_probabilities(tmp_path):
    specs = [{"dataset": "A", "parent": f"a{k}"} for k in range(4)]
    specs += [{"dataset": "B", "parent": "b"}, {"dataset": "C", "parent": "fast", "speed": 2.}]
    sampler = build(make_corpus(tmp_path, specs), weights={"moving": 3., "fast": 1.})
    description = sampler.describe()
    # sqrt(4):sqrt(1) gives A:B=2:1 within moving, not uniform segments.
    assert description["expected_dataset_probability"] == pytest.approx({"A": .5, "B": .25, "C": .25})
    assert torch.all(sampler.expected_probabilities[:80] == .75*(2/3)/4/20)
    assert torch.all(sampler.expected_probabilities[80:100] == .75*(1/3)/20)
    assert torch.all(sampler.expected_probabilities[100:] == .25/20)
    draws = sampler.sample(24000)
    assert Counter(draws.strata)["fast"]/24000 == pytest.approx(.25, abs=.015)
    assert Counter(draws.datasets)["A"]/24000 == pytest.approx(.5, abs=.015)
    expected_probability = torch.where(draws.motion_ids < 4,
        torch.tensor(.75*(2/3)/4/20, dtype=torch.float64),
        torch.tensor(.25/20, dtype=torch.float64))
    assert torch.allclose(draws.probabilities, expected_probability, atol=1e-15, rtol=0)


def test_dataset_cap_is_exact_when_feasible_and_never_silently_relaxes(tmp_path):
    specs = [{"dataset": "A", "parent": f"a{k}"} for k in range(4)]+[{"dataset": "B"}]
    corpus = make_corpus(tmp_path, specs)
    sampler = build(corpus, dataset_cap=.55)
    assert sampler.describe()["expected_dataset_probability"] == pytest.approx({"A": .55, "B": .45})
    with pytest.raises(ValueError, match="unsupported"):
        build(corpus, dataset_cap=.4)


def test_sparse_positive_weight_bucket_is_rejected(tmp_path):
    corpus = make_corpus(tmp_path, [{}, {"speed": 2.}])
    with pytest.raises(ValueError, match="Unsupported positive-weight bucketfast"):
        ExpertWindowSampler.from_corpus(corpus, target_weights={"fast": 1.},
            generator=torch.Generator().manual_seed(1), minimum_parents_per_bucket=5,
            minimum_windows_per_bucket=10)
    with pytest.raises(ValueError, match="Unsupported positive-weight bucketturning"):
        build(corpus, weights={"moving": 1., "turning": .1})


def test_supplied_generator_is_not_advanced_by_registry_construction(tmp_path):
    corpus = make_corpus(tmp_path, [{}])
    generator = torch.Generator().manual_seed(7)
    original = generator.get_state().clone()
    sampler = ExpertWindowSampler.from_corpus(corpus, target_weights={"moving": 1.}, generator=generator,
        minimum_parents_per_bucket=1, minimum_windows_per_bucket=1)
    assert torch.equal(generator.get_state(), original)
    sampler.sample(17)
    assert not torch.equal(generator.get_state(), original)


def test_checkpoint_reproduces_all_random_draws_and_counters(tmp_path):
    corpus = make_corpus(tmp_path, [{"dataset": "A"}, {"dataset": "B"}, {"speed": 2., "dataset": "A"}])
    sampler = build(corpus, weights={"moving": .7, "fast": .3})
    sampler.sample(15)
    saved = sampler.state_dict()
    expected = sampler.sample(123)
    restored = build(corpus, weights={"moving": .7, "fast": .3}, seed=999)
    restored.load_state_dict(saved)
    actual = restored.sample(123)
    for field in ("motion_ids", "frame_indices", "absolute_indices", "history_indices", "next_indices", "probabilities"):
        assert torch.equal(getattr(expected, field), getattr(actual, field))
    assert expected.strata == actual.strata
    assert expected.datasets == actual.datasets
    assert expected.parent_sources == actual.parent_sources
    assert restored.sampled_windows == sampler.sampled_windows == 138
    assert restored.sampled_batches == sampler.sampled_batches == 2


def test_checkpoint_rejects_changed_registry_and_configuration(tmp_path):
    corpus = make_corpus(tmp_path, [{}])
    sampler = build(corpus)
    saved = sampler.state_dict()
    with pytest.raises(ValueError, match="config_sha256"):
        build(corpus, dataset_weighting="uniform").load_state_dict(saved)
    corpus.data["gvs"][0, 0, 0] += .001
    with pytest.raises(ValueError, match="registry_sha256"):
        build(corpus).load_state_dict(saved)
    bad = {**saved, "generator_state": torch.zeros(3, dtype=torch.uint8)}
    with pytest.raises(ValueError, match="Generator state"):
        sampler.load_state_dict(bad)
    with pytest.raises(ValueError, match="complete exact state"):
        sampler.load_state_dict({"generator_state": saved["generator_state"]})


@pytest.mark.parametrize("weights", [{"typo": 1.}, {"moving": 0.}, {"moving": -1.},
    {"moving": float("nan")}, {"moving": float("inf")}, {"moving": True}, {"moving": "1"}])
def test_unknown_or_invalid_priors_rejected(tmp_path, weights):
    corpus = make_corpus(tmp_path, [{}])
    with pytest.raises(ValueError):
        ExpertWindowSampler.from_corpus(corpus, target_weights=weights, generator=torch.Generator(),
            minimum_parents_per_bucket=1, minimum_windows_per_bucket=1)


@pytest.mark.parametrize("kwargs", [{"history_frames": True}, {"history_frames": 11},
    {"future_frames": True}, {"dataset_cap": True}, {"dataset_weighting": []},
    {"thresholds": False}])
def test_invalid_contract_types_rejected(tmp_path, kwargs):
    with pytest.raises(ValueError):
        build(make_corpus(tmp_path, [{}]), **kwargs)


@pytest.mark.parametrize("field", list(WindowThresholds.__dataclass_fields__))
def test_numeric_boolean_thresholds_rejected(field):
    with pytest.raises(ValueError, match="not a boolean"):
        replace(WindowThresholds(), **{field: True}).validate()


def test_nonfinite_bad_quaternion_and_non30hz_rejected(tmp_path):
    corpus = make_corpus(tmp_path, [{}])
    corpus.data["gvs"][12, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        build(corpus)
    corpus.data["gvs"][12, 0, 0] = 1
    corpus.data["grs"][12, 0, 3] = .5
    with pytest.raises(ValueError, match="unit XYZW"):
        build(corpus)
    corpus.data["grs"][12, 0, 3] = 1
    corpus.data["motion_dt"][0] = 1/50
    with pytest.raises(ValueError, match="real30Hz"):
        build(corpus)


@pytest.mark.parametrize("mutation", ["dataset", "subject", "parent", "bounds", "frames", "duplicate"])
def test_actual_metadata_conflicts_rejected(tmp_path, mutation):
    corpus = make_corpus(tmp_path, [{"parent": "p", "segment_start": 0}, {"parent": "p", "segment_start": 30}])
    row = corpus.entry["clips"][0]
    if mutation == "dataset":
        row["dataset"] = "invented"
    elif mutation == "subject":
        row["subject_namespace"] = "A/wrong"
    elif mutation == "parent":
        row["parent_source_path"] = "/processed/smpl/A/subject/wrong.motion"
    elif mutation == "bounds":
        row["parent_control_frame_stop"] += 1
    elif mutation == "frames":
        row["frames"] = 31
    else:
        corpus.entry["clips"].append(dict(row))
    rewrite_manifest(corpus)
    with pytest.raises(ValueError):
        build(corpus)


def test_overlapping_parent_intervals_and_heldout_leakage_rejected(tmp_path):
    corpus = make_corpus(tmp_path, [{"parent": "p", "segment_start": 0}, {"parent": "p", "segment_start": 20}])
    with pytest.raises(ValueError, match="Overlapping parent"):
        build(corpus)
    corpus = make_corpus(tmp_path, [{"parent": "p", "segment_start": 0}])
    corpus.manifest["splits"]["validation"]["clips"] = [{"source_path": "/processed/smpl/A/subject/p.motion"}]
    rewrite_manifest(corpus)
    with pytest.raises(ValueError, match="held-out parent"):
        build(corpus)
    corpus.manifest["splits"]["validation"]["clips"] = []
    rewrite_manifest(corpus)
    corpus.split = "validation"
    with pytest.raises(ValueError, match="held-out split"):
        build(corpus)


def test_original50hz_dependency_overlap_and_actual_bank_overlap_rejected(tmp_path):
    corpus = make_corpus(tmp_path, [{"parent": "p", "segment_start": 0}, {"parent": "p", "segment_start": 30}])
    corpus.entry["clips"][1]["parent_source_frame_start"] = 20
    rewrite_manifest(corpus)
    with pytest.raises(ValueError, match="Overlapping parent"):
        build(corpus)
    corpus = make_corpus(tmp_path, [{}, {}])
    corpus.data["length_starts"][1] = 20
    with pytest.raises(ValueError, match="bank frame storage"):
        build(corpus)


def test_describe_is_json_safe_and_cannot_mutate_sampler_config(tmp_path):
    sampler = build(make_corpus(tmp_path, [{}]))
    description = sampler.describe()
    json.dumps(description, allow_nan=False)
    description["config"]["future_frames"] = 99
    description["support"]["moving"]["windows"] = 0
    assert sampler.config["future_frames"] == 1
    assert sampler.support["moving"]["windows"] == 20
    assert sampler.describe()["source"]["bank_sha256_recomputed_here"] is False


@pytest.mark.parametrize("batch", [True, 0, -1, 1.5])
def test_invalid_batch_rejected(tmp_path, batch):
    sampler = build(make_corpus(tmp_path, [{}]))
    with pytest.raises(ValueError):
        sampler.sample(batch)
