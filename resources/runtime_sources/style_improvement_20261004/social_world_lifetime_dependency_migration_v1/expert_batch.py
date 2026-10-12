"""Source-bound expert batches for a caller-built social World and AMASS corpus.

Stdlib at import. No corpus/world/sampler construction, seeding or NN imports.
Future sampling uses the exact original AMPWorld.expert_batch function AST.
"""
from __future__ import annotations

import ast
import __future__
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from protomotions.repository import repository_root
ROOT = repository_root(__file__)
MANIFEST = ROOT / "data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json"
MANIFEST_SHA = "68d41be28908f28a09c47f7658af39d55fad01b32212bd7c0ced2b1e6b1489ca"
EXPERT_AST_SHA = "dbf7c3d3419e4c9cb6a4be571a726a854f41c9f6f9fd633d703093256c5be787"
PINS = {
    "world": ("resources/runtime_sources/style_improvement_20261004/shared_group_world_owned_lifetime_revision_v2/shared_group_world.py", 39960, 'd7400a48364b64971f4e2e942d87ebd9e108b5a8ff018e8a8f404b817bd02038'),
    "corpus": ("CrowdSim/humanoid/amass_corpus.py", 12126, '2d4e4f7549aa38f5fd6763de4d3924a9cdd969746682c649ab773f033d747cf7'),
    "original_expert": ("CrowdSim/humanoid/amp_world.py", 30450, 'a114f1b0763b7a2307a3f7fdd9409c38b4385ed4db02a9fbf36a56854dee554a'),
    "sampler": ("CrowdSim/humanoid/training/expert_sampling.py", 32462, 'b16984a920f21ca92c3fe01976d9c0453a75bf54d3f46df369984ff27453f9cd'),
}


def _record(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _class_module(value, key, name):
    module = sys.modules.get(type(value).__module__)
    if (module is None or Path(getattr(module, "__file__", "")).resolve() != ROOT / PINS[key][0]
            or type(value) is not getattr(module, name, None)):
        raise TypeError("Caller-built original " + name + " is required")
    return module


def _sampling_identity(sampler):
    description = sampler.describe()
    window = description["sampler"]
    return deepcopy({"configuration": description["configuration"],
        **{name: window[name] for name in ("schema", "config_sha256", "registry_sha256", "probabilities_sha256", "source")}})


def _require_idle(world):
    world._stationary_clock()
    if world._prepared is not None or world._pending is not None:
        raise RuntimeError("Expert sampling requires idle World after completed control/reset")


def _corpus_config(corpus):
    return (corpus.split, corpus.device, Path(corpus.manifest_path).resolve(), corpus.dof_velocity_frame,
            corpus.stored_dof_velocity_frame, corpus.output_dof_velocity_frame, corpus._convert_dof_velocity)


class _RecordingSource:
    def __init__(self, source):
        self.source, self.calls, self.result = source, 0, None

    def sample_reference(self, batch, history_frames):
        self.calls += 1
        if self.calls != 1 or history_frames != 10:
            raise RuntimeError("One original history10 expert draw is required")
        self.result = self.source.sample_reference(batch, history_frames)
        return self.result


class SocialAMASSExpert:
    """Connect the existing expert prior without altering physical reset sampling.

    Uniform is uniform train clip then a legal center, not uniform windows.
    Optional q needs an already built exact sampler and explicit matching identity.
    Source/bank integrity here is manifest bytes + bank stat/declaration only.
    """

    def __init__(self, world, corpus, *, expert_sampler=None, sampling_config=None, sampling_provenance=None):
        sources = {}
        for key, (relative, size, sha) in PINS.items():
            row = _record(ROOT / relative)
            if row["bytes"] != size or row["sha256"] != sha:
                raise ValueError("Expert dependency changed: " + key)
            sources[key] = row
        module = _class_module(world, "world", "SharedGroupWorld")
        _class_module(corpus, "corpus", "AMASSCorpus")
        torch = module.torch
        _require_idle(world)
        if (world.curriculum.corpus is not corpus or corpus.device != world.device or corpus.split != "train"
                or Path(corpus.manifest_path).resolve() != MANIFEST):
            raise ValueError("Use the live World's same original train corpus/device/manifest")
        if not isinstance(corpus.generator, torch.Generator) or corpus.generator.device.type != "cpu":
            raise ValueError("The actual original shared CPU corpus generator is required")
        manifest_bytes = MANIFEST.read_bytes()
        if hashlib.sha256(manifest_bytes).hexdigest() != MANIFEST_SHA:
            raise ValueError("Fixed reconstructed v7 manifest bytes changed")
        document = json.loads(manifest_bytes)
        entry = document["splits"]["train"]
        if (document["schema"] != "navisaaclab.amass_reconstruction.v1" or document.get("author_clip_list") is not False
                or document != corpus.manifest or entry != corpus.entry or len(entry["clips"]) != 2848 or entry["clip_count"] != 2848):
            raise ValueError("Exact local reconstructed2848 train entry is required")
        ids = torch.tensor([row["motion_id"] for row in entry["clips"]], dtype=torch.int64)
        if (not isinstance(corpus.ids, torch.Tensor) or corpus.ids.dtype != torch.int64 or corpus.ids.device.type != "cpu"
                or not torch.equal(corpus.ids, ids)):
            raise ValueError("Actual train ID pool differs from the bound manifest")
        counts = corpus.data["motion_num_frames"][ids]
        expected_counts = torch.tensor([row["frames"] for row in entry["clips"]], dtype=counts.dtype)
        if not torch.equal(counts, expected_counts):
            raise ValueError("Caller-loaded train frame counts differ from actual manifest metadata")
        bank = Path(entry["path"])
        if not bank.is_absolute():
            bank = MANIFEST.parent / bank
        if bank.stat().st_size != entry["bytes"]:
            raise ValueError("Caller-loaded bank size differs from the declaration")
        if type(world.mapping) is not world.deps.observations.PacerSMPLMapping:
            raise TypeError("The World's exact named COMMON/PACER mapping is required")
        for limits in (world.lower, world.upper):
            if (not isinstance(limits, torch.Tensor) or limits.shape != (69,) or limits.dtype != torch.float32
                    or limits.device != world.device or not bool(torch.isfinite(limits).all())):
                raise ValueError("Live named69 float32 PD limits are required")
        if bool((world.lower > world.upper).any()):
            raise ValueError("Live PD limits are unordered")
        self._torch, self._world, self._corpus = torch, world, corpus
        self._source, self._sampling_identity = corpus, None
        if expert_sampler is None:
            if sampling_config is not None or sampling_provenance is not None:
                raise ValueError("Sampling configuration/provenance requires an explicit prepared sampler")
        else:
            sampler_module = _class_module(expert_sampler, "sampler", "ExpertReferenceSampler")
            _class_module(expert_sampler.sampler, "sampler", "ExpertWindowSampler")
            if (expert_sampler.corpus is not corpus or sampling_config is None or sampling_provenance is None
                    or not isinstance(expert_sampler.generator, torch.Generator) or expert_sampler.generator.device.type != "cpu"
                    or expert_sampler.generator is corpus.generator or expert_sampler.sampler.generator is not expert_sampler.generator
                    or sampler_module.normalize_expert_sampling_config(sampling_config) != expert_sampler.config):
                raise ValueError("Prepared expert sampler/corpus and explicit configuration must match")
            identity = _sampling_identity(expert_sampler)
            if (identity != sampling_provenance or identity["source"]["manifest_sha256"] != MANIFEST_SHA
                    or identity["source"]["bank_sha256_declared"] != entry["sha256"]):
                raise ValueError("Prepared expert sampler identity/source differs from explicit provenance")
            self._source, self._sampling_identity = expert_sampler, identity
        source = (ROOT / PINS["original_expert"][0]).read_bytes()
        klass = next(node for node in ast.parse(source).body if isinstance(node, ast.ClassDef) and node.name == "AMPWorld")
        method = next(node for node in klass.body if isinstance(node, ast.FunctionDef) and node.name == "expert_batch")
        tree = ast.Module(body=[method], type_ignores=[])
        if hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest() != EXPERT_AST_SHA:
            raise ValueError("Original expert function AST changed")
        globals_ = {"torch": torch, "build_amp_observation": world.deps.observations.build_amp_observation,
                    "action_to_pd": world.deps.action_to_pd}
        exec(compile(tree, str(ROOT / PINS["original_expert"][0]), "exec", flags=__future__.annotations.compiler_flag), globals_)
        self._expert = globals_["expert_batch"]
        self._host = SimpleNamespace(corpus=corpus, history_frames=10, mapping=world.mapping,
                                     lower=world.lower.clone(), upper=world.upper.clone(), device=world.device)
        self._document, self._ids, self._corpus_config = document, ids.clone(), _corpus_config(corpus)
        self._generator, self._data, self._deps = corpus.generator, corpus.data, world.deps
        self._mapping_names = (tuple(world.mapping.body_names), tuple(world.mapping.dof_names))
        self._private_generator = None if expert_sampler is None else expert_sampler.generator
        self._mask = torch.tensor([name.rsplit("_", 1)[0] not in {"L_Hand", "R_Hand", "L_Toe", "R_Toe"}
            for name in world.deps.observations.PACER_DOF_NAMES], dtype=torch.bool, device=world.device)
        if int(self._mask.sum()) != 57:
            raise ValueError("Exact57 effective action coordinates are required")
        self.batches, self.windows, self._faulted = 0, 0, False
        self._metadata = {"schema": "local.neutral_smpl.social_amass_expert.v1", "sources": sources,
            "manifest": {"path": str(MANIFEST), "bytes": len(manifest_bytes), "sha256": MANIFEST_SHA},
            "train_clips": 2848, "bank": {"path": str(bank.resolve()), "bytes_stat": bank.stat().st_size,
                "sha256_declared": entry["sha256"], "sha256_recomputed_here": False},
            "author_clip_list": False, "sampler": "uniform_clip_then_legal_center" if expert_sampler is None else "explicit_prepared_stratified_windows",
            "sampling_identity": self._sampling_identity,
            "dof_velocity": {"requested": corpus.dof_velocity_frame, "stored": corpus.stored_dof_velocity_frame,
                             "output": corpus.output_dof_velocity_frame},
            "expert_action": "clipped next-frame mocap position / named PD scale; zero frozen hand/toe; not torque or measured author demonstration action",
            "original_expert_AST_sha256": EXPERT_AST_SHA, "physics_training_or_quality_verified": False}

    def _check_live(self):
        world, corpus, torch = self._world, self._corpus, self._torch
        _require_idle(world)
        if (world.curriculum.corpus is not corpus or corpus.device != world.device
                or corpus.generator is not self._generator or corpus.data is not self._data
                or _corpus_config(corpus) != self._corpus_config or corpus.manifest != self._document
                or corpus.entry != self._document["splits"]["train"] or not torch.equal(corpus.ids, self._ids)
                or world.deps is not self._deps or world.mapping is not self._host.mapping
                or (tuple(world.mapping.body_names), tuple(world.mapping.dof_names)) != self._mapping_names):
            raise ValueError("Live expert corpus/generator/configuration/named mapping changed")
        for actual, expected in ((world.lower, self._host.lower), (world.upper, self._host.upper)):
            if (not isinstance(actual, torch.Tensor) or actual.dtype != expected.dtype or actual.device != expected.device
                    or not torch.equal(actual, expected)):
                raise ValueError("Live named PD limits changed")
        if self._sampling_identity is not None:
            if (self._source.corpus is not corpus or self._source.generator is not self._private_generator
                    or self._source.sampler.generator is not self._private_generator
                    or _sampling_identity(self._source) != self._sampling_identity):
                raise ValueError("Prepared expert sampling configuration/source/generator changed")

    def sample(self, batch):
        """Owned features[B,10,195], action69, exact sampled train IDs/frames."""
        if type(batch) is not int or batch < 1:
            raise ValueError("Expert batch must be a positive exact integer")
        if self._faulted:
            raise RuntimeError("Expert adapter faulted after possible sampling RNG consumption")
        world, torch = self._world, self._torch
        self._check_live()
        recorder = _RecordingSource(self._source)
        self._host.expert_reference_sampler = recorder
        try:
            features, actions = self._expert(self._host, batch)
            _, history, next_position, ids, frames = recorder.result
            if (recorder.calls != 1 or features.shape != (batch, 10, 195) or actions.shape != (batch, 69) or next_position.shape != (batch, 69)
                    or any(value.dtype != torch.float32 or value.device != world.device or not bool(torch.isfinite(value).all())
                           for value in (features, actions, next_position))
                    or bool((actions[:, ~self._mask] != 0).any()) or bool((actions.abs() > 1).any())):
                raise ValueError("Original expert output violates history10/195/action69/frozen57")
            if (any(not isinstance(value, torch.Tensor) or value.shape != (batch,) or value.dtype != torch.int64
                    or value.device.type != "cpu" for value in (ids, frames))
                    or not bool(torch.isin(ids, self._corpus.ids).all()) or bool((frames < 9).any())
                    or bool((frames + 1 >= self._corpus.data["motion_num_frames"][ids]).any())):
                raise ValueError("Sampled expert IDs/history/next frame cross train clip boundaries")
            owned_history = SimpleNamespace(**{name: value.detach().clone() for name, value in vars(history).items()})
            self._check_live()
            result = SimpleNamespace(features=features.detach().clone(), actions=actions.detach().clone(), history=owned_history,
                next_position_common=next_position.detach().clone(), motion_ids=ids.detach().clone(), frame_indices=frames.detach().clone(),
                effective_action_mask=self._mask.clone(), batches=self.batches + 1, windows=self.windows + batch)
            self.batches, self.windows = result.batches, result.windows
            return result
        except BaseException:
            self._faulted = True
            raise

    def describe(self):
        return deepcopy({**self._metadata, "sampled_batches": self.batches, "sampled_windows": self.windows})
