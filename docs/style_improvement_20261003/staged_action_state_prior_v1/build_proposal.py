from pathlib import Path
import difflib
import hashlib
import json

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
EXISTING = ['CrowdSim/humanoid/application/settings.py', 'CrowdSim/humanoid/training/trainer.py', 'CrowdSim/humanoid/runtime/checkpoint_contract.py', 'CrowdSim/humanoid/application/resources.py']
NEW_HELPER = 'CrowdSim/humanoid/application/state_prior_initialization.py'
NEW_PROFILE = 'configs/diffusion_amp/amp_style_mixed_state_action_state_prior_terrain_protected.yaml'

helper = '''"""Stdlib-only contract for a public state prior with zero action extension.

Validation reads saved JSON provenance only; resume/inference never reopen the
public checkpoint or require trained action columns to remain zero.
"""
from pathlib import Path

PUBLIC_PACER_SHA256 = "bc0cb89b4d7ceaeb198339c34f3d736eb7449e7e479a7ee0f80d32563737d615"
STRATEGY = "public_pacer_state_prior_zero_action_extension"
PUBLIC_COUNTS = {"self_normalizer": 61931520001.0, "task_normalizer": 61931520001.0,
                 "amp_normalizer": 15482880001.0, "value_normalizer": 6881280001.0}
EXTENSION = {"schema": "navisaaclab.action_state_prior_initialization.v1",
             "history_columns": 1950, "action_columns": 69,
             "action_column_initialization": "zeros",
             "public_prior_training_data_overlap_with_current_amass": "unknown"}


def validate_action_state_prior_config(config):
    enabled = getattr(config, "warm_start_action_discriminator_from_state_prior", False)
    if type(enabled) is not bool:
        raise ValueError("warm_start_action_discriminator_from_state_prior must be an explicit boolean")
    if enabled and (config.warm_start_discriminator is not False
            or config.discriminator_input != "state_action"
            or config.amp_normalization != "per_history_coordinate"
            or (config.history_frames, config.amp_frame_dim, tuple(config.discriminator_layers)) != (10, 195, (1024, 512))
            or config.normalize_observations is not True or config.normalize_values is not True
            or config.normalization_clip != 5.):
        raise ValueError("Action state prior requires state_action10x195 exact layers/history RMS, normalized observations/values clip5 and warm_start_discriminator=False")
    return enabled


def validate_action_state_prior_initialization(config, initialization):
    """Require the coupled D/history-RMS provenance without mutating it."""
    enabled = validate_action_state_prior_config(config)
    if not enabled:
        if type(initialization) is dict and (
                "state_prior_action_extension" in initialization
                or any(type(initialization.get(name)) is str and initialization[name] == STRATEGY
                       for name in ("discriminator_initialization", "action_conditioned_discriminator_initialization"))):
            raise ValueError("Action state prior initialization provenance requires its explicit flag")
        return
    if type(initialization) is not dict:
        raise ValueError("Action state prior requires complete initialization provenance")
    expected = {"mode": "transfer_warm_start_public_pacer_actor_critic",
                "sha256": PUBLIC_PACER_SHA256, "actor_critic_task_transferred": True,
                "observation_value_moments_transferred": True,
                "action_conditioned_discriminator_initialization": STRATEGY,
                "discriminator_initialization": STRATEGY, "discriminator_input": "state_action",
                "amp_normalization": "per_history_coordinate",
                "published_state_only_discriminator_transferred": True,
                "published_amp_history_moments_transferred": True,
                "author_navisaaclab_weights": False}
    for name, value in expected.items():
        if type(initialization.get(name)) is not type(value) or initialization[name] != value:
            raise ValueError(f"Invalid action state prior initialization provenance: {name}")
    path = initialization.get("path")
    if type(path) is not str or not path.strip() or "://" in path or not Path(path).is_absolute():
        raise ValueError("Invalid action state prior initialization provenance: path")
    extension = initialization.get("state_prior_action_extension")
    if (type(extension) is not dict or set(extension) != set(EXTENSION)
            or any(type(extension[name]) is not type(value) or extension[name] != value
                   for name, value in EXTENSION.items())):
        raise ValueError("Invalid action state prior initialization provenance: extension")
    counts = initialization.get("normalizer_counts_before_cap")
    if (type(counts) is not dict or set(counts) != set(PUBLIC_COUNTS)
            or any(type(counts[name]) is not float or counts[name] != value
                   for name, value in PUBLIC_COUNTS.items())):
        raise ValueError("Invalid action state prior initialization provenance: public counts")
    cap = initialization.get("normalizer_count_cap")
    if ((cap is not None and type(cap) not in (int, float))
            or cap != config.warm_start_normalizer_count_cap
            or "normalizer_count_cap" not in initialization):
        raise ValueError("Invalid action state prior initialization provenance: count cap")
'''

def replace_once(source, old, new):
    assert source.count(old) == 1, old[:120]
    return source.replace(old, new, 1)

source = (ROOT / EXISTING[0]).read_text()
source = replace_once(source, 'import math\n', 'import math\n\nfrom .state_prior_initialization import validate_action_state_prior_config\n')
source = replace_once(source, '    warm_start_discriminator: bool = False\n', '    warm_start_discriminator: bool = False\n    warm_start_action_discriminator_from_state_prior: bool = False\n')
source += '        validate_action_state_prior_config(self)\n'
proposed = {EXISTING[0]: source}
source = (ROOT / EXISTING[1]).read_text()
source = replace_once(source, 'from ..application.settings import TrainingConfig\n', 'from ..application.settings import TrainingConfig\nfrom ..application.state_prior_initialization import (EXTENSION, STRATEGY,\n    validate_action_state_prior_config, validate_action_state_prior_initialization)\n')
source = replace_once(source, '        self.terrain_statistics_policy.require_fresh_warm_start(self.update_count)\n', '''        validate_action_state_prior_config(self.config)
        if self.config.warm_start_action_discriminator_from_state_prior and (
                self.physical_transition_count or self.provenance.get("initialization") is not None
                or self.policy_replay is not None and self.policy_replay.size):
            raise ValueError("Action state prior requires a fresh uninitialized trainer before any physical transition/replay")
        self.terrain_statistics_policy.require_fresh_warm_start(self.update_count)
''')
source = replace_once(source, '        if (self.config.task_terrain_statistics_mode == "preserve_declared_suffix"\n', '        if ((self.config.task_terrain_statistics_mode == "preserve_declared_suffix"\n                or self.config.warm_start_action_discriminator_from_state_prior)\n')
source = replace_once(source, '        transfer_disc = self.config.warm_start_discriminator\n        if transfer_disc:\n', '''        state_prior = self.config.warm_start_action_discriminator_from_state_prior
        transfer_disc = self.config.warm_start_discriminator
        if state_prior:
            self.model.discriminator[0].weight[:, :1950].copy_(network._disc_mlp[0].weight)
            self.model.discriminator[0].weight[:, 1950:].zero_()
            self.model.discriminator[0].bias.copy_(network._disc_mlp[0].bias)
            self.model.discriminator[2].load_state_dict(network._disc_mlp[2].state_dict(), strict=True)
            self.model.discriminator[4].load_state_dict(network._disc_logits.state_dict(), strict=True)
        if transfer_disc:
''')
source = replace_once(source, '            amp = reference.discriminator_normalizer\n', '        if transfer_disc or state_prior:\n            amp = reference.discriminator_normalizer\n')
source = replace_once(source, '        if cap is not None:\n', '        if cap is not None or state_prior:\n')
source = replace_once(source, '                normalizer.count.clamp_max_(cap)\n', '                if cap is not None:\n                    normalizer.count.clamp_max_(cap)\n')
source = replace_once(source, '        if self.config.task_terrain_statistics_mode == "preserve_declared_suffix":\n            self.terrain_statistics_policy.bind_public_warm_start(', '''        if state_prior:
            self.provenance["initialization"].update({
                "action_conditioned_discriminator_initialization": STRATEGY,
                "discriminator_initialization": STRATEGY,
                "published_state_only_discriminator_transferred": True,
                "published_amp_history_moments_transferred": True,
                "state_prior_action_extension": dict(EXTENSION)})
            validate_action_state_prior_initialization(self.config, self.provenance["initialization"])
        if self.config.task_terrain_statistics_mode == "preserve_declared_suffix":
            self.terrain_statistics_policy.bind_public_warm_start(''')
# Restrict save admission; update math and execution order remain original.
source = replace_once(source, '        path = Path(path)\n        path.parent.mkdir(parents=True, exist_ok=True)\n', '        validate_action_state_prior_initialization(self.config, self.provenance.get("initialization"))\n        path = Path(path)\n        path.parent.mkdir(parents=True, exist_ok=True)\n')
source = replace_once(source, '        trainer.model.load_state_dict(state["model"], strict=True)\n', '        validate_action_state_prior_initialization(trainer.config, trainer.provenance.get("initialization"))\n        trainer.model.load_state_dict(state["model"], strict=True)\n')
source = replace_once(source, 'Eq2 and must train anew. An explicit state-only public PACER adaptation can\ntransfer its matching complete discriminator and history-coordinate moments.\n', 'Eq2. Its legacy initialization is random; an explicit state-prior strategy\ntransfers public history weights/moments with zero action columns. An explicit\nstate-only adaptation transfers the matching complete public discriminator.\n')
source = replace_once(source, '        and its per-history-coordinate moments without inventing expert actions.\n', '        and its per-history-coordinate moments without inventing expert actions.\n        A separate explicit action strategy extends that state prior with zero\n        action columns and copies the same AMP history moments before count cap.\n')
source = replace_once(source, '    def _observations(self, self_obs, task_obs):\n', '    def _observations(self, self_obs, task_obs):\n        validate_action_state_prior_initialization(self.config, self.provenance.get("initialization"))\n')
source = replace_once(source, '    def _amp_inputs(self, history, applied_actions):\n', '    def _amp_inputs(self, history, applied_actions):\n        validate_action_state_prior_initialization(self.config, self.provenance.get("initialization"))\n')
source = replace_once(source, '    def update(self, rollout, expert_history, expert_actions):\n', '    def update(self, rollout, expert_history, expert_actions):\n        validate_action_state_prior_initialization(self.config, self.provenance.get("initialization"))\n')
proposed[EXISTING[1]] = source
source = (ROOT / EXISTING[2]).read_text()
source = replace_once(source, 'from CrowdSim.humanoid.application.settings import TrainingConfig\n', 'from CrowdSim.humanoid.application.settings import TrainingConfig\nfrom CrowdSim.humanoid.application.state_prior_initialization import validate_action_state_prior_initialization\n')
source = replace_once(source, '    config = validate_reconstruction_config(bundle.get("config"))\n', '    config = validate_reconstruction_config(bundle.get("config"))\n    validate_action_state_prior_initialization(config, bundle["provenance"].get("initialization"))\n')
proposed[EXISTING[2]] = source
source = (ROOT / EXISTING[3]).read_text()
source = replace_once(source, '    records = [inspect_resource("profile", args.config),\n', '    if args.from_scratch and profile["training"].get("warm_start_action_discriminator_from_state_prior", False):\n        raise ResourceError("From-scratch training conflicts with the declared public action state prior; verified public warm start is required")\n    records = [inspect_resource("profile", args.config),\n')
proposed[EXISTING[3]] = source
proposed[NEW_HELPER] = helper
profile = (ROOT / 'configs/diffusion_amp/amp_style_mixed_state_action_terrain_protected.yaml').read_text()
profile = replace_once(profile, '# Staged candidate only: not integrated or trained. Single terrain-statistics strategy variable.\n', '# Staged candidate only: one coupled discriminator/history-RMS initialization strategy.\n')
profile = replace_once(profile, '  discriminator: independent_random_history_and_action\n', '  discriminator: public_pacer_state_prior_zero_action_extension\n  amp_history_moments: same_verified_public_state_prior\n')
profile = replace_once(profile, '  warm_start_discriminator: false\n', '  warm_start_discriminator: false\n  warm_start_action_discriminator_from_state_prior: true\n')
profile = replace_once(profile, '  baseline. Only the discriminator input and initialization change.', '  baseline. Relative to the protected state_action baseline only the coupled D/history-RMS initialization changes.')
profile = replace_once(profile, '  moments. The2019-input history+action discriminator and its1950 history-coordinate RMS begin independently;\n  the public1950 state-only discriminator and AMP moments are not transferred. An existing state-only mixed\n  checkpoint is not valid resume.', '  moments. The2019-input history+action discriminator copies the public1950 history columns, first bias,\n  later layers and head, and initializes all69 additional action columns to zero. The same public1950 AMP\n  history mean/variance/count are copied before the existing count cap. This is one coupled initialization\n  strategy, not a claim that only input dimensions changed. Public prior AMASS overlap is unknown, and these\n  are not author NavIsaacLab weights. An old state-only or random-action checkpoint is not valid resume.')
profile = replace_once(profile, '  representation. Initial history mean0/variance1/count0 are not public transferred statistics; action coordinates\n  are already normalized and receive no learned action RMS.', '  representation. Public AMP mean/variance/count are transferred as part of the new coupled initialization;\n  action coordinates are already normalized and receive no learned action RMS.')
profile = replace_once(profile, 'profile: fixed_slots_amass_procedural_xy_state_action_terrain_statistics_protection\n', 'profile: fixed_slots_amass_procedural_xy_state_action_public_state_prior_terrain_statistics_protection\n')
proposed[NEW_PROFILE] = profile
manifest = {'schema': 'bounded_candidate_patch.v1', 'root': str(ROOT), 'maximum_file_bytes': 120000, 'maximum_total_bytes': 150000, 'targets': []}
patches = []
for relative, after in proposed.items():
    original = ROOT / relative
    before = original.read_bytes() if original.exists() else b''
    if original.exists():
        dest = HERE / 'before' / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(before)
    payload = after.encode()
    dest = HERE / 'proposal' / relative
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(payload)
    manifest['targets'].append({'path': relative, 'before_sha256': hashlib.sha256(before).hexdigest() if original.exists() else None,
                                'before_bytes': len(before) if original.exists() else None,
                                'after_sha256': hashlib.sha256(payload).hexdigest(), 'after_bytes': len(payload)})
    patches.extend(difflib.unified_diff(before.decode().splitlines(keepends=True), after.splitlines(keepends=True),
                                       fromfile='a/'+relative if original.exists() else '/dev/null', tofile='b/'+relative))
(HERE / 'proposed.patch').write_text(''.join(patches))
(HERE / 'merge_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
print(json.dumps(manifest, indent=2))
