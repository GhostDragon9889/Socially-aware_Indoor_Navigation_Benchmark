"""Source-only opt-in gates and exact negative-baseline learning config."""

from dataclasses import asdict
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

from CrowdSim.humanoid.application.configuration import load_amp_profile, parse_amp_args, validate_training_source
from CrowdSim.humanoid.application.settings import TrainingConfig


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT/'configs/diffusion_amp/amp_style_amass_root_curriculum.yaml'


def argv(tmp_path, *extra):
    checkpoint = tmp_path/'source-only-CLI.ckpt'
    checkpoint.write_bytes(b'CPU existence contract, not actual weights')
    return ['--diffusion-checkpoint', str(checkpoint), '--output', str(tmp_path/'new'), *extra]


def test_every_canonical_training_field_equals_negative_baseline_and_expert_sampling_disabled():
    baseline = load_amp_profile(ROOT/'configs/diffusion_amp/amp_style_amass_backward_penalty.yaml')
    new = load_amp_profile(PROFILE)
    assert asdict(TrainingConfig(**baseline['training'])) == asdict(TrainingConfig(**new['training']))
    assert new['training']['expert_sampling'] is None
    assert new['trajectory_training']['source'] == 'amass_root'
    assert new['trajectory_training']['minimum_real_future_frames'] == 31


@pytest.mark.parametrize('flags', [(), ('--trajectory-source', 'pacer_curriculum', '--terrain-extent-m', '160', '--spawn-margin-m', '80'),
    ('--evaluate-steps', '10', '--resume', 'actor.pt', '--diffusion-guidance-mode', 'trace_clean')])
def test_every_previous_default_cli_field_is_exactly_unchanged(tmp_path, flags):
    source = ROOT/'docs/style_improvement_20261002/amass_root_curriculum_before/CrowdSim/humanoid/application/configuration.py'
    spec = importlib.util.spec_from_file_location('CrowdSim.humanoid.application.amass_before', source)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    args = argv(tmp_path, *flags)
    assert vars(parse_amp_args(args)) == vars(old.parse_amp_args(args))


@pytest.mark.parametrize('flags', [('--evaluate-steps', '10', '--resume', 'actor.pt'),
    ('--compile-diffusion-forward',), ('--diffusion-guidance-mode', 'trace_clean'),
    ('--guidance-weight', '20'), ('--curriculum-mode', 'pacer_knots'),
    ('--curriculum-speed-max-mps', '2'), ('--curriculum-accel-max-mps2', '1.5')])
def test_curriculum_refuses_deployment_or_procedural_mutations(tmp_path, flags):
    with pytest.raises(SystemExit) as error:
        parse_amp_args(argv(tmp_path, '--trajectory-source', 'amass_root', *flags))
    assert error.value.code == 2


def test_profile_training_source_is_explicit_but_deployment_remains_diffusion(tmp_path):
    profile = load_amp_profile(PROFILE)
    training = parse_amp_args(argv(tmp_path, '--trajectory-source', 'amass_root'))
    validate_training_source(training, profile)
    with pytest.raises(ValueError):
        validate_training_source(parse_amp_args(argv(tmp_path)), profile)
    evaluation = parse_amp_args(argv(tmp_path, '--evaluate-steps', '10', '--resume', 'actor.pt'))
    validate_training_source(evaluation, profile)
    assert evaluation.trajectory_source == 'diffusion'


def test_new_profile_and_cli_preflight_gate_import_no_torch_or_sdk(tmp_path):
    args = argv(tmp_path, '--trajectory-source', 'amass_root')
    code = f"""
import sys
sys.path.insert(0, {str(ROOT)!r})
from CrowdSim.humanoid.application.configuration import parse_amp_args, load_amp_profile, validate_training_source
args=parse_amp_args({args!r})
validate_training_source(args,load_amp_profile({str(PROFILE)!r}))
assert not any(n=='torch' or n.startswith(('isaacsim','isaaclab','omni','pxr')) for n in sys.modules)
"""
    subprocess.run([sys.executable, '-I', '-c', code], check=True, capture_output=True, text=True)


def test_missing_or_invalid_source_declaration_rejected_before_sdk(tmp_path):
    training = parse_amp_args(argv(tmp_path, '--trajectory-source', 'amass_root'))
    with pytest.raises(ValueError, match='explicit profile'):
        validate_training_source(training, {})
    profile = load_amp_profile(PROFILE)
    profile['trajectory_training']['tail_policy'] = []
    with pytest.raises(ValueError, match='tail_policy'):
        validate_training_source(training, profile)


def test_older_missing_declaration_remains_supported_in_original_modes(tmp_path):
    for source in ('diffusion', 'pacer_curriculum'):
        args = parse_amp_args(argv(tmp_path, '--trajectory-source', source, '--terrain-extent-m', '192', '--spawn-margin-m', '48'))
        validate_training_source(args, {})
