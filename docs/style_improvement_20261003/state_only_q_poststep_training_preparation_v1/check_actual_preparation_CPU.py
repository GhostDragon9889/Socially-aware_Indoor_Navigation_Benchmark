"""One actual copied-source/config/resources/import gate, without Torch or SDK."""
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys
from prepare_experiment import HERE, NEW, OLD, PROFILE, STAGED, record, write


def main():
    spec = importlib.util.spec_from_file_location('poststep_actual_isolated_launcher', NEW / 'launch_isolated.py')
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    source = launcher.verify()  # Existing strict source/links and6 opaque input hashes; no deserialization.
    before = {r['path']:r for r in json.loads((OLD / 'source_manifest.json').read_text())['files']}
    after = {r['path']:r for r in source['files']}
    changed = sorted(name for name in before if before[name]['sha256'] != after[name]['sha256'])
    added = sorted(set(after) - set(before))
    assert changed == ['code/scripts/train_paper_amp.py'] and added == [PROFILE]
    assert all((NEW / name).stat().st_nlink == 1 and not (NEW / name).stat().st_mode & 0o222
               and (NEW / name).stat().st_ino != (OLD / name).stat().st_ino for name in before)
    plan = json.loads((HERE / 'root_GPU_launch_commands.json').read_text())
    ledger = launcher.ImportLedger(HERE / 'actual_CPU_imports_final.json')
    launcher.activate(ledger)
    from CrowdSim.humanoid.application.configuration import load_amp_profile, parse_amp_args
    from CrowdSim.humanoid.application.resources import preflight_amp_resources
    from CrowdSim.humanoid.application.settings import TrainingConfig
    profile = load_amp_profile(NEW / PROFILE)
    parent = load_amp_profile(OLD / 'code/configs/diffusion_amp/isolated_state_only_matched_q_absolute.yaml')
    config = TrainingConfig(**profile['training'])
    assert profile['training'] == parent['training'] and profile['reward'] == parent['reward']
    assert profile['initialization'] == parent['initialization']
    assert profile['trajectory_training'] == parent['trajectory_training']
    assert profile['terrain_statistics'] == parent['terrain_statistics']
    assert config.discriminator_input == 'state' and config.history_frames * config.amp_frame_dim == 1950
    args_by_phase = {}
    for name, command in plan['commands'].items():
        args = parse_amp_args(command[command.index('--') + 1:])
        assert args.resume is None and not args.from_scratch and args.seed == 7
        resources = preflight_amp_resources(args, profile)
        assert Path(resources.require('warm_start_checkpoint')).resolve() == Path(profile['initialization']['checkpoint']).resolve()
        args_by_phase[name] = vars(args)
    assert not any(n.split('.')[0] in {'torch','isaacsim','isaaclab','omni','pxr','warp'} for n in sys.modules)
    ledger.finish('passed_source_config_resource_preflight')
    document = {'schema':'state_only_q.poststep_actual_CPU_preparation.v1','status':'passed',
        'source_manifest':record(NEW / 'source_manifest.json'),'source_count':len(after),
        'parent_source':record(OLD / 'source_manifest.json'),'modified_parent_paths':changed,'added_paths':added,
        'actual_selected_profile':record(NEW / PROFILE),'training_config':asdict(config),
        'configuration_machine_fields_exact_q936':True,'D_input':1950,
        'AMP_reward_timestamp':'post-step h[t+1..] before resets',
        'actual_parser_arguments':args_by_phase,'actual_fresh_parser_and_resource_gates_passed':True,
        'full_opaque_resource_hashes_verified_by_unchanged_launcher':True,'opaque_resource_count':len(source['resources']),
        'q_bank_not_rebuilt':True,'actual_CPU_imports':record(HERE / 'actual_CPU_imports_final.json'),
        'all_observed_owned_modules_new_isolated_bundle':True,
        'approved_timing_CPU_source_freeze':record(STAGED / 'source_freeze_v1.json'),
        'models_deserialized':False,'Torch_NN_SDK_GPU_started':False,'sealed34_outputs_read':False,
        'actual_training_payload_admitted':False,'future_effect':False}
    write(HERE / 'actual_CPU_readiness.json', document)
    print(json.dumps({'status':'passed','source_count':len(after),'opaque_resources':len(source['resources']),
        'actual_CPU_readiness':record(HERE / 'actual_CPU_readiness.json')}))


if __name__ == '__main__':
    main()
