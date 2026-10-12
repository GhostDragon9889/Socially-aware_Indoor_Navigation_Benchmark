#!/usr/bin/env python3
"""Run a fresh grouped Sim61 validation, separate from historical experiments.

This exercises the real collector, expert, PPO/AMP update, checkpoint and full
App shutdown. It does not release or claim completion of a paper experiment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]


def validate_grouped_reset(world):
    """Check link/COM writes in real assets before collection, then restore them."""
    import torch
    from protomotions.simulator.base_simulator.simulator_state import ResetState, StateConversion
    from protomotions.simulator.isaaclab.utils.runtime_compat import as_torch
    from protomotions.utils.rotations import quat_apply

    adapter = world.adapter
    before = adapter.snapshot()
    root_fields = ("root_pos", "root_rot", "root_vel", "root_ang_vel")
    fields = {key: getattr(before.root_state, key).clone() for key in root_fields}
    fields.update({key: getattr(before.robot_state, key).clone() for key in ("dof_pos", "dof_vel")})
    original = {key: value.clone() for key, value in fields.items()}
    group_ids = torch.arange(len(world.rows), device=world.device)[::2].flip(0)
    ids = world.rows[group_ids].flatten().flip(0)
    untouched = torch.ones(world.num_envs, dtype=torch.bool, device=world.device)
    untouched[ids] = False
    if not len(ids) or not bool(untouched.any()):
        raise ValueError("Grouped reset validation needs selected and untouched groups")
    row = ids.float()[:, None] * .001
    fields["root_vel"][ids] = torch.tensor([.1, -.2, .3], device=world.device) + row
    fields["root_ang_vel"][ids] = torch.tensor([1.1, -.7, 2.3], device=world.device) + row
    fields["root_rot"][ids] = torch.tensor([0., 0., 2 ** -.5, 2 ** -.5], device=world.device)
    inputs = {key: value.clone() for key, value in fields.items()}
    control, physics = world.control_step, world._counter()
    runtime = {key: getattr(world, key).clone() for key in
               ("age", "episode_ids", "amp_history", "amp_history_episode_ids", "plan_age", "plans", "plan_valid")}
    errors, restore_errors, corrections = {}, {}, []

    def compare(actual, expected, label):
        error = float((actual - expected).abs().max())
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-5,
                                   msg=lambda message: f"{label}: {message}")
        return error

    try:
        adapter.write_common_reset(ResetState(state_conversion=StateConversion.COMMON, **fields), ids)
        world.sim.forward()
        actual = adapter.snapshot()
        for key in fields:
            state = actual.root_state if key in root_fields else actual.robot_state
            value = getattr(state, key)
            errors[key] = compare(value[ids], inputs[key][ids], "Grouped reset " + key)
            compare(value[untouched], original[key][untouched], "Untouched group " + key)
            assert torch.equal(fields[key], inputs[key]), "Reset input mutated: " + key
        for slot, asset in enumerate(world.assets):
            offset = as_torch(asset.data.body_com_pos_b)[group_ids, 0]
            rotation = as_torch(asset.data.root_link_quat_w)[group_ids]
            omega = inputs["root_ang_vel"][world.rows[group_ids, slot]]
            correction = torch.cross(omega, quat_apply(rotation, offset, w_last=True), dim=-1)
            corrections.append(float(correction.abs().max()))
        assert all(value > 1e-5 for value in corrections), "COM velocity conversion was not exercised"
    finally:
        adapter.write_common_reset(ResetState(state_conversion=StateConversion.COMMON,
            **{key: value.clone() for key, value in original.items()}),
            torch.arange(world.num_envs, device=world.device))
        world.sim.forward()
        world.current = adapter.snapshot()
        for key in original:
            state = world.current.root_state if key in root_fields else world.current.robot_state
            restore_errors[key] = compare(getattr(state, key), original[key], "Restored initial " + key)
        assert world.control_step == control and world._counter() == physics
        for key, value in runtime.items():
            assert torch.equal(getattr(world, key), value), "Reset probe changed World state: " + key
    return {"passed": True, "actors": world.num_envs, "assets": len(world.assets),
            "selected_group_ids": group_ids.cpu().tolist(), "selected_actor_ids": ids.cpu().tolist(),
            "max_absolute_errors": errors, "restored_initial_max_absolute_errors": restore_errors,
            "max_COM_velocity_corrections_m_s": corrections,
            "input_state_unchanged": True, "untouched_groups_unchanged": True,
            "physical_steps_advanced": 0, "performed_before_collection": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--variant", choices=("episode31", "future81"), required=True)
    parser.add_argument("--actors", type=int, choices=(8, 256), default=8)
    parser.add_argument("--controls", type=int, choices=(4, 32), default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.variant == "episode31":
        from scripts import train_shared_social_episode_reference as entry
    else:
        from scripts import train_shared_social_minimum_horizon as entry
    from CrowdSim.humanoid.application import episode_reference_clock_grid as clocks
    from CrowdSim.humanoid.runtime import grouped_source_migration as migration
    from CrowdSim.humanoid.runtime.social_evaluation_lifecycle import record

    module, binding = entry.prepare_entry()
    profile = json.loads((entry.PROFILE_DIR / ("smoke.json" if args.actors == 8 else "smoke256.json")).read_text())
    profile["runtime"].update(rollout_controls=args.controls, episode_controls=args.controls)
    profile["runtime"]["archive_controls_before_reset"] = args.controls == 4
    output = args.output.resolve()
    # The parent owns creation/refusal of the actual output directory.
    profile_path = output.with_name(output.name + ".profile.json")
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    with profile_path.open("x") as stream:
        json.dump(profile, stream, indent=2)
        stream.write("\n")
    profile_path.chmod(0o444)
    profile_record = record(profile_path)
    parent_load, parent_run = module._load, module._run
    expected_profile, _ = module._profile(entry.PROFILE_DIR / ("smoke.json" if args.actors == 8 else "smoke256.json"))
    if expected_profile["runtime"]["actors"] != args.actors:
        raise ValueError("Validation actors disagree with the registered baseline")
    validation_report = {}

    def load(path, name, identity):
        loaded = parent_load(path, name, identity)
        if Path(path) == module.PINS["probe_setup"][0] and args.variant == "episode31":
            loaded = clocks.adapt_probe(loaded)
        for key in ("expert", "collector"):
            if Path(path) == module.PINS[key][0]:
                # Validate the complete original declaration before selecting
                # the separately sealed World. Other roles remain unchanged.
                migration.pins_for(path, loaded.PINS)
                world = binding["World"]
                selected = (entry.WORLD, world["bytes"], world["sha256"], True)
                if key == "expert":
                    selected = (str(entry.WORLD.relative_to(ROOT)), *selected[1:3])
                loaded.PINS = {**loaded.PINS, "world": selected}
                if key == "collector":
                    original_init = loaded.SocialRolloutCollector.__init__

                    def checked_init(self, world, trainer):
                        original_init(self, world, trainer)
                        validation_report["grouped_reset_validation"] = validate_grouped_reset(world)

                    loaded.SocialRolloutCollector.__init__ = checked_init
        return loaded

    def checked_profile(path):
        if Path(path).resolve() != profile_path or record(path) != profile_record:
            raise ValueError("Grouped validation profile changed")
        return profile, profile_record

    def run(*values):
        nonlocal validation_report
        validation_report = values[3]
        values[3]["grouped_compatibility_validation"] = {
            "entry": record(__file__), "variant": args.variant,
            "experiment_owner_completion_claimed": False,
            "historical_output_overwritten": False}
        return parent_run(*values)

    module._load, module._profile, module._run = load, checked_profile, run
    saved = sys.argv
    try:
        sys.argv = [str(Path(__file__)), "--config", str(profile_path), "--output", str(output), "--device", "cuda:0"]
        return module.main()
    finally:
        sys.argv = saved


if __name__ == "__main__":
    raise SystemExit(main())
