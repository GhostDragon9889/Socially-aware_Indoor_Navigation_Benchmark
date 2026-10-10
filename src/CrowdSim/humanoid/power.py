"""Declared joint-power measurement identities and explicit training stages."""
from __future__ import annotations


LEGACY_POWER = "legacy_pd_proxy_misaligned"
PROJECTED_POWER = "projected_joint_effort_substep_mean"


def power_contract(source):
    if not isinstance(source, str):
        raise ValueError("Joint power measurement source must be a string")
    if source == LEGACY_POWER:
        return {"source": source, "effort": "ImplicitActuator pre-step clipped PD estimate",
                "velocity": "last substep post-step angular velocity",
                "timestamp_aligned": False, "reduction": "last substep only",
                "units": "W", "isolated_motor_drive_torque": False}
    if source == PROJECTED_POWER:
        return {"source": source,
                "effort": "PhysX get_dof_projected_joint_forces: incoming joint wrench projected onto active DOFs, averaged over the last simulation substep",
                "velocity": "same substep end angular velocity",
                "timestamp_aligned": True,
                "reduction": "arithmetic time average of four equal 120Hz substep effort*velocity sums per 30Hz control step",
                "quadrature": "substep-average effort paired with end-of-substep velocity",
                "units": "W", "isolated_motor_drive_torque": False}
    raise ValueError("Unknown joint power measurement source")


def power_stage(saved_contract, requested_source, *, explicit_stage=False, evaluation=False):
    """Reject implicit reward changes; old unannotated bundles have known legacy semantics."""
    previous = power_contract(LEGACY_POWER) if saved_contract is None else saved_contract
    if not isinstance(previous, dict) or type(explicit_stage) is not bool or type(evaluation) is not bool:
        raise ValueError("Power stage requires a dictionary contract and boolean transition options")
    if previous != power_contract(previous.get("source")):
        raise ValueError("Saved power measurement contract is invalid or changed")
    current = power_contract(requested_source)
    changed = previous != current
    if explicit_stage and (evaluation or not changed or previous["source"] != LEGACY_POWER or requested_source != PROJECTED_POWER):
        raise ValueError("A new power stage requires legacy-to-projected training resume")
    if changed and not (evaluation or explicit_stage):
        raise ValueError("Resume power measurement changed; use --new-power-stage explicitly")
    return previous, current, changed
