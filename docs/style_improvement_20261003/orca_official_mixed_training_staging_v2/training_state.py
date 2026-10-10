"""Explicit optimizer-update accumulation/EMA checkpoint semantics.

This module leaves the reused five-state NN/DDPM/loss implementation unchanged.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
import random

import numpy as np
import torch

from paper_interact import PAPER_SCHEMA

PHASE = "official_maps_interact_five_state_EMA_accumulation_v2"
EMA_RECIPE = {"decay": .995, "every_optimizer_updates": 1, "start_optimizer_update": 2000,
              "timing": "before_first_micro_forward_of_optimizer_update",
              "warmup": "whole_state_dict_copy", "state": "whole_state_dict_including_buffers",
              "integer_buffers": "author_float_blend_copy_cast"}
CHECKPOINT_KIND = "local_five_state.full_raw_EMA_accumulation.v2"


class WholeStateEMA:
    def __init__(self, raw_model):
        self.model = copy.deepcopy(raw_model)
        self.model.requires_grad_(False)
        self.before_update_events = 0

    def before_update(self, raw_model, update_index):
        if type(update_index) is not int or update_index != self.before_update_events:
            raise ValueError("EMA events must follow exact optimizer-update order")
        with torch.no_grad():
            if update_index < EMA_RECIPE["start_optimizer_update"]:
                self.model.load_state_dict(raw_model.state_dict(), strict=True)
            else:
                ema_state = self.model.state_dict()
                for key, value in raw_model.state_dict().items():
                    target = ema_state[key]
                    # Exact author operation, including integer-buffer copy cast.
                    target.copy_(EMA_RECIPE["decay"] * target + (1. - EMA_RECIPE["decay"]) * value)
        self.before_update_events += 1

    def metadata(self):
        return {"recipe": copy.deepcopy(EMA_RECIPE), "before_update_events": self.before_update_events,
                "last_pre_update_index": self.before_update_events - 1 if self.before_update_events else None}


def recipe_identity(*, microbatch, accumulation, learning_rate, max_grad_norm, missing_map_encoding):
    if type(microbatch) is not int or type(accumulation) is not int or min(microbatch, accumulation) < 1:
        raise ValueError("Positive exact integer microbatch and accumulation required")
    return {"phase": PHASE, "microbatch": microbatch, "accumulation": accumulation,
            "effective_batch": microbatch * accumulation, "learning_rate": learning_rate,
            "max_grad_norm": max_grad_norm, "missing_map_encoding": missing_map_encoding,
            "EMA": copy.deepcopy(EMA_RECIPE), "loss_accumulation": "mean_micro_loss_divided_by_accumulation",
            "clip_scope": "once_on_complete_accumulated_gradient",
            "optimizer": "Adam_default_betas_eps_weight_decay0",
            "BN_scope": "microbatch_not_effective_batch", "default_inference_weights": "EMA"}


def rng_state(device):
    numpy_rng = np.random.get_state()
    return {"torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            "python": random.getstate(),
            "numpy": [numpy_rng[0], numpy_rng[1].tolist(), numpy_rng[2], numpy_rng[3], numpy_rng[4]]}


def restore_rng(state, device):
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    value = state["numpy"]
    np.random.set_state((value[0], np.asarray(value[1], dtype=np.uint32), value[2], value[3], value[4]))
    if state["cuda"]:
        if device.type != "cuda" or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Saved CUDA RNG topology differs")
        torch.cuda.set_rng_state_all(state["cuda"])


def validate_resume(artifact, *, config, identity, corpus_identity):
    if (artifact.get("schema") != PAPER_SCHEMA or artifact.get("checkpoint_kind") != CHECKPOINT_KIND
            or artifact.get("config") != asdict(config) or artifact.get("training_recipe") != identity
            or artifact.get("dataset_metadata", {}).get("corpus_identity") != corpus_identity):
        raise ValueError("Full raw/EMA recipe, model or corpus resume identity differs")
    updates = artifact.get("optimizer_updates")
    if (type(updates) is not int or updates < 1 or type(artifact.get("step")) is not int
            or artifact.get("step") != updates):
        raise ValueError("Invalid complete optimizer-update count")
    if (any(type(artifact.get(key)) is not int for key in
            ("microsteps", "training_windows", "microsteps_in_current_update"))
            or artifact.get("microsteps") != updates * identity["accumulation"]
            or artifact.get("training_windows") != updates * identity["effective_batch"]
            or artifact.get("microsteps_in_current_update") != 0
            or type(artifact.get("EMA", {}).get("before_update_events")) is not int
            or type(artifact.get("EMA", {}).get("last_pre_update_index")) is not int
            or artifact.get("EMA") != {"recipe": EMA_RECIPE, "before_update_events": updates,
                                      "last_pre_update_index": updates - 1}):
        raise ValueError("Incomplete accumulation or EMA counters")
    if not isinstance(artifact.get("training_raw_model"), dict) or not isinstance(artifact.get("model"), dict):
        raise ValueError("Both raw training state and EMA deployment state are required")
    def finite_tensors(value):
        if isinstance(value, torch.Tensor):
            return bool(torch.isfinite(value).all())
        if isinstance(value, dict):
            return all(finite_tensors(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return all(finite_tensors(item) for item in value)
        return True
    if not all(finite_tensors(artifact.get(key)) for key in ("training_raw_model", "model", "optimizer")):
        raise ValueError("Nonfinite raw/EMA/Adam checkpoint state")


def optimizer_update(planner, ema, optimizer, corpus, *, make_batch, identity, update_index):
    """One Adam update; same original loss, explicit accumulation and optional clip."""
    ema.before_update(planner.model, update_index)
    optimizer.zero_grad(set_to_none=True)
    metrics, domain_counts = [], {"maps": 0, "interact": 0}
    for _ in range(identity["accumulation"]):
        sample = corpus.diffusion_batch(identity["microbatch"])
        batch = make_batch(sample)
        loss, values = planner.training_loss(batch.clean_trajectory, batch.context, drop_conditioning=True, training=True)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite micro loss; optimizer not stepped")
        (loss / identity["accumulation"]).backward()
        metrics.append({"loss": float(loss.detach()), **{key: float(value) for key, value in values.items()}})
        for name in domain_counts:
            domain_counts[name] += sample["domain_names"].count(name)
    gradients = [parameter.grad for parameter in planner.model.parameters() if parameter.grad is not None]
    if not all(bool(torch.isfinite(gradient).all()) for gradient in gradients):
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError("Nonfinite accumulated gradient; optimizer not stepped")
    if identity["max_grad_norm"] is None:
        grad_norm = torch.sqrt(sum((gradient.square().sum() for gradient in gradients),
                                  torch.zeros((), device=planner.device)))
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(planner.model.parameters(), identity["max_grad_norm"],
                                                  error_if_nonfinite=True)
    optimizer.step()
    planner.trained_steps = update_index + 1
    return {key: sum(row[key] for row in metrics) / len(metrics) for key in metrics[0]} | {
        "optimizer_updates": planner.trained_steps, "grad_norm_before_optional_clip": float(grad_norm),
        "microsteps": planner.trained_steps * identity["accumulation"],
        "training_windows": planner.trained_steps * identity["effective_batch"],
        "batch_domain_counts": domain_counts}


def checkpoint_payload(planner, ema, optimizer, corpora, *, identity, corpus_identity, initialization):
    updates = planner.trained_steps
    if updates < 1 or ema.before_update_events != updates:
        raise ValueError("Only complete optimizer-update boundaries may be saved")
    return {"schema": PAPER_SCHEMA, "checkpoint_kind": CHECKPOINT_KIND, "config": asdict(planner.config),
            "model": ema.model.state_dict(),  # Existing deployment Planner's model key is EMA.
            "training_raw_model": planner.model.state_dict(), "optimizer": optimizer.state_dict(),
            "step": updates, "optimizer_updates": updates, "microsteps": updates * identity["accumulation"],
            "training_windows": updates * identity["effective_batch"], "microsteps_in_current_update": 0,
            "training_recipe": identity, "EMA": ema.metadata(),
            "dataset_metadata": {"phase": PHASE, "corpus_identity": corpus_identity,
                                 "initialization": initialization,
                                 "sampler_rng": {name: copy.deepcopy(corpus.generator.bit_generator.state)
                                                 for name, corpus in corpora.items()}},
            "rng": rng_state(planner.device), "inference_weight_source": "EMA",
            "training_resume_requires": "training_raw_model_plus_raw_Adam_and_complete_EMA_recipe",
            "reference": "Independent Nav five-state adaptation; author TRACE architecture and data identity not claimed"}


def restore_training(artifact, planner, ema, optimizer, corpora, *, identity, corpus_identity):
    validate_resume(artifact, config=planner.config, identity=identity, corpus_identity=corpus_identity)
    # Never combine EMA deployment model with raw Adam as a training resume.
    planner.model.load_state_dict(artifact["training_raw_model"], strict=True)
    ema.model.load_state_dict(artifact["model"], strict=True)
    optimizer.load_state_dict(artifact["optimizer"])
    planner.trained_steps = artifact["optimizer_updates"]
    ema.before_update_events = artifact["EMA"]["before_update_events"]
    for name, corpus in corpora.items():
        corpus.generator.bit_generator.state = artifact["dataset_metadata"]["sampler_rng"][name]
    restore_rng(artifact["rng"], planner.device)
