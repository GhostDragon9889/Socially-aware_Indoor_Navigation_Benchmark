"""Nested tensor shape operations used by the TRACE inference port."""

import math

import torch


def _map(value, fn):
    if isinstance(value, dict):
        return {key: _map(item, fn) for key, item in value.items()}
    if isinstance(value, list):
        return [_map(item, fn) for item in value]
    if isinstance(value, tuple):
        return tuple(_map(item, fn) for item in value)
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        raise TypeError("TRACE shape utility requires tensors or nested tensor containers")
    return fn(value)


def reshape_dimensions(value, begin_axis, end_axis, target_dims):
    return _map(value, lambda x: x.reshape(*x.shape[:begin_axis], *target_dims, *x.shape[end_axis:]))


def join_dimensions(value, begin_axis, end_axis):
    return _map(value, lambda x: x.reshape(*x.shape[:begin_axis], math.prod(x.shape[begin_axis:end_axis]), *x.shape[end_axis:]))


def repeat_by_expand_at(value, repeats, dim):
    return _map(value, lambda x: x.repeat_interleave(repeats, dim=dim))
