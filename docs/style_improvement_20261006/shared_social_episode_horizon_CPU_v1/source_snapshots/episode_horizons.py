"""CPU accounting of finite AMASS reset windows and shared group episodes."""
from __future__ import annotations

import numpy as np

from .tracking_error_decomposition import require


def reset_prior(frame_counts, future_frames):
    """Ideal uniform clip/then legal-frame law; no RNG or actual exposure claim.

    Corpus._indices uses past9 and floor(uniform * (N-9-future)). Thus
    remaining real controls range from future to N-10 inclusive. The ideal
    discrete-uniform law abstracts the finite precision of torch.rand.
    """
    counts = np.asarray(frame_counts)
    require(counts.ndim == 1 and len(counts) > 0 and counts.dtype == np.int64,
            "Require all integer frame counts, without filtering")
    require(type(future_frames) is int and future_frames >= 1
        and (counts >= 10 + future_frames).all(), "Every clip must retain history10 plus the required real future")
    remaining_max = counts - 10
    room = remaining_max - future_frames + 1
    support = np.arange(int(remaining_max.max()) + 1, dtype=np.int64)
    legal_cdf = np.minimum(np.maximum(support[:, None] - future_frames + 1, 0), room[None]) / room[None]
    cdf = legal_cdf.mean(1)
    require((np.diff(cdf) >= 0).all() and cdf[-1] == 1., "Invalid ideal reset distribution")
    mean = float(((remaining_max + future_frames) / 2).mean())
    require(abs(float((1 - cdf).sum()) - mean) < 1e-10, "Reset-prior mean/CDF identity changed")
    offsets = np.arange(15, 151, 15)
    beyond = np.minimum(np.maximum(offsets[:, None] - future_frames, 0), room[None]) / room[None]
    full_real = np.minimum(np.maximum(remaining_max - 150 + 1, 0), room) / room
    return {"clip_count": len(counts), "required_real_future_frames": future_frames,
        "minimum_real_future_s": future_frames / 30, "all_clips_retained": True,
        "legal_reset_centers": int(room.sum()), "ideal_remaining_mean_s": mean / 30,
        "ideal_two_independent_AMASS_minimum_mean_s": float(np.square(1 - cdf).sum() / 30),
        "ideal_reset_ten_point_beyond_source_fraction": float(beyond.mean()),
        "ideal_reset_full_five_second_real_probability": float(full_real.mean()),
        "cdf_remaining_frames": support.tolist(), "ideal_remaining_cdf": cdf.tolist(),
        "actual_training_exposure_measured": False,
        "finite_precision_RNG_law_abstracted": True}


def group_accounting(statistics, group_rows, post_age, reset_mask):
    """Subtract right-censored final age to recover completed episode mean.

    Reset-group counts exclude the initial reset. Source-associated group
    bounds retain overlap with physical terminal causes; they are not counts
    of episodes ended solely by source exhaustion.
    """
    rows = np.asarray(group_rows)
    post_age, reset_mask = np.asarray(post_age), np.asarray(reset_mask)
    require(post_age.ndim == 2 and post_age.shape[0] > 0 and post_age.shape[1] > 0
        and post_age.dtype == np.int64 and (post_age >= 1).all()
        and reset_mask.shape == post_age.shape and reset_mask.dtype == np.bool_,
        "Require retained POST ages and reset masks")
    actors = post_age.shape[1]
    require(rows.ndim == 2 and rows.shape[1] == 4 and rows.dtype == np.int64
        and np.array_equal(np.sort(rows.reshape(-1)), np.arange(actors)), "Four-person groups must partition every actor")
    require((rows < actors // 2).sum(1).tolist() == [2] * len(rows), "Every actual group must contain two AMASS actors")
    require(np.all(post_age[:, rows] == post_age[:, rows[:, :1]])
        and np.all(reset_mask[:, rows] == reset_mask[:, rows[:, :1]]), "Age/reset must be shared within each whole group")
    for key in ("actor_controls", "reset_groups", "reset_actor_rows", "initial_reset_actor_rows", "source_end_actor_controls"):
        require(type(statistics[key]) is int and statistics[key] >= 0, "Invalid cumulative integer counter: " + key)
    require(statistics["initial_reset_actor_rows"] == actors
        and statistics["reset_actor_rows"] == 4 * statistics["reset_groups"]
        and statistics["actor_controls"] % actors == 0
        and statistics["actor_controls"] >= post_age.size, "Initial/reset/complete-control counters disagree")
    final_age = np.where(reset_mask[-1], 0, post_age[-1])
    require((final_age <= statistics["actor_controls"] // actors).all(), "Final age exceeds cumulative actor controls")
    unfinished = int(final_age[rows[:, 0]].sum())
    all_controls = statistics["actor_controls"] // 4
    completed = all_controls - unfinished
    resets, source_ends = statistics["reset_groups"], statistics["source_end_actor_controls"]
    require(completed >= 0 and source_ends <= 2 * resets
        and int(reset_mask[:, rows[:, 0]].sum()) <= resets, "Captured/cumulative group exposure counters disagree")
    require(resets > 0 or completed == 0, "Completed exposure without an observed group reset")
    return {"groups": len(rows), "actors": actors, "completed_group_episodes": resets,
        "all_group_controls": all_controls, "right_censored_final_group_controls": unfinished,
        "completed_group_controls": completed,
        "completed_group_episode_mean_s": None if resets == 0 else completed / resets / 30,
        "terminal_window_group_resets": int(reset_mask[:, rows[:, 0]].sum()),
        "source_end_actor_controls": source_ends,
        "source_associated_group_reset_lower_bound": (source_ends + 1) // 2,
        "source_associated_group_reset_upper_bound": min(source_ends, resets),
        "source_only_group_reset_count_known": False,
        "final_actor_age": final_age.tolist()}
