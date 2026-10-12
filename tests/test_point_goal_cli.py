"""Run the published train/evaluate CLI with real bounded CPU rollouts."""

import json
import subprocess
import sys

import torch

from crowdsim_ppo import pipeline


def test_default_config_works_in_checkout_and_installed_package(tmp_path, monkeypatch):
    checked_in = pipeline.default_config_path()
    assert (
        checked_in.name == "point_goal.json"
        and pipeline.load_config(None)["environment"]["num_envs"] > 0
    )
    package = tmp_path / "site-packages/crowdsim_ppo"
    (package / "configs").mkdir(parents=True)
    bundled = package / "configs/point_goal.json"
    bundled.write_text('{"installed_fixture":42}')
    monkeypatch.setattr(pipeline, "__file__", str(package / "pipeline.py"))
    assert pipeline.default_config_path() == bundled
    assert pipeline.load_config(None) == {"installed_fixture": 42}


def test_auto_device_selects_available_backend_without_allocating(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert pipeline.resolve_device("auto") == torch.device("cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert pipeline.resolve_device("auto") == torch.device("cuda")


def test_real_module_train_periodic_save_and_evaluate(tmp_path):
    values = {
        "environment": {"num_envs": 1, "num_neighbors": 0, "max_episode_steps": 1},
        "network": {
            "hidden_dims": [8],
            "vector_hidden_dims": [8],
            "actor_hidden_dims": [8],
            "critic_hidden_dims": [8],
        },
        "algorithm": {"ppo_epochs": 1, "minibatch_size": 1},
        "training": {"rollout_steps": 1, "total_steps": 2, "save_interval": 1},
    }
    config = tmp_path / "config.json"
    config.write_text(json.dumps(values))
    output = tmp_path / "run's 中文"
    trained = subprocess.run(
        [
            sys.executable,
            "-m",
            "crowdsim_ppo.train",
            "--config",
            str(config),
            "--output",
            str(output),
            "--device",
            "cpu",
            "--seed",
            "5",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    checkpoint = output / "latest.pt"
    assert trained.stdout.strip().splitlines()[-1] == f"saved: {checkpoint}"
    assert (output / "checkpoint_1.pt").is_file() and (
        output / "checkpoint_2.pt"
    ).is_file()
    assert torch.load(checkpoint, weights_only=False)["step"] == 2
    evaluated = subprocess.run(
        [
            sys.executable,
            "-m",
            "crowdsim_ppo.evaluate",
            "--config",
            str(config),
            "--checkpoint",
            str(checkpoint),
            "--episodes",
            "1",
            "--num-envs",
            "1",
            "--device",
            "cpu",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    result = json.loads(evaluated.stdout)
    assert result["episodes"] == 1 and result["mean_episode_length"] == 1
    assert all(torch.isfinite(torch.tensor(value)) for value in result.values())
