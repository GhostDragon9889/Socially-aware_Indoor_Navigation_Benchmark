"""CPU regressions for truthful labels, without simulator/checkpoint imports."""

import json

import pytest

from scripts import summarize_paper_training as summary


@pytest.mark.parametrize("assumptions,mode,kind,diffusion_title,amp_title", [
    ({"dataset": "Individual AMASS root trajectories",
      "neighbors": "Empty: AMASS supplies no synchronized observations",
      "raster": "Uniform walkable=1, obstacles=0"},
     "state_action", "amass_individual", "AMASS root-trajectory", "Action-conditioned"),
    ({"data": "Pinned TRACE/RVO2 ORCA experts with actual obstacle maps and synchronized neighbors",
      "conditions": "Real neighbors and their valid masks; local occupancy"},
     "state", "orca_social", "ORCA social-trajectory", "State-only"),
    ({}, None, "unknown", "Diffusion: raw training metrics", "input mode unverified"),
])
def test_actual_plots_and_report_use_input_declarations(
    tmp_path, monkeypatch, assumptions, mode, kind, diffusion_title, amp_title,
):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    titles = []
    savefig = Figure.savefig

    def record_title(figure, *args, **kwargs):
        titles.extend(ax.get_title() for ax in figure.axes)
        if figure._suptitle is not None:
            titles.append(figure._suptitle.get_text())
        return savefig(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", record_title)
    diffusion = {"status": "completed", "completed_step": 2, "assumptions": assumptions,
                 "initialization": {"path": "AMASS_initial_checkpoint.ckpt"},
                 "metrics": [{"step": 1, "loss": .2}, {"step": 2, "loss": .1}]}
    result = {"status": "passed", "update_count": 2,
              "training_config": {"expert_action_semantics": "inverse_pd_target_from_mocap_pose"},
              "arguments": {"config": "/recorded/profile.yaml"}}
    if mode is not None:
        result["training_config"]["discriminator_input"] = mode
    diffusion_path = tmp_path / "AMASS_named_result.json"
    amp_path = tmp_path / "amp_result.json"
    metrics_path = tmp_path / "metrics.jsonl"
    diffusion_path.write_text(json.dumps(diffusion))
    amp_path.write_text(json.dumps(result))
    metrics_path.write_text("\n".join(json.dumps({"update_count": step, "elapsed_s": step})
                                      for step in (1, 2)) + "\n")
    out = tmp_path / "plots"
    summary.main(["--diffusion-result", str(diffusion_path), "--amp-result", str(amp_path),
                  "--amp-metrics", str(metrics_path), "--output-dir", str(out)])
    report = json.loads((out / "summary.json").read_text())
    assert report["diffusion"]["label_metadata"]["dataset_kind"] == kind
    assert report["amp"]["label_metadata"]["discriminator_input"] == (mode or "unknown")
    assert report["amp"]["label_metadata"]["profile_path"] == "/recorded/profile.yaml"
    assert any(diffusion_title in title for title in titles)
    assert any(amp_title in title for title in titles)
    assert report["sources"]["diffusion_result"]["sha256"]
    assert report["sources"]["amp_result"]["sha256"]
    assert len(report["artifacts"]) == 3
    assert all(artifact["bytes"] > 1000 for artifact in report["artifacts"])
    markdown = (out / "report.md").read_text()
    assert kind in markdown and (mode or "unknown") in markdown
    if kind != "amass_individual":
        assert "AMASS supplies no" not in markdown
        assert "individual AMASS with empty neighbors" not in "\n".join(report["limitations"])


def test_dataset_labels_ignore_initialization_and_preserve_conflicting_data_evidence():
    assert summary.diffusion_labels({"initialization": {"path": "amass.ckpt"}})["dataset_kind"] == "unknown"
    result = {"assumptions": {"dataset": "Individual AMASS trajectories", "data": "ORCA social experts"}}
    labels = summary.diffusion_labels(result)
    assert labels["dataset_kind"] == "unknown"
    assert labels["declarations"] == [
        {"field": "assumptions.dataset", "value": "Individual AMASS trajectories"},
        {"field": "assumptions.data", "value": "ORCA social experts"},
    ]


@pytest.mark.parametrize("result", [
    {"update_count": 1, "training_config": {"discriminator_input": "state"}},
    {"update_count": 2, "training_config": {"discriminator_input": "state"},
     "initialization": {"discriminator_input": "state_action"}},
    {"update_count": 2, "training_config": {"discriminator_input": True},
     "initialization": {"discriminator_input": "state"}},
    {"update_count": 2, "training_config": {"expert_action_semantics": "inverse_pd_target_from_mocap_pose"}},
])
def test_discriminator_labels_require_explicit_current_consistent_mode(result):
    assert summary.amp_labels(result, [{"update_count": 2}])["discriminator_input"] == "unknown"


def test_jsonl_incomplete_tail_and_resume_session_contracts_are_preserved(tmp_path):
    path = tmp_path / "metrics.jsonl"
    records = [{"update_count": 1, "elapsed_s": 1, "resets": 0},
               {"update_count": 2, "elapsed_s": 2, "resets": 4},
               {"update_count": 3, "elapsed_s": .1, "resets": 0}]
    path.write_text("\n".join(json.dumps(row) for row in records) + '\n{"update_count":')
    loaded, provenance = summary.read_metrics(path)
    assert loaded == records
    assert provenance["incomplete_final_record_ignored"] is True
    sessions, boundaries = summary.split_sessions(loaded)
    assert sessions == [records[:2], records[2:]]
    assert boundaries == [{"before_update": 3, "reasons": ["elapsed_s restarted", "resets restarted"]}]
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="Malformed complete JSONL record"):
        summary.read_metrics(path)
