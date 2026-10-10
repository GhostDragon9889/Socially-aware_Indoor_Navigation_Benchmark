"""Required tiny Conv/BN/AdamW checks; no project neural-network construction."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def main():
    started = time.perf_counter()
    sys.path.insert(0, str(ROOT))
    import torch
    from torch import nn
    from convert_official_raw import record

    torch.set_num_threads(4)
    torch.manual_seed(17)
    spec = importlib.util.spec_from_file_location("interact_map_skip_actual", HERE / "paper_interact.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    checks = []

    def check(name, condition):
        assert condition, name
        checks.append({"name": name, "passed": True})

    class TinyMapEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(2, 4, 4, stride=4, bias=False)
            self.bn = nn.BatchNorm2d(4)
            self.calls = []

        @staticmethod
        def feat_map_out_dim(height, width):
            return height / 4, width / 4

        def forward(self, raster):
            self.calls.append(raster.detach().clone())
            return None, self.bn(self.conv(raster))

    def host(encoder):
        return SimpleNamespace(config=module.PaperDiffusionConfig(map_grid_feature_dim=4), normalize=lambda value: value,
            ego_encoder=lambda value, mask: value.new_zeros((len(value), 128)),
            neighbor_encoder=lambda value, mask: value.new_zeros((len(value), 128)),
            conditioning=lambda value: value, map_encoder=encoder)

    raster = torch.arange(3 * 2 * 224 * 224, dtype=torch.float32).reshape(3, 2, 224, 224) / (3 * 2 * 224 * 224)
    context = {"history": torch.zeros(3, 31, 5), "history_mask": torch.ones(3, 31, dtype=torch.bool),
               "neighbors": torch.zeros(3, 19, 31, 5), "neighbor_mask": torch.ones(3, 19, 31, dtype=torch.bool), "raster": raster}
    original = TinyMapEncoder()
    alltrue = deepcopy(original)
    legacy_grid = module.PaperDiffusionModel.encode_context(host(original), context)["map_grid"]
    flagged_grid = module.PaperDiffusionModel.encode_context(host(alltrue), {**context, "map_condition_mask": torch.ones(3, dtype=torch.bool)})["map_grid"]
    check("alltrue_matches_legacy_actual_result_and_buffers", torch.equal(legacy_grid, flagged_grid)
          and all(torch.equal(value, alltrue.state_dict()[key]) for key, value in original.state_dict().items()))
    check("legacy_alltrue_forward_exact_full_batch", len(original.calls) == len(alltrue.calls) == 1
          and torch.equal(original.calls[0], raster) and torch.equal(alltrue.calls[0], raster))

    mixed = TinyMapEncoder()
    available_only = deepcopy(mixed)
    availability = torch.tensor([True, False, True])
    mixed_grid = module.PaperDiffusionModel.encode_context(host(mixed), {**context, "map_condition_mask": availability})["map_grid"]
    expected_grid = available_only(raster[availability])[1]
    check("mixed_missing_rows_never_enter_encoder", len(mixed.calls) == 1
          and torch.equal(mixed.calls[0], raster[availability]) and mixed.calls[0].shape[0] == 2)
    check("mixed_scatter_matches_available_only_BN_buffers", torch.equal(mixed_grid[availability], expected_grid)
          and (mixed_grid[~availability] == 0).all()
          and all(torch.equal(value, available_only.state_dict()[key]) for key, value in mixed.state_dict().items()))

    encoder = TinyMapEncoder()
    auxiliary = nn.Parameter(torch.tensor(1.))
    optimizer = torch.optim.AdamW([*encoder.parameters(), auxiliary], lr=.01, weight_decay=.1)
    # Populate real Adam moments before an all-missing batch, then retain them unchanged.
    warm_grid = module.PaperDiffusionModel.encode_context(host(encoder), context)["map_grid"]
    (warm_grid.square().mean() + auxiliary.square()).backward()
    optimizer.step()
    before = {name: value.detach().clone() for name, value in encoder.state_dict().items()}
    map_optimizer_before = {parameter: deepcopy(optimizer.state[parameter]) for parameter in encoder.parameters()}
    calls_before = len(encoder.calls)
    auxiliary_before = auxiliary.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    missing_grid = module.PaperDiffusionModel.encode_context(host(encoder), {**context, "map_condition_mask": torch.zeros(3, dtype=torch.bool)})["map_grid"]
    (missing_grid.sum() + auxiliary.square()).backward()
    check("allmissing_skips_encoder_grad_None", len(encoder.calls) == calls_before
          and all(parameter.grad is None for parameter in encoder.parameters()))
    optimizer.step()
    check("allmissing_BN_buffers_and_map_weights_unchanged_after_AdamW_decay", all(torch.equal(value, encoder.state_dict()[name]) for name, value in before.items()))
    check("allmissing_map_Adam_moments_steps_unchanged", all(
        torch.equal(value, optimizer.state[parameter][key]) if isinstance(value, torch.Tensor) else value == optimizer.state[parameter][key]
        for parameter, saved in map_optimizer_before.items() for key, value in saved.items()))
    check("actual_AdamW_step_exercised_for_other_parameters", not torch.equal(auxiliary_before, auxiliary.detach())
          and optimizer.defaults["weight_decay"] == .1)
    check("allmissing_grid_shape_dtype_from_fixed_interface", missing_grid.shape == (3, 4, 56, 56)
          and missing_grid.dtype == raster.dtype and (missing_grid == 0).all())
    check("no_CUDA_or_SDK", not torch.cuda.is_initialized()
          and not any(name.split(".")[0] in {"isaacsim", "isaaclab", "omni", "pxr"} for name in sys.modules))
    report = {"schema": "local_orca_interact.map_encoder_skip_CPU.v2", "status": "passed_required_small_Conv_BN_AdamW_cases",
              "checks": checks, "actual_cases": ["legacy_alltrue", "mixedavailable", "allmissing_after_initialized_AdamW"],
              "paper_source": record(HERE / "paper_interact.py"), "source_derivation": record(HERE / "source_derivation_v2.json"),
              "tiny_random_Conv_BN_and_AdamW_executed": True, "project_NN_model_constructed": False,
              "weights_or_models_loaded": False, "DDPM_executed": False, "GPU_used": False, "SDK_started": False,
              "full_conversion_executed": False, "training_pipeline_executed": False, "CPU_elapsed_s": time.perf_counter() - started}
    with (HERE / "actual_map_encoder_skip_CPU.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "checks": len(checks), "report": record(HERE / "actual_map_encoder_skip_CPU.json"),
                      "CPU_elapsed_s": report["CPU_elapsed_s"]}))


if __name__ == "__main__":
    main()
