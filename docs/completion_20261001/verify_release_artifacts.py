#!/usr/bin/env python3
"""Verify local wheel/sdist contracts and run the standalone wheel from /tmp.

Uses an offline, dependency-free uv --target installation. Existing NumPy/Torch
come from the explicitly selected interpreter; its environment is not modified.
"""

from __future__ import annotations

import argparse
import configparser
from email.parser import Parser
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tempfile
import zipfile


REPO = Path(__file__).resolve().parents[2]
REQUIRED_SDIST = (
    "MANIFEST.in", "FULL_PIPELINE.md", "CrowdSim/humanoid/simulate.py",
    "CrowdSim/config/env.yaml", "CrowdSim/ppo/train_ppo.py",
    "src/pyproject.toml", "src/README.md", "src/LICENSE",
    "src/crowdsim_ppo/train.py", "src/crowdsim_ppo/evaluate.py",
    "src/crowdsim_ppo/configs/point_goal.json", "configs/point_goal.json",
    "scripts/isaaclab3_environment.py", "scripts/audit_repository.py",
    "scripts/check_isaaclab_smpl_fk.py", "scripts/check_motion_asset_consistency.py",
    "env/isaaclab3_sim6/pyproject.toml", "env/isaaclab3_sim6/uv.lock",
    "env/isaaclab3_sim6/profile.json", "env/isaaclab3_sim6/README.md",
    "docs/ISAAC_SIM6_LAB3_MIGRATION_ZH.md", "data/smpl/install_local_smpl.py",
    "protomotions/tests/test_checkpoint_bundle.py",
    "protomotions/tests/test_recording_lifecycle.py",
)
SOURCE_SUFFIXES = {".py", ".json", ".yaml", ".yml", ".toml", ".lock", ".md", ".txt", ".rst"}
WEIGHT_SUFFIXES = {".ckpt", ".pt", ".pth", ".safetensors", ".onnx", ".pkl", ".pickle"}


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def entries(path: Path) -> dict[str, bytes]:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            return {name: archive.read(name) for name in archive.namelist() if not name.endswith("/")}
    with tarfile.open(path, "r:gz") as archive:
        return {member.name.split("/", 1)[1]: archive.extractfile(member).read()
                for member in archive.getmembers() if member.isfile() and "/" in member.name}


def licensed_smpl_asset(name: str) -> bool:
    name = name.lower()
    return (name.startswith("protomotions/data/assets/mesh/smpl/")
            or name.startswith("protomotions/data/assets/mjcf/smpl")
            or name.startswith("protomotions/data/assets/usd/smpl")
            or (name.startswith("data/smpl/") and Path(name).suffix in {".pkl", ".npz"}))


def metadata(content: dict[str, bytes]):
    names = [name for name in content if name.endswith(".dist-info/METADATA")]
    if len(names) != 1:
        raise ValueError(f"Expected one wheel METADATA, got {names}")
    return Parser().parsestr(content[names[0]].decode())


def console_entries(content: dict[str, bytes]) -> dict[str, str]:
    names = [name for name in content if name.endswith(".dist-info/entry_points.txt")]
    if len(names) != 1:
        raise ValueError("Wheel has no unique entry_points.txt")
    config = configparser.ConfigParser()
    config.read_string(content[names[0]].decode())
    return dict(config["console_scripts"])


def one(directory: Path, pattern: str) -> Path:
    candidates = list(directory.glob(pattern))
    if len(candidates) != 1:
        raise ValueError(f"Expected one {pattern} under {directory}, got {candidates}")
    return candidates[0].resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("/tmp/protomotions-completion-packaging"))
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--uv", default="uv")
    parser.add_argument("--output-dir", type=Path, default=REPO / "docs/completion_20261001")
    parser.add_argument("--prefix", default="release_artifacts_final")
    parser.add_argument("--preliminary", action="store_true", help="Report absent/newer source payloads as rebuild warnings")
    parser.add_argument("--skip-runtime", action="store_true", help="Check archives only; never marks the runtime as passed")
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Preserve the venv invocation path: resolving its symlink would select
    # uv's underlying interpreter without this environment's Torch/NumPy.
    args.python = args.python.absolute()
    checks, artifacts = [], []
    runtime = {"status": "not_run"}

    def check(label, condition, detail=None, *, freshness=False):
        checks.append({"check": label, "status": "passed" if condition else "rebuild_required" if freshness and args.preliminary else "failed", "detail": detail})

    def run(label, command, *, cwd, env=None, timeout=45):
        print(f"{label}...", flush=True)
        result = subprocess.run([str(item) for item in command], cwd=cwd, env=env,
                                capture_output=True, text=True, timeout=timeout)
        (args.output_dir / f"{args.prefix}_{label}.log").write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(f"{label} exited {result.returncode}; see {args.prefix}_{label}.log")
        return result.stdout

    try:
        paths = {
            "core_wheel": one(args.artifacts, "protomotions-*.whl"),
            "core_sdist": one(args.artifacts, "protomotions-*.tar.gz"),
            "crowd_wheel": one(args.artifacts, "crowdsim_ppo-*.whl"),
            "crowd_sdist": one(args.artifacts, "crowdsim_ppo-*.tar.gz"),
        }
        contents = {}
        for kind, path in paths.items():
            content = entries(path)
            contents[kind] = content
            artifacts.append({"kind": kind, "path": str(path), "bytes": path.stat().st_size,
                              "sha256": digest(path.read_bytes()), "file_count": len(content)})
            forbidden = [name for name in content if licensed_smpl_asset(name) or Path(name).suffix.lower() in WEIGHT_SUFFIXES]
            check(f"{kind}: no licensed SMPL asset or external weight file", not forbidden, forbidden)
            check(f"{kind}: safe archive member names", not any(Path(name).is_absolute() or ".." in Path(name).parts for name in content))
            stale = []
            for name, value in content.items():
                local = REPO / ("src" if kind.startswith("crowd") else "") / name
                if Path(name).suffix in SOURCE_SUFFIXES and local.is_file() and digest(local.read_bytes()) != digest(value):
                    stale.append(name)
            check(f"{kind}: shipped source matches frozen tree", not stale, stale, freshness=True)

        for kind, package in (("core_wheel", "protomotions"), ("crowd_wheel", "crowdsim_ppo")):
            content = contents[kind]
            outsiders = [name for name in content if not name.startswith(package + "/") and ".dist-info/" not in name]
            check(f"{kind}: isolated package namespace", not outsiders, outsiders)
        check("core_wheel: checkpoint helper delivered", "protomotions/utils/checkpoint.py" in contents["core_wheel"])
        core_meta = metadata(contents["core_wheel"])
        check("core_wheel: correct distribution", core_meta["Name"] == "protomotions")

        required = list(REQUIRED_SDIST)
        # The parent writes this report after the first archive build. Its
        # absence before creation is not a final defect; once created it ships.
        if (REPO / "docs/PROJECT_COMPLETION_20261001_ZH.md").is_file():
            required.append("docs/PROJECT_COMPLETION_20261001_ZH.md")
        missing = [name for name in required if name not in contents["core_sdist"]]
        check("core_sdist: integration/profile/check scripts delivered", not missing, missing, freshness=True)
        check("sdists: generated build trees omitted", not any("build" in Path(name).parts for kind in ("core_sdist", "crowd_sdist") for name in contents[kind]))

        expected_entries = {"crowdsim-ppo-train": "crowdsim_ppo.train:main", "crowdsim-ppo-eval": "crowdsim_ppo.evaluate:main"}
        crowd_meta = metadata(contents["crowd_wheel"])
        dependencies = crowd_meta.get_all("Requires-Dist", [])
        dependency_names = {re.match(r"[A-Za-z0-9_.-]+", value).group().lower().replace("_", "-") for value in dependencies}
        check("crowd_wheel: no ProtoMotions/Isaac hard dependency", dependency_names <= {"numpy", "torch", "pytest"}, dependencies)
        check("crowd_wheel: correct console entries", console_entries(contents["crowd_wheel"]) == expected_entries, console_entries(contents["crowd_wheel"]))
        defaults = {}
        for kind in ("crowd_wheel", "crowd_sdist"):
            name = "crowdsim_ppo/configs/point_goal.json"
            defaults[kind] = json.loads(contents[kind][name])
            check(f"{kind}: bundled default JSON delivered", isinstance(defaults[kind], dict))
        check("crowd artifacts: default JSON agrees", defaults["crowd_wheel"] == defaults["crowd_sdist"])

        if not args.skip_runtime:
            work = Path(tempfile.mkdtemp(prefix="protomotions-release-verification-", dir="/tmp"))
            target = work / "installed"
            cwd = work / "arbitrary-working-directory"
            cwd.mkdir()
            runtime = {"status": "running", "temporary_directory": str(work), "cwd": str(cwd)}
            run("install_target", [args.uv, "pip", "install", "--offline", "--no-index", "--no-deps", "--python", args.python,
                                   "--target", target, "--cache-dir", work / "uv-cache", paths["crowd_wheel"]], cwd=cwd)
            env = dict(os.environ)
            env.update(PYTHONPATH=str(target), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
            probe = r'''
import hashlib, importlib.abc, json, pathlib, sys
class RejectSimulator(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'protomotions','CrowdSim','isaaclab','isaacsim'}:
            raise ModuleNotFoundError('Standalone wheel attempted simulator import: '+fullname)
sys.meta_path.insert(0,RejectSimulator())
import crowdsim_ppo
from crowdsim_ppo.pipeline import default_config_path, load_config
target=pathlib.Path(sys.argv[1]).resolve()
package=pathlib.Path(crowdsim_ppo.__file__).resolve()
default=default_config_path().resolve()
assert package.is_relative_to(target)
assert default==target/'crowdsim_ppo/configs/point_goal.json'
values=load_config(None)
print(json.dumps({'package':str(package),'default_config':str(default),'default_sha256':hashlib.sha256(default.read_bytes()).hexdigest(),'num_envs':values['environment']['num_envs'],'rollout_steps':values['training']['rollout_steps'],'simulator_import_guard':'passed'}))
'''
            default_result = json.loads(run("default_probe", [args.python, "-c", probe, target], cwd=cwd, env=env).strip())
            wrappers = {name: target / "bin" / name for name in expected_entries}
            if not all(path.is_file() for path in wrappers.values()):
                raise RuntimeError("uv --target did not deliver both console wrappers")
            output = work / "trained"
            checkpoint = output / "latest.pt"
            inspect_checkpoint = r'''
import hashlib,json,torch,sys
state=torch.load(sys.argv[1],map_location='cpu',weights_only=False)
def tensors(value):
    if isinstance(value,torch.Tensor): yield value
    elif isinstance(value,dict):
        for item in value.values(): yield from tensors(item)
    elif isinstance(value,(list,tuple)):
        for item in value: yield from tensors(item)
all_tensors=list(tensors(state))
assert all(torch.isfinite(value).all() for value in all_tensors)
model=state['model']; sha=hashlib.sha256()
for key,value in sorted(model.items()):
    sha.update(key.encode()); sha.update(value.detach().cpu().contiguous().numpy().tobytes())
optimizer_steps=[float(value['step']) for value in state['optimizer']['state'].values() if 'step' in value]
assert optimizer_steps
print(json.dumps({'step':state['step'],'model_sha256':sha.hexdigest(),'model_parameters':sum(v.numel() for v in model.values()),'finite_tensor_count':len(all_tensors),'optimizer_step_min':min(optimizer_steps),'optimizer_step_max':max(optimizer_steps),'curriculum_present':'curriculum' in state.get('extra_state',{})}))
'''
            run("train", [args.python, wrappers["crowdsim-ppo-train"], "--device", "cpu", "--total-steps", "1", "--output", output], cwd=cwd, env=env)
            first = json.loads(run("checkpoint_initial", [args.python, "-c", inspect_checkpoint, checkpoint], cwd=cwd, env=env).strip())
            rollout_size = default_result["num_envs"] * default_result["rollout_steps"]
            if first["step"] != rollout_size:
                raise RuntimeError(f"Expected one default rollout ({rollout_size}), got {first['step']}")
            run("resume", [args.python, wrappers["crowdsim-ppo-train"], "--device", "cpu", "--total-steps", str(first["step"] + 1), "--resume", checkpoint, "--output", output], cwd=cwd, env=env)
            resumed = json.loads(run("checkpoint_resumed", [args.python, "-c", inspect_checkpoint, checkpoint], cwd=cwd, env=env).strip())
            if resumed["step"] != first["step"] + rollout_size or resumed["model_sha256"] == first["model_sha256"]:
                raise RuntimeError("Resume failed to advance optimizer/model state by one default rollout")
            if not first["curriculum_present"] or not resumed["curriculum_present"] or resumed["optimizer_step_max"] <= first["optimizer_step_max"]:
                raise RuntimeError("Resume did not preserve curriculum and advance saved optimizer steps")
            evaluation = json.loads(run("evaluate", [args.python, wrappers["crowdsim-ppo-eval"], "--device", "cpu", "--checkpoint", checkpoint, "--episodes", "2", "--num-envs", "2"], cwd=cwd, env=env).strip())
            if evaluation["episodes"] != 2 or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in evaluation.values()):
                raise RuntimeError("Evaluation did not return exactly two episode metrics")
            runtime = {"status": "passed", "python": str(args.python), "temporary_directory": str(work), "cwd": str(cwd),
                       "install": "uv pip install --offline --no-index --no-deps --target TEMP", "default": default_result,
                       "initial": first, "resumed": resumed, "evaluation": evaluation,
                       "existing_environment_modified": False, "scope": "Wheel code isolated in /tmp; existing NumPy/Torch dependencies reused; no Kit, no GPU training"}
    except Exception as error:
        if runtime["status"] == "running":
            runtime["status"] = "failed"
        checks.append({"check": "verification execution", "status": "failed", "detail": f"{type(error).__name__}: {error}"})

    failed = any(check["status"] == "failed" for check in checks)
    rebuild = any(check["status"] == "rebuild_required" for check in checks)
    status = ("failed" if failed else "preliminary_pass_requires_rebuild" if rebuild
              else "archives_passed_runtime_not_run" if args.skip_runtime else "passed")
    payload = {"status": status, "mode": "preliminary" if args.preliminary else "final", "artifacts": artifacts,
               "checks": checks, "runtime": runtime, "root_report_creation_deferred": not (REPO / "docs/PROJECT_COMPLETION_20261001_ZH.md").exists(),
               "limitations": ["Reuses existing NumPy/Torch; this is not a fresh dependency-resolution or Python-version matrix test.",
                               "No Kit, physical humanoid/robot acceptance, production training quality, upload or publication.",
                               "Use final mode after frozen sources and the parent report have been rebuilt into all four artifacts."]}
    (args.output_dir / f"{args.prefix}.json").write_text(json.dumps(payload, indent=2) + "\n")
    body = [f"# 发行工件验证：{status}", "", "仅离线安装到 /tmp target；现有环境未安装/升级包。", "",
            "| 工件 | 大小（bytes） | SHA256 |", "| --- | ---: | --- |"]
    body += [f"| {Path(item['path']).name} | {item['bytes']} | `{item['sha256']}` |" for item in artifacts]
    body += ["", "## 检查", ""] + [f"- {item['status']}: {item['check']}" + (f" — `{item['detail']}`" if item.get('detail') and item['status'] != 'passed' else "") for item in checks]
    body += ["", "## 临时 wheel 实际执行", "", "```json", json.dumps(runtime, indent=2, ensure_ascii=False), "```", "",
             (f"短训练使用 wheel 自带默认 JSON，按完整 rollout 提交步数；请求 1 步实际为 {runtime['initial']['step']} 步，恢复后为 {runtime['resumed']['step']} 步。此处验证安装/入口/保存恢复链，指标不代表策略质量。"
              if runtime["status"] == "passed" else "运行未完成或已跳过；没有将训练、恢复、评估记为通过。"), "",
             "## 再次运行", "", "```bash", f"{args.python} {Path(__file__).resolve()} --artifacts {args.artifacts.resolve()} --prefix release_artifacts_final", "```", "",
             "初验使用 `--preliminary` 将尚未重建的源文件/新脚本列为待重建；最终模式要求发行代码与冻结源码匹配。父任务主报告创建之前，不把它的缺失算作最终缺陷。"]
    (args.output_dir / f"{args.prefix}.md").write_text("\n".join(body) + "\n")
    print(json.dumps({"status": status, "runtime": runtime["status"], "report": str(args.output_dir / f'{args.prefix}.json')}, indent=2))
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
