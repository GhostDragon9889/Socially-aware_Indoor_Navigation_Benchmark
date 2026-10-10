"""CPU independent public-statistic/RNG checks; not a trainer integration gate."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from CrowdSim.humanoid.training.trainer import RunningNormalizer  # noqa: E402

CANDIDATE = ROOT/"docs/style_improvement_20261002/staged_terrain_rms_candidate_v1/statistics_policy.py"
SNAPSHOT = CANDIDATE.parent/"public_task_statistics_snapshot.npz"
PUBLIC = ROOT/"data/pretrained_models/paper_references/pacer/output/release/pacer_no_shape/Humanoid.pth"
CHECKS = []


def require(value, message):
    if not value:
        raise ValueError(message)
    CHECKS.append(message)


def record(path):
    path=Path(path).resolve()
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b""):
            digest.update(block)
    return {"path":str(path),"bytes":path.stat().st_size,"sha256":digest.hexdigest()}


def rejection(call, message):
    try:
        call()
    except ValueError:
        CHECKS.append(message)
    else:
        raise ValueError("Expected refusal: "+message)


def main():
    torch.set_num_threads(4)
    require(not torch.cuda.is_initialized(),"CUDA not initialized")
    candidate=record(CANDIDATE)
    spec=importlib.util.spec_from_file_location("RMS_independent_prototype",CANDIDATE)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    public_record=record(PUBLIC)
    require(public_record["sha256"]=="bc0cb89b4d7ceaeb198339c34f3d736eb7449e7e479a7ee0f80d32563737d615","actual public checkpoint full SHA recomputed")
    public=torch.load(PUBLIC,map_location="cpu",weights_only=True,mmap=True)["running_mean_std"]
    snapshot=record(SNAPSHOT)
    with np.load(SNAPSHOT,allow_pickle=False) as archive:
        mean,variance,count=archive["task_mean"],archive["task_variance"],archive["reference_count"]
    require(torch.equal(torch.from_numpy(mean),public["running_mean"][357:1401]),"full snapshot mean equals actual public task coordinates")
    require(torch.equal(torch.from_numpy(variance),public["running_var"][357:1401]),"full snapshot variance equals actual public task coordinates")
    require(count.item()==public["count"].item()==61931520001.,"immutable original reference count independent of global warm cap")
    normalizer=RunningNormalizer(1044)
    normalizer.mean.copy_(torch.from_numpy(mean))
    normalizer.variance.copy_(torch.from_numpy(variance))
    normalizer.count.fill_(200000.)
    old=deepcopy(normalizer)
    python_before=random.getstate()
    numpy_before=np.random.get_state()
    torch_before=torch.get_rng_state().clone()
    policy=module.StagedTerrainRMSPolicy("preserve_declared_suffix",reference_record=snapshot,fresh_experiment=True)
    policy.attach(normalizer)
    # Synthetic raw task inputs test statistic arithmetic only; no physics claim.
    values=torch.zeros(33,1044,dtype=torch.float32)
    values[:,:20]=torch.arange(33,dtype=torch.float32)[:,None]/30
    for _ in range(2):
        normalizer.update(values)
        old.update(values)
        prefix=(normalizer.mean[:20].clone(),normalizer.variance[:20].clone(),normalizer.count.clone())
        policy.after_update(normalizer)
        require(torch.equal(normalizer.mean[:20],prefix[0]) and torch.equal(normalizer.variance[:20],prefix[1]) and torch.equal(normalizer.count,prefix[2]),"hook does not overwrite trajectory prefix20/global count")
        require(torch.equal(normalizer.mean[20:],torch.from_numpy(mean[20:])) and torch.equal(normalizer.variance[20:],torch.from_numpy(variance[20:])),"terrain1024 retains exact public moments")
    require(normalizer.count.item()==old.count.item()==200066. and policy.reference_count==61931520001.,"global real-batch count and immutable reference count separate")
    saved={"normalizer":deepcopy(normalizer.state_dict()),"policy":policy.state_dict()}
    resumed=RunningNormalizer(1044)
    resumed.load_state_dict(saved["normalizer"],strict=True)
    resumed_policy=module.StagedTerrainRMSPolicy("preserve_declared_suffix",reference_record=snapshot,fresh_experiment=True)
    resumed_policy.load_state_dict(saved["policy"],resumed)
    for norm,hook in [(normalizer,policy),(resumed,resumed_policy)]:
        norm.update(values)
        hook.after_update(norm)
    require(all(torch.equal(v,resumed.state_dict()[k]) for k,v in normalizer.state_dict().items()),"next moment update after restored state equals uninterrupted arithmetic")
    rejection(lambda:resumed_policy.load_state_dict(None,resumed),"protected missing resume state rejected")
    changed=deepcopy(saved["policy"])
    changed["suffix_mean"][7]+=.01
    rejection(lambda:resumed_policy.load_state_dict(changed,resumed),"protected changed immutable suffix rejected")
    rejection(lambda:module.StagedTerrainRMSPolicy().load_state_dict(saved["policy"],old),"legacy cannot silently restore protected state")
    changed=deepcopy(saved["policy"])
    changed["calls"]=99
    old_saved=RunningNormalizer(1044)
    old_saved.load_state_dict(saved["normalizer"],strict=True)
    module.StagedTerrainRMSPolicy("preserve_declared_suffix",reference_record=snapshot,fresh_experiment=True).load_state_dict(changed,old_saved)
    require(random.getstate()==python_before,"Python caller RNG unchanged by prototype")
    numpy_after=np.random.get_state()
    require(numpy_before[0]==numpy_after[0] and np.array_equal(numpy_before[1],numpy_after[1]) and numpy_before[2:]==numpy_after[2:],"NumPy caller RNG unchanged by prototype")
    require(torch.equal(torch_before,torch.get_rng_state()) and not torch.cuda.is_initialized(),"Torch CPU caller RNG unchanged, CUDA not touched")
    require(record(CANDIDATE)==candidate and record(SNAPSHOT)==snapshot,"candidate and reference bytes not modified")
    report={"schema":"terrain_RMS_independent_prototype_review.v1","status":"prototype_arithmetic_passed_integration_pending",
            "CPU_only":True,"SDK_or_GPU_started":False,"production_or_candidate_modified":False,"checks_passed":len(CHECKS),"checks":CHECKS,
            "candidate":candidate,"public_checkpoint":public_record,"snapshot":snapshot,
            "reference_count":61931520001.,"initial_global_task_count":200000.,"global_task_count_after_three_updates":200099.,
            "raw_observations":"synthetic33 raw task vectors, trajectory prefix varies, terrain suffix zero; no real rollout claim",
            "prototype_limit_calls99_accepted_for_actual2_updates":True,
            "integration_required_gates":["calls==trainer.update_count, including strict missing/extra/tampered state handling",
                "GPU map_location moves saved suffix tensors: explicit validated CPU transport before the prototype's CPU-only state check",
                "Protected second warmstart refused before any model/statistics copies; attach failure must not leave a partially overwritten trainer",
                "Actual warmstart controller source/moments binds protection without an external deployment NPZ dependency",
                "Inference validates internal bundle protection state/moments/count/source, no attach/overwrite or optimizer/replay",
                "Legacy missing new config/state accepted only as old legacy and output/RNG semantics stay exact"],
            "production_integration_reviewed":False,"real4plus4_GPU_resume_verified":False,"navigation_improvement_claim":False}
    target=Path(__file__).resolve().parent/"prototype_review.json"
    with target.open("x") as stream:
        json.dump(report,stream,indent=2,allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status":report["status"],"checks_passed":report["checks_passed"],"reference_count":report["reference_count"]}))


if __name__=="__main__":
    main()
