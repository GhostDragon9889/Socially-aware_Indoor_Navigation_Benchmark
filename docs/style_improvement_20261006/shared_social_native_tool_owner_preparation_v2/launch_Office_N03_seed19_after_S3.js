// Run only in functions.exec. The actual cold S3 gate precedes every SDK/metadata step.
const nativeRoot = "/project/EXP/SangforAndSaut/ProtoMotions";
const nativePython = "/project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python";
const nativeRegistration = nativeRoot + "/docs/style_improvement_20261006/shared_social_native_tool_owner_preparation_v2/Office_N03_seed19_registration.json";
const nativeQuote = value => "'" + String(value).replace(/'/g, "'\\''") + "'";
const nativeLoadCode = "import json,sys;from pathlib import Path;from scripts import prepare_shared_social_native_owner as o;"
  + "p,r=o.admitted_plan(Path(sys.argv[1]));assert o.bridge.record(o.DRIVER)['sha256']=='d75bcde8965c243086b8b0d2a90f2df3c73d8be02454f233359ab9b354929aa8';"
  + "print(json.dumps({'envelope':{'registration':r,'plan':o.owner_view(p)},'source':o.DRIVER.read_text()}))";
const nativeLoaded = await tools.exec_command({cmd: "env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 "
  + nativeQuote(nativePython) + " -c " + nativeQuote(nativeLoadCode) + " " + nativeQuote(nativeRegistration),
  workdir: nativeRoot, yield_time_ms: 1000, max_output_tokens: 24000});
if (nativeLoaded.exit_code !== 0) throw new Error("Cold native owner loading did not actually exit0; no SDK started");
const nativePayload = JSON.parse(nativeLoaded.output);
const nativeRun = new Function(nativePayload.source + "\nreturn runRegisteredNativeToolOwner;")();
const nativeCompleted = await nativeRun(tools, nativePayload.envelope, notify);
text({case_id: nativeCompleted.case_id, stage: nativeCompleted.stage, paper_effect_achieved: false,
  external_owner_cell_completion_still_requires_root_observation: true});
