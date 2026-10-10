await (async function supervisePopartTerminal() {
  const root = "/project/EXP/SangforAndSaut/ProtoMotions";
  const py = "/project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python";
  const evidence = root + "/docs/style_improvement_20261005/shared_social_evaluation_v1";
  const run = root + "/output/diffusion_amp/shared_social_popart_pilot2000_v1";
  const trainingSession = 16374;
  const ownerSource = evidence + "/popart_terminal_tool_owner.js";
  const statePath = evidence + "/popart_terminal_owner_state.json";
  const exitsPath = evidence + "/popart_pilot2000_observed_exits.json";
  const auditPath = evidence + "/popart_pilot2000_CPU_audit.json";
  const receiptPath = evidence + "/popart_pilot2000_execution.json";
  const postOutput = root + "/output/diffusion_amp/shared_social_popart_terminal2000_evaluation_v1";
  const q = s => "'" + String(s).replace(/'/g, "'\\''") + "'";
  const environment = "env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 ";
  const state = {schema:"shared_social.active_tool_exit_owner.v1", started_at_utc:new Date().toISOString(),
    training_session_id:trainingSession, run_directory:run, stage:"initializing_owner",
    training_exit_observed:false, paper_effect_achieved:false, default_model_promoted:false, stages:[]};
  const finish = async response => {
    while (typeof response.exit_code !== "number") {
      if (typeof response.session_id !== "number") throw new Error("Tool response has neither a session nor an actual exit");
      response = await tools.write_stdin({session_id:response.session_id,chars:"",yield_time_ms:30000,max_output_tokens:400});
    }
    return response;
  };
  const cpu = async cmd => finish(await tools.exec_command({cmd:environment+cmd,workdir:root,yield_time_ms:1000,max_output_tokens:500}));
  const writeJSON = async (path,value,exclusive=false,readonly=false) => {
    const code = "import json,sys;from pathlib import Path;p=Path(sys.argv[1]);v=json.loads(sys.argv[2]);"
      + "p.open('x' if sys.argv[3]=='1' else 'w').write(json.dumps(v,indent=2,allow_nan=False)+'\\n');"
      + "\np.chmod(0o444) if sys.argv[4]=='1' else None";
    const response = await cpu(q(py)+" -c "+q(code)+" "+q(path)+" "+q(JSON.stringify(value))
      +" "+q(exclusive?"1":"0")+" "+q(readonly?"1":"0"));
    if (response.exit_code !== 0) throw new Error("Owner metadata write failed");
  };
  const save = async () => {state.updated_at_utc=new Date().toISOString();await writeJSON(statePath,state);};
  const noteStage = async (name,response) => {
    state.stages.push({name,actual_exit_code:response.exit_code,terminal_tool_chunk:response.chunk_id,
      observed_at_utc:new Date().toISOString()});
    await save();
    if (response.exit_code !== 0) throw new Error(name+" exited "+response.exit_code+"; subsequent stages not started");
    notify({stage:name,actual_exit_code:response.exit_code,terminal_tool_chunk:response.chunk_id});
  };
  try {
    const source = "await (" + supervisePopartTerminal.toString() + ")();\n";
    const added = await tools.apply_patch("*** Begin Patch\n*** Add File: "+ownerSource+"\n"
      +source.split("\n").filter((v,i,a)=>i<a.length-1||v).map(line=>"+"+line).join("\n")+"\n*** End Patch");
    if (added && added.isError) throw new Error("Could not store immutable tool owner source");
    const prepCode = "import json,sys;from pathlib import Path;from scripts.run_shared_social_post_training import SOURCE_PATHS;"
      +"from CrowdSim.humanoid.runtime.social_evaluation_lifecycle import record;"
      +"p=Path(sys.argv[1]);p.chmod(0o444);"
      +"print(json.dumps({'owner_source':record(p),'guarded_sources':[record(s) for s in SOURCE_PATHS]"
      +"+[record(s) for s in ['scripts/train_shared_social.py','scripts/audit_shared_social_popart.py','scripts/seal_shared_social_training.py']]}))";
    const preparation = await cpu(q(py)+" -c "+q(prepCode)+" "+q(ownerSource));
    if (preparation.exit_code !== 0) throw new Error("Owner source binding failed");
    state.bindings=JSON.parse(preparation.output.trim());
    const guard = async () => {
      const code="import json,sys;from CrowdSim.humanoid.runtime.social_evaluation_lifecycle import record;"
        +"rows=json.loads(sys.argv[1]);assert all(record(r['path'])==r for r in rows),'Guarded experiment sources changed'";
      const response=await cpu(q(py)+" -c "+q(code)+" "+q(JSON.stringify(state.bindings.guarded_sources)));
      if(response.exit_code!==0) throw new Error("Guarded experiment source changed; outputs retained");
    };
    state.stage="waiting_for_actual_training_exit";
    await save();
    notify({owner:"active",training_session_id:trainingSession,state_file:statePath,
      automatic_sequence:"actual training exit -> CPU audit -> sealed receipt -> serial full22/style/navigation/CPU arrays",
      paper_effect_achieved:false});
    let training = await tools.write_stdin({session_id:trainingSession,chars:"",yield_time_ms:30000,max_output_tokens:400});
    let lastSaved=Date.now(),lastNotified=0;
    while(typeof training.exit_code!=="number") {
      if(typeof training.session_id!=="number") throw new Error("Original training session unavailable; no exit inferred from PID/results");
      if(Date.now()-lastSaved>60000) {
        const progressCode="import json;from pathlib import Path;from scripts.shared_social_training_status import read_progress;"
          +"r=read_progress(Path("+JSON.stringify(run)+"));print(json.dumps({k:r[k] for k in ['logged_updates','planned_updates','logged_physical_transitions','status']}))";
        const progress=await cpu(q(py)+" -c "+q(progressCode));
        if(progress.exit_code===0) {
          state.latest_log_progress=JSON.parse(progress.output.trim());
          if(state.latest_log_progress.logged_updates-lastNotified>=250) {
            lastNotified=state.latest_log_progress.logged_updates;
            notify({stage:"training_log_progress",...state.latest_log_progress,quality_inferred:false});
          }
        }
        await save();lastSaved=Date.now();
      }
      training=await tools.write_stdin({session_id:trainingSession,chars:"",yield_time_ms:30000,max_output_tokens:400});
    }
    state.training_exit_observed=true;
    state.training_terminal=training;
    await noteStage("training",training);
    await guard();
    state.stage="CPU_terminal_audit";await save();
    const audit=await cpu("CUDA_VISIBLE_DEVICES='' "+q(py)+" scripts/audit_shared_social_popart.py "+q(run)
      +" --output "+q(auditPath)+" > "+q(evidence+"/popart_pilot2000_CPU_audit.log")+" 2>&1");
    await noteStage("CPU_audit",audit);
    await writeJSON(exitsPath,{schema:"shared_social.observed_training_and_audit_exits.v1",run_directory:run,
      training_session_id:trainingSession,training,CPU_audit:audit,owner_source:state.bindings.owner_source,
      observed_at_utc:new Date().toISOString()},true,true);
    state.stage="sealing_completed_training";await save();
    await guard();
    const sealing=await cpu("CUDA_VISIBLE_DEVICES='' "+q(py)+" scripts/seal_shared_social_training.py"
      +" --run-directory "+q(run)+" --audit-report "+q(auditPath)+" --exit-metadata "+q(exitsPath)
      +" --output "+q(receiptPath)+" > "+q(evidence+"/popart_pilot2000_sealing.log")+" 2>&1");
    await noteStage("sealing",sealing);
    state.stage="independent_tracking_style_navigation";await save();
    await guard();
    const post=await finish(await tools.exec_command({cmd:environment+q(py)+" scripts/run_shared_social_post_training.py"
      +" --execution-receipt "+q(receiptPath)+" --registration "+q(evidence+"/popart_pilot2000_registration.json")
      +" --output "+q(postOutput)+" > "+q(evidence+"/popart_post_training_owner.log")+" 2>&1",
      workdir:root,sandbox_permissions:"require_escalated",
      justification:"用户已授权持续训练和独立评价。前置训练进程已实际退出0、终态CPU审计和封存通过，按预登记协议串行执行完整动作风格及四人Diffusion导航；沙箱没有CUDA，需要主机GPU。",
      yield_time_ms:1000,max_output_tokens:500}));
    await noteStage("complete_post_training_pipeline",post);
    await guard();
    const resultCode="import json,sys;from pathlib import Path;from CrowdSim.humanoid.runtime.social_evaluation_lifecycle import record;"
      +"p=Path(sys.argv[1]);d=json.loads(p.read_text());assert d['status']=='passed_execution';"
      +"print(json.dumps({'pipeline':record(p),'measurements':d['measurements']}))";
    const measured=await cpu(q(py)+" -c "+q(resultCode)+" "+q(postOutput+"/execution.json"));
    if(measured.exit_code!==0) throw new Error("Completed pipeline result admission failed");
    state.completed_measurements=JSON.parse(measured.output.trim());
    await writeJSON(evidence+"/popart_post_training_execution.json",{
      schema:"shared_social.actual_post_training_owner_exit.v1",actual_exit_code:post.exit_code,
      terminal_tool_chunk:post.chunk_id,owner_source:state.bindings.owner_source,
      ...state.completed_measurements,quality_or_author_equivalence_established:false,
      recorded_at_utc:new Date().toISOString()},true,true);
    state.stage="completed_measurements_quality_unclassified";await save();
    notify({owner:"completed",state_file:statePath,measurement_directory:postOutput,
      paper_effect_achieved:false,quality_requires_independent_comparison:true});
  } catch(error) {
    state.stage="failed_owner_or_stage";
    state.error=String(error);
    try {await save();} catch(writeError) {notify({owner_metadata_error:String(writeError)});}
    notify({owner:"failed",error:String(error),outputs_retained:true,exit_not_fabricated:true});
  }
})();
