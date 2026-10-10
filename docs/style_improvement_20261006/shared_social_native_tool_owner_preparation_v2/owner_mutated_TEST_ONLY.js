// Executed in functions.exec with the actual tools object. No Node/filesystem APIs.
// CPU tests pass explicitly labelled in-memory tools; they are never SDK receipts.
async function observeNativeTerminal(tools, request, onResponse = async () => {}) {
  let response = await tools.exec_command(request);
  let session;
  while (true) {
    if (!response || typeof response !== "object") throw new Error("Invalid tools API response");
    await onResponse(response);
    if (response.exit_code !== undefined && response.exit_code !== null) {
      if (!Number.isInteger(response.exit_code) || typeof response.chunk_id !== "string" || !response.chunk_id) {
        throw new Error("Require actual integer tool exit and terminal chunk");
      }
      return response;
    }
    if (!Number.isInteger(response.session_id) || response.session_id <= 0 ||
        (session !== undefined && response.session_id !== session)) {
      throw new Error("Original tool session missing or replaced; no restart or inferred exit");
    }
    session = response.session_id;
    // A pending response is not terminal. Keep observing precisely this handle.
    let errors = 0;
    while (true) {
      try {
        response = await tools.write_stdin({session_id: session, chars: "", yield_time_ms: 30000, max_output_tokens: 1200});
        break;
      } catch (error) {
        // Observation errors say nothing about child termination. Re-poll the same handle.
        await onResponse({session_id: session, observation_error: String(error)});
        if (++errors >= 3) {
          const interrupted = new Error("Original session observation unavailable; process outcome unknown; no restart");
          interrupted.observationInterrupted = true;
          throw interrupted;
        }
      }
    }
  }
}

async function runRegisteredNativeToolOwner(tools, envelope, notify = () => {}) {
  const {plan, registration} = envelope;
  if (!plan || plan.schema !== "shared_social.registered_native_serial_tool_plan.v1" ||
      plan.registration_path !== registration.path || plan.SDK_started !== false ||
      !Array.isArray(plan.actions) || plan.actions.length !== 21) {
    throw new Error("Require the exact prepared native case envelope");
  }
  const root = "/project/EXP/SangforAndSaut/ProtoMotions";
  const python = "/project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python";
  const quote = value => "'" + String(value).replace(/'/g, "'\\''") + "'";
  const CPU = "env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 ";
  const cpu = cmd => observeNativeTerminal(tools, {cmd, workdir: root, yield_time_ms: 1000, max_output_tokens: 16000});
  const requireZero = (response, role) => {
    if (response.exit_code !== 0) throw new Error(role + " actual exit " + response.exit_code + "; subsequent stages stopped");
  };
  // No metadata directories or SDK children until the entire S3 owner is admitted.
  const preflight = await cpu(plan.execution_preflight_command);
  requireZero(preflight, "S3 execution preflight");
  const ready = JSON.parse(preflight.output.trim());
  if (ready.status !== "native_serial_S3_execution_ready" || JSON.stringify(ready.registration) !== JSON.stringify(registration) ||
      JSON.stringify(ready.owner_plan) === JSON.stringify(plan)) {
    throw new Error("Execution readiness belongs to another registration");
  }
  const state = {schema: "shared_social.actual_native_serial_tools_owner_state.v1", registration,
    case_id: plan.case_id, external_owner_completion: ready.external_owner_completion,
    stage: "starting_serial_case", observed_stages: [], paper_effect_achieved: false, default_model_promoted: false};
  const write = async (path, value, exclusive, readonly) => {
    const code = "import json,sys;from pathlib import Path;p=Path(sys.argv[1]);root=Path(sys.argv[5]);"
      + "assert p.is_relative_to(root) and not any(q.is_symlink() for q in (p,*p.parents));"
      + "p.parent.mkdir(parents=True,exist_ok=True);"
      + "p.open('x' if sys.argv[3]=='1' else 'w').write(json.dumps(json.loads(sys.argv[2]),indent=2,allow_nan=False)+'\\n');"
      + "p.chmod(0o444) if sys.argv[4]=='1' else None";
    const response = await cpu(CPU + quote(python) + " -c " + quote(code) + " " + quote(path) + " "
      + quote(JSON.stringify(value)) + " " + quote(exclusive ? "1" : "0") + " " + quote(readonly ? "1" : "0")
      + " " + quote(plan.observation_directory));
    requireZero(response, "Actual tool observation metadata write");
  };
  const save = async () => {
    state.updated_at_utc = new Date().toISOString();
    await write(plan.state_path, state, false, false);
  };
  const guard = async () => {
    const response = await cpu(plan.guard_command);
    requireZero(response, "Native source/S3 guard");
    const bound = JSON.parse(response.output.trim());
    if (bound.status !== "native_serial_sources_and_S3_guarded" || JSON.stringify(bound.registration) !== JSON.stringify(registration)) {
      throw new Error("Native guarded registration differs");
    }
  };
  const execute = async action => {
    await guard();
    state.stage = action.role;
    await save();
    const request = {cmd: action.cmd, workdir: action.workdir, yield_time_ms: 1000, max_output_tokens: 1200};
    if (action.requires_host_GPU === true) {
      request.sandbox_permissions = "require_escalated";
      request.justification = "用户已授权行人模拟与独立评估；S3及其后续GPU评价已完成并准入，按固定案例串行执行原生SDK。";
    }
    let returnedChunks = 0;
    const response = await observeNativeTerminal(tools, request, async observed => {
      returnedChunks++;
      state.current_tool_observation = {role: action.role, cmd: request.cmd,
        session_id: observed.session_id, exit_code: observed.exit_code, chunk_id: observed.chunk_id,
        observation_error: observed.observation_error};
      await save();
      notify({case_id: plan.case_id, stage: action.role, observed_chunk: observed.chunk_id,
        actual_exit_code: observed.exit_code, session_id: observed.session_id});
    });
    await write(action.observation, {schema: "shared_social.external_native_tool_observation.v1",
      cmd: request.cmd, workdir: request.workdir, actual_terminal_response_observed: true,
      response, returned_tool_chunks: returnedChunks,
      captured_stdout_scope: "Actual terminal API response; earlier stdout may be tool-truncated",
      observed_at_utc: new Date().toISOString()}, true, true);
    state.observed_stages.push({role: action.role, actual_exit_code: response.exit_code,
      terminal_tool_chunk: response.chunk_id, observation_path: action.observation});
    await save();
    return response;
  };
  try {
    await save();
    for (const action of plan.actions) {
      const response = await execute(action);
      if (response.exit_code !== 0 && action.failure_publication) {
        // Publish the actual failed SDK/raw exit before stopping. Never relabel it successful.
        const failedPublication = await execute(action.failure_publication);
        state.failed_exit_publication_exit_code = failedPublication.exit_code;
        await save();
      }
      requireZero(response, action.role);
    }
    await guard();
    state.stage = "completed_measurements_quality_unclassified";
    await save();
    notify({case_id: plan.case_id, stage: state.stage, paper_effect_achieved: false});
    return state;
  } catch (error) {
    const current = state.current_tool_observation;
    state.stage = error.observationInterrupted || (current && Number.isInteger(current.session_id) &&
      !Number.isInteger(current.exit_code)) ? "observation_interrupted_process_outcome_unknown" : "failed_owner_or_stage";
    state.error = String(error);
    try { await save(); } catch (writeError) { notify({metadata_write_error: String(writeError)}); }
    notify({case_id: plan.case_id, stage: state.stage, error: String(error), outputs_retained: true});
    throw error;
  }
}
