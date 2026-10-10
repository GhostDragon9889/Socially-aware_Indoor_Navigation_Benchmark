// In-memory tools fixtures only; no SDK, files, receipts or root completion certificate.
async function nativeToolOwnerCPUContracts(runOwner, observe) {
  let passed = 0;
  const require = (value, message) => { if (!value) throw new Error(message); };
  const rejects = async (operation, message) => {
    let error;
    try { await operation(); } catch (caught) { error = caught; }
    require(error && String(error).includes(message), "Expected refusal: " + message + "; got " + error);
  };
  const test = async (name, operation) => { await operation(); passed++; };
  const terminal = code => ({exit_code: code, chunk_id: "CPU_FIXTURE_NOT_ACTUAL_TERMINAL", output: ""});
  await test("pending session is observed until integer exit", async () => {
    let starts = 0, polls = 0;
    const tools = {exec_command: async () => { starts++; return {session_id: 731}; },
      write_stdin: async request => { require(request.session_id === 731, "Wrong original session");
        return ++polls === 1 ? {session_id: 731} : terminal(-9); }};
    const result = await observe(tools, {cmd: "CPU_FIXTURE"});
    require(result.exit_code === -9 && starts === 1 && polls === 2, "Exit inferred or child restarted");
  });
  for (const code of [true, 0.5, "0"]) {
    await test("non-integer exit refused", () => rejects(() => observe({exec_command: async () => terminal(code)}, {}), "integer tool exit"));
  }
  await test("missing terminal chunk refused", () => rejects(() => observe({exec_command: async () => ({exit_code: 0})}, {}), "terminal chunk"));
  await test("replacement session refused", () => rejects(() => observe({exec_command: async () => ({session_id: 731}),
    write_stdin: async () => ({session_id: 732})}, {}), "session missing or replaced"));
  await test("transient observation failure re-polls same handle", async () => {
    let polls = 0;
    const response = await observe({exec_command: async () => ({session_id: 731}), write_stdin: async request => {
      require(request.session_id === 731, "Wrong handle after timeout");
      if (++polls === 1) throw new Error("CPU_FIXTURE_TRANSIENT_TIMEOUT");
      return terminal(0);
    }}, {});
    require(response.exit_code === 0 && polls === 2, "Transient observation was treated as terminal");
  });
  await test("repeated observation failures never fabricate exit or restart", async () => {
    let starts = 0, polls = 0;
    await rejects(() => observe({exec_command: async () => { starts++; return {session_id: 731}; },
      write_stdin: async () => { polls++; throw new Error("CPU_FIXTURE_TIMEOUT"); }}, {}), "process outcome unknown");
    require(starts === 1 && polls === 3, "Unexpected restart or failure polling");
  });
  const fixture = () => {
    const registration = {path: "/tmp/CPU_FIXTURE_NOT_ACTUAL_registration.json", bytes: 1, sha256: "CPU_FIXTURE"};
    const actions = Array.from({length: 21}, (_, index) => ({role: "CPU_FIXTURE_stage" + index,
      cmd: "CPU_FIXTURE_command" + index, workdir: "/tmp", observation: "/tmp/CPU_FIXTURE_NOT_ACTUAL_" + index,
      requires_host_GPU: index === 0}));
    actions[0].failure_publication = actions[1];
    return {registration, plan: {schema: "shared_social.registered_native_serial_tool_plan.v1", SDK_started: false,
      registration_path: registration.path, actions, case_id: "CPU_FIXTURE_NOT_ACTUAL_CASE",
      execution_preflight_command: "CPU_FIXTURE_preflight", guard_command: "CPU_FIXTURE_guard",
      observation_directory: "/tmp/CPU_FIXTURE_NOT_ACTUAL", state_path: "/tmp/CPU_FIXTURE_NOT_ACTUAL_state.json"}};
  };
  const mock = (envelope, failRole, preflight = 0) => {
    const children = [], writes = [];
    const tools = {exec_command: async request => {
      if (request.cmd === "CPU_FIXTURE_preflight") return {...terminal(preflight), output: JSON.stringify({
        status: "native_serial_S3_execution_ready", registration: envelope.registration,
        external_owner_completion: {CPU_FIXTURE_NOT_ACTUAL: true}})};
      if (request.cmd === "CPU_FIXTURE_guard") return {...terminal(0), output: JSON.stringify({
        status: "native_serial_sources_and_S3_guarded", registration: envelope.registration})};
      const row = envelope.plan.actions.find(action => action.cmd === request.cmd);
      if (row) {
        require((request.sandbox_permissions === "require_escalated") === row.requires_host_GPU, "Host GPU scope differs");
        children.push(row.role);
        return terminal(row.role === failRole ? -9 : 0);
      }
      writes.push(request.cmd);
      return terminal(0); // Metadata writes remain entirely in memory.
    }};
    return {tools, children, writes};
  };
  await test("S3 gate failure starts no SDK and writes no metadata", async () => {
    const envelope = fixture(), simulated = mock(envelope, null, 2);
    await rejects(() => runOwner(simulated.tools, envelope), "S3 execution preflight");
    require(simulated.children.length === 0 && simulated.writes.length === 0, "Work started before readiness");
  });
  await test("all stages execute in order without quality promotion", async () => {
    const envelope = fixture(), simulated = mock(envelope);
    const result = await runOwner(simulated.tools, envelope);
    require(result.stage === "completed_measurements_quality_unclassified" && result.paper_effect_achieved === false,
      "Execution was promoted to quality");
    require(JSON.stringify(simulated.children) === JSON.stringify(envelope.plan.actions.map(row => row.role)), "Serial order differs");
  });
  await test("SDK failure publication runs once then later stages stop", async () => {
    const envelope = fixture(), simulated = mock(envelope, envelope.plan.actions[0].role);
    await rejects(() => runOwner(simulated.tools, envelope), "actual exit -9");
    require(simulated.children.length === 2 && simulated.children[1] === envelope.plan.actions[1].role, "Failure was not published or later work started");
  });
  await test("failed CPU stage stops subsequent children", async () => {
    const envelope = fixture(), simulated = mock(envelope, envelope.plan.actions[2].role);
    await rejects(() => runOwner(simulated.tools, envelope), "actual exit -9");
    require(simulated.children.length === 3, "Later stage started after CPU failure");
  });
  return {status: "native_tool_owner_in_memory_CPU_contracts_passed", passed,
    SDK_started: false, fixtures_published_as_actual_evidence: false};
}
