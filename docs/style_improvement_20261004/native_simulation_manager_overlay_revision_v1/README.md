# Native simulation-manager shutdown overlay: source plan

Status: mutable source proposal awaiting root/independent review. This author has not compiled, loaded, or executed the plugin, SDK, or any model. No installed SDK file has changed. Actual build, overlay origin, shutdown, and process-exit evidence remain null in `source_review.json`.

## Failure and candidate boundary

Root's v17 GDB run (`d52cb6`, session51089, completion96e2c9) reported inferior SIGSEGV during C `exit` handlers. The PC was inside the still-mapped simulation-manager plugin, offset `0x1291f`, at a virtual call through an unmapped target. The surrounding decrement-at-+8/dispose/+12/destroy sequence agrees with GNU11 `shared_ptr` control-block release. It does not identify which shared pointer or prove a service-unload ordering defect.

The fixed [official v6.0.0 implementation](https://github.com/isaac-sim/IsaacSim/blob/v6.0.0/source/extensions/isaacsim.core.simulation_manager/plugins/isaacsim.core.simulation_manager/PluginInterface.cpp) retains `g_stageUpdate` and `g_physicsEventSubscription` until global destruction. Its extension `onShutdown` only unsubscribes two integer subscriptions. `createStageUpdateNode` returns a handle which the original code discards. This provides a specific, reviewable shutdown candidate, not an established cause of the actual crash.

## Three exact source edits

`PluginInterface.shutdown.patch` applies to the downloaded 35,300-byte official source, SHA256 `5bf8d3a073301f60b0df397355c447583e740fe2e4392fc12b03f2d3989b995e`:

1. Store one `omni::kit::StageUpdateNode*`, initialized null.
2. Save the original `createStageUpdateNode(desc)` return value.
3. After the original two guarded unsubscribe calls, destroy only this plugin's node with the still-owned `g_stageUpdate`, clear the handle, reset the physics-event subscription, then reset this plugin's StageUpdatePtr.

The exact API is `destroyStageUpdateNode`, not `destroyNode`. Installed `IStageUpdate.h:116–118` declares creation/destruction; line164 defines StageUpdatePtr as `std::shared_ptr<StageUpdate>`. The default StageUpdate belongs to the shared USD context: this patch does not destroy it. Cleared pointers/handle and the original invalid-subscription guards make a second normal shutdown take the empty path. It does not add late API acquisition to `carbOnPluginShutdown`.

All other source bytes reverse exactly. Physics callbacks, stepping, counters, scene collection, time sampling, interface methods, and Python bindings are unchanged. Shutdown behavior intentionally changes: the header does not specify whether node destruction synchronously invokes `onDetach`, or a callback/thread barrier. Existing `g_timeStorage` cleanup stays in the original `onDetach`; this patch does not claim all globals are cleared or concurrent shutdown is proved safe. Public extension headers also do not guarantee Python/native shutdown ordering.

## Version and ABI evidence

| Item | Observed source evidence |
|---|---|
| Official source | v6.0.0 commit `40316786340b3f034a229d9e12650df1ac0b68ab`, VERSION `6.0.0-rc.59` |
| Installed runtime | `6.0.0-rc.59+release.41464.5f2772bc.gl` |
| Extension | `isaacsim.core.simulation_manager` 1.15.4 |
| Interface | ISimulationManager 2.0; complete ISimulationManager.h, TimeSampleStorage.h, UsdNoticeListener.h match installed bytes/SHA |
| Kit | `110.1.1+production.305458.6312fa25.gl` |
| USD | package `usd.py312.manylinux_2_35_x86_64.stock.release`, `0.25.11.kit.2-gl.19811` |
| Compiler | Installed ELF `.comment` GNU11.4.0; official repo checks GCC11 and specifies C++17 |
| Python ABI | cp312 / libpython3.12; keep installed Python binding unchanged |

Three public headers matching does not prove that every private-build source byte, compiler flag, linked dependency, or binary ABI is identical. The installed private commit `5f2772bc` has not been established as the tag commit. A build must record its own inputs and loader/symbol results. `source_review.json` pins source inputs, installed headers/config/plugin, and the full inverse derivation. Do not use IsaacSim main/6.1 or Kit110.3 headers.

## Exact target and minimum build inputs

The [extension premake](https://github.com/isaac-sim/IsaacSim/blob/v6.0.0/source/extensions/isaacsim.core.simulation_manager/premake5.lua) names the native project **`isaacsim.core.simulation_manager.plugin`**. The other project is the Python binding and need not be rebuilt for this proposal. A standalone four-translation-unit build is a local adaptation of that target, not a claimed execution of the official whole-repository build:

- Patched PluginInterface.cpp; unchanged PhysicsScene.cpp, TimeSampleStorage.cpp, UsdNoticeListener.cpp from the same tag. All four are pinned in the JSON. Compile all four, not just PluginInterface.cpp. The final standalone argv links installed SDK libraries while retaining the downloaded matching development headers; its first draft with downloaded link-library paths is preserved separately.
- Extension include directory (including PhysicsScene.h); current installed `isaacsim.core.includes/include` and `omni.kit.loop-isaac/include`; Kit `dev/include`, `dev/fabric/include`, and current usdrt.scenegraph include directory.
- Matching USD include directory, Python3.12 include directory, omni_physics release include directory, and USD physics-schema release include directory.
- Link against the installed SDK libraries: physxSchema, omni.usd, carb; the public USD helper adds usd_ar/arch/gf/plug/sdf/tf/usd/vt plus usdGeom/usdUtils/usdPhysics/ts, usd_python, Python3.12; pthread/dl/rt and the release TBB dependency. Use resolved actual library paths. The original ELF's DT_NEEDED is evidence about its realized links, not a complete list of include/build dependencies.

Source-grounded standalone argument requirements:

- `/usr/bin/g++-11`, `-std=c++17`, shared library and position-independent objects; RTTI/exceptions enabled.
- `ISAACSIM_CORE_SIMULATION_MANAGER_EXPORT`, `NDEBUG`, `__STDC_VERSION__=0` from the tag extension/root premakes.
- `OMNI_COMPILE_AS_MODULE` and `carb_eventdispatcher_IEventDispatcher=CARB_HEXVERSION(1,4)` from installed `kit/dev/premake5-public.lua:713–715`.
- Linux C++11 ABI1: tag repo.toml declares `linux_x86_64_cxx_abi=true`; public.lua:684–685 says the current default is1. An explicit `_GLIBCXX_USE_CXX11_ABI=1` documents that value.
- `TBB_USE_DEBUG=0`, `TBB_SUPPRESS_DEPRECATED_MESSAGES=1`, `-pthread` from public.lua's `add_usd`/`add_tbb`; `dl`, `pthread`, `rt` from those helpers/project_common.
- Do not add old `BOOST_ALL_DYN_LINK` rules: current `premake5-usd.lua` applies them to Windows USD older than24.11. Do not invent epsilon, interface, or exported-symbol changes.

These reproduce identified source settings. A full generated compiler-command equivalence has not been demonstrated. Optimization/debug-info choices of a standalone diagnostic build must be recorded explicitly. Future ELF checks should include class/machine, SONAME, DT_NEEDED/RUNPATH, undefined symbols and required GLIBCXX versions, then actual loader origin. Compilation alone is not ABI or exit validation.

## Minimum dependencies, without a whole runtime/ML pull

Root's isolated Packman project `/tmp/navisaacsim_native_build_deps_v1/minimal.packman.xml` selects only:

- `omni_physics` `110.1.11-52655635-release_omni_110.1-943e727b-manylinux_2_35_x86_64` (root reported successful pull; supplies the three physics API headers).
- The exact USD package above (root reported actual pull15883 completed, aa2a7c exit0).

The first package's `deps/schema-deps.packman.xml` specifies `usd_ext_physics` `25.11+feature.71522.e6698074.gl.manylinux_2_35_x86_64.release`. Root reported a separately downloaded/extracted schema package at `/tmp/navisaacsim_native_schema_25_11_v1`, archive SHA256 `699aa0fcc75cc56ff5addbf420fc6d16ad335abcdb69ad33e685c451cfb8702b`. Use its headers. Root reported its physxSchema library matches the installed library bytes; the standalone plan nevertheless selects installed runtime libraries. The downloaded USD development libraries differ in size/SHA from installed runtime libraries: matching package versions do not make those files byte-identical, and stripping is only a possible explanation. These selections are sufficient identified missing development dependencies; packages may contain runtime bytes. No separate header-only package was proved. Reuse already installed Kit/fabric/usdrt/omniUSD/GSL/Python headers where present. Do not pull CUDA, Torch, ROS, ML pip bundles, or another complete IsaacSim runtime for this target.

The official [repository build configuration](https://github.com/isaac-sim/IsaacSim/blob/v6.0.0/repo.toml) pulls many unrelated dependency/pip projects. This proposal does not run that broad build. Exact Packman commands and their actual outcomes belong to root's real dependency operations; this author has not rerun them.

## Workspace overlay and future execution boundary

Use a new workspace extension directory, copied from the **installed** extension, excluding `__pycache__`/pyc. Preserve its installed config, Python files, binding, data and headers byte-for-byte. Replace only `bin/libisaacsim.core.simulation_manager.plugin.so` after a reviewed build. The installed toml differs from the tag by authors metadata and removed test-only ML dependencies, so copying the installed config avoids introducing unrelated dependencies.

Prefer an explicit complete copy over a Python-directory symlink. Source file origins therefore genuinely change: terminal helper v9 currently pins two Python paths; a separate reviewed v10 should migrate only those two origins while keeping their full SHA unchanged. Use `PYTHONDONTWRITEBYTECODE=1`. Do not alter sys.modules/module.__file__ to disguise origins.

Select this extension with explicit Kit `/app/exts/devPaths` pointing at that exact directory **before** extension resolution. [Kit's source-linking documentation](https://docs.omniverse.nvidia.com/kit/docs/kit-manual/110.1.1/guide/extensions_advanced.html#extensions-source-linking-dev-paths) describes development-path priority. Confirm the actual extension-manager path, Python function origins, binding file, plugin loaded mapping and binary SHA. An argv/path proposal alone does not establish which binary loaded.

A future isolated run must preserve existing physical setup/cleanup gates and record true SDK process return, including any crash. It must not suppress normal exit handlers or relax prior weakref/owner checks. Successful cleanup booleans and a successful native process exit are separate observations. This source proposal neither launches that run nor declares the current SIGSEGV solved.

## Optional reproducible compiler-only executor

`standalone_build_arguments.json` contains the exact current argument vectors. `build.py` pins that JSON and the source-review JSON, verifies the three derivation edits, hashes supplied sources/current SDK inputs/libraries, preprocesses all four units to identify and hash their actually used headers, then compiles and links in an exclusive new output directory. It rechecks those byte identities afterwards and retains actual command return codes/logs and ELF metadata. It creates no overlay and does not import or load the SDK/plugin. A missing dependency or nonzero command produces a failed report and preserved partial outputs; it does not silently retry or relax a flag.

Future command, **not executed by this author**:

```sh
python docs/style_improvement_20261004/native_simulation_manager_overlay_revision_v1/build.py --output /tmp/navisaacsim_native_plugin_repro_build_v1
```

The directory must not already exist. No installed SDK file is written. The standalone choices `-O2/-g` and link-time `-z defs` are declared local choices; they are not a claim of complete official generated-argv equivalence. Root's separate actual four-object compilation is not rerun by this author or by merely authoring this script. Root's `-fvisibility=hidden` choice likewise belongs to its recorded local build, not an inferred original compiler command. Actual overlay creation, loading and exit testing remain separate root operations.
