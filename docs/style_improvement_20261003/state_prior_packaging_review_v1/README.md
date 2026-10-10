# V11 source distribution CPU review

Status: **passed**. Full evidence: `packaging_review.json`, `sdist_verification.json`, and `extracted_frontend_checks.json`.

The actual archive is `artifacts/protomotions-3.1.0-v11-review.tar.gz`: 41,618,202 bytes, SHA256 `cc4598f78f02a2c1ef4283748d0957ab659799081ae5b77f970b0f9fa680643f`.

## Verified scope

- Used the existing declared PEP517 backend, `setuptools.build_meta.build_sdist`, in a separate `/tmp` copy. Setuptools 81.0.0 and wheel 0.48.0 were already installed. No dependency installation, production implementation write, GPU, Kit, or training.
- Actual archive bytes match all nine requested entrypoints, the original 26 required paths, and the two new helper/profile paths. There are 1,016 regular archive files from 1,008 copied inputs.
- All 183 selected V11 source files still match the immutable V11 manifest. The distribution contains 170 of these exact files; the remaining 13 are evidence-only docs not selected by the existing MANIFEST rules, explicitly listed in the report.
- No external weights, AMASS packages, SMPL body parameters/meshes/rigs, generated outputs, or selected scene dependencies are in the archive. All six original `protomotions.egg-info` files retain their prior bytes. The earlier source distribution remains unchanged.
- Extracted `simulate_paper_pedestrians.py --help` and `train_paper_amp.py --help` return zero. The new state-prior profile validates with the new flag true, the old discriminator flag false, state_action/history10x195, and history RMS. Every imported owned module is from the extracted tree, with production import paths removed.
- Extracted resource preflight rejects absent packaged AMASS data and forbidden from-scratch. It succeeds against explicitly external original AMASS, diffusion, public Pacer, and SMPL resources; the public Pacer SHA is recomputed. No Torch, NumPy, or SDK modules are imported and no models are deserialized. This is a local file/header/configuration gate, not a physics or style result.

## Snapshot boundary

The build report completed at **2026-10-03 02:40:13 UTC**. `guide_snapshots/` preserves the actual copied text for the four required guides. Parent updated those guides afterward; the current text hashes are recorded separately. These four Markdown paths are outside the selected V11 source183. The actual built archive was preserved without rebuilding.

An earlier agent message incorrectly described humanoid README as a selected V11 member. Actual manifest inspection has no README basename and all183 selected files are exact. No V12 was created.

## Evidence and reproduction

`build_isolated_v11_sdist.py` follows the earlier packaging selection/backend rules. `check_extracted_v11_frontend.py` checks the actual built archive in `/tmp`. `finalize_review.py` preserves its bytes and binds the CPU proof, snapshots, frozen source, and original egg-info.

The first resource check omitted the external diffusion path, which the CLI rejects before resource inspection. Its diagnostic is retained in `extracted_frontend_failed_parser.log`. The final gate supplies the immutable external diffusion path explicitly; the three successful/rejected resource cases each retain arguments, stdout, and stderr. The nine entrypoints are byte checks; two cold help frontends were executed.

Run the scripts with the existing SDK Python, `CUDA_VISIBLE_DEVICES=''`, `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, and `PYTHONDONTWRITEBYTECODE=1`. The archived proof binds this particular package snapshot; a later build must use a new evidence directory.
