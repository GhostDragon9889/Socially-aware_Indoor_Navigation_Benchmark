"""Use the original complete payload reader with truthful ToDesk background admission."""
from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OWNER_DIR = ROOT / 'docs/style_improvement_20261003/state_only_matched_q_actual_runs_v1'
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(OWNER_DIR))
import background_GPU_evidence_v2 as background  # noqa: E402
import state_prior_checks as original_helper  # noqa: E402


def prepare_reader():
    binding = json.loads(background.BINDING.read_text())
    for row in binding['new_files'] + binding['original_files']:
        background.require(background.record(row['path']) == row and not Path(row['path']).stat().st_mode & 0o222,
                           'Exact readonly original/v2 reader sources required')
    source = inspect.getsource(original_helper._verify_launch)
    old = 'launch["no_parallel_GPU_at_launch"] is True'
    new = 'launch["no_parallel_training_GPU_at_launch"] is True'
    background.require(source.count(old) == 1, 'One launch GPU metadata field expected')
    revised = source.replace(old, new)
    background.require(revised.replace(new, old) == source, 'Original launch function inverse bytes required')
    namespace = dict(original_helper.__dict__)
    exec(compile(revised, __file__, 'exec'), namespace)
    revised_verify = namespace['_verify_launch']

    def verify_with_background(*args, **kwargs):
        launch, proof = revised_verify(*args, **kwargs)
        background.validate_launch_evidence(launch, args[4], args[5])
        return launch, proof

    original_helper._verify_launch = verify_with_background
    entry = HERE / 'audit_state_only_training.py'
    spec = importlib.util.spec_from_file_location('state_only_original_complete_payload_reader', entry)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    original_write = core.write

    def write_with_execution_identity(path, document):
        if Path(path).name == 'audit.json':
            document.update(actual_execution_adapter=background.record(__file__),
                            execution_background_helper=background.record(background.__file__),
                            background_metadata_source_binding=background.record(background.BINDING),
                            original_execution_reader=background.record(entry),
                            original_full_payload_math_unchanged=True,
                            launch_GPU_metadata_adapted_for_exact_ToDesk_video=True)
        return original_write(path, document)

    core.write = write_with_execution_identity
    return core


def main():
    return prepare_reader().run()


if __name__ == '__main__':
    main()
