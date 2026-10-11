Installation
============

The benchmark defaults to IsaacLab 3.0 EA / Isaac Sim 6.1 with an isolated
project runtime. Its source package supports Python 3.12 on Linux x86_64.
The other backend sections retain upstream or historical environment references;
backend availability and physics must be verified separately.

**Backend references and migration target:**

.. raw:: html

   <p>
     <a href="https://pypi.org/project/newton/1.0.0/"><img src="https://img.shields.io/badge/Newton-1.0.0-brightgreen.svg" alt="Newton"></a>
     <a href="https://github.com/isaac-sim/IsaacLab/commit/ae37b028ea415c91ea2bc32609efcd759ed2b974"><img src="https://img.shields.io/badge/IsaacLab-3.0-blue.svg" alt="IsaacLab"></a>
     <a href="https://developer.nvidia.com/isaac-gym"><img src="https://img.shields.io/badge/IsaacGym-Preview_4-blue.svg" alt="IsaacGym"></a>
     <a href="https://github.com/Genesis-Embodied-AI/Genesis"><img src="https://img.shields.io/badge/Genesis-untested-lightgrey.svg" alt="Genesis"></a>
     <a href="https://github.com/google-deepmind/mujoco"><img src="https://img.shields.io/badge/MuJoCo-3.0+-orange.svg" alt="MuJoCo"></a>
   </p>

.. note::

   We recommend creating a **separate virtual environment** for each simulator to avoid dependency conflicts.
   We recommend using **conda** or **venv** for IsaacGym, Genesis, and MuJoCo, and **uv** for IsaacLab and Newton.

Which installation path?
------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 30 48

   * - Simulator
     - Supported install
     - Notes
   * - MuJoCo, Newton, Genesis
     - Source checkout **or** uv dependency
     - Genesis is experimental.
   * - IsaacLab
     - **Pinned source checkout**
     - IsaacLab 17.0.2 / product 3.0 EA with Isaac Sim 6.1.0.0; Python 3.12, Linux x86_64.
   * - IsaacGym
     - **Source checkout only**
     - IsaacGym is not distributed on PyPI: you download it from NVIDIA and
       install it by hand, and it requires **Python 3.8**.

Use a source checkout if you want the pretrained checkpoints, motion files, or
the ``examples/`` experiments — those live in Git LFS, not in the package.

Prerequisites
-------------

After cloning the repository, fetch and check out files stored in Git LFS:

.. code-block:: bash

   git lfs install
   git lfs pull

This can take a while because pretrained checkpoints, motion files, meshes, and
USD assets are large. If you fetch a subset of assets manually, make sure the
files are checked out and not still Git LFS pointer files. Pointer files start
with ``version https://git-lfs.github.com/spec/v1`` and can cause errors such as
``is not a valid usda layer`` when IsaacLab loads robot assets.

Upstream ProtoMotions as a dependency (uv)
------------------------------------------

These upstream package examples do not install the benchmark scenes, local
models or CrowdSim reproduction configuration. Follow the benchmark IsaacLab
bootstrap below for the default runtime.

Install upstream ProtoMotions directly from Git. Robot meshes and USD assets are Git LFS
objects, so the source must be fetched with LFS enabled — ``lfs = true``
requires uv 0.11.32+:

.. code-block:: bash

   uv init --python 3.11 my-project
   cd my-project
   uv add --lfs "protomotions[newton] @ git+https://github.com/NVlabs/ProtoMotions.git"

Equivalently, configure the dependency in the downstream ``pyproject.toml``:

.. code-block:: toml

   [project]
   dependencies = ["protomotions[newton]"] # or [mujoco] / [isaaclab] / [genesis]

   [tool.uv]
   required-version = ">=0.11.32"

   [tool.uv.sources]
   protomotions = { git = "https://github.com/NVlabs/ProtoMotions.git", lfs = true }

Then run training through the installed entry point:

.. code-block:: bash

   uv run protomotions train-agent \
       --robot-name g1 --simulator newton \
       --experiment-path experiments/my_experiment.py \
       --experiment-name my_run \
       --motion-file data/my_motion.pt \
       --num-envs 4096 --batch-size 16384

``uv run protomotions info`` prints the resolved asset root and which simulator
modules are importable.

The package ships the Python modules and the full robot asset tree **except**
the SMPL/SMPL-H assets, which carry their own licence terms. Pretrained
checkpoints, motion files, and the ``examples/`` experiments are not included;
keep a Git LFS checkout and set ``PROTOMOTIONS_ASSET_ROOT`` if you need them.

IsaacLab as a dependency
~~~~~~~~~~~~~~~~~~~~~~~~

The supported IsaacLab stack is a pinned source workspace rather than a
standalone ProtoMotions dependency resolution. Follow the IsaacLab procedure
below to create the SDK and its separate project overlay. The controlled
bootstrap validates the SDK source, exact locks, binary providers and project
requirements. A package extra alone does not reproduce this environment.

Choose Your Simulator(s)
------------------------

IsaacGym
~~~~~~~~

IsaacGym requires **Python 3.8**. Use a compatible historical upstream
ProtoMotions checkout for this section; the current benchmark source package
requires Python 3.12.

1. Create a conda environment:

   .. code-block:: bash

      conda create -n isaacgym python=3.8
      conda activate isaacgym

2. Download IsaacGym Preview 4:

   .. code-block:: bash

      wget https://developer.nvidia.com/isaac-gym-preview-4
      tar -xvzf isaac-gym-preview-4

3. Install IsaacGym Python API:

   .. code-block:: bash

      pip install -e isaacgym/python

4. Install ProtoMotions and dependencies:

   .. code-block:: bash

      pip install -e /path/to/protomotions
      pip install -r /path/to/protomotions/requirements_isaacgym.txt

IsaacLab
~~~~~~~~

The benchmark defaults to IsaacLab 3.0 EA source
``ae37b028ea415c91ea2bc32609efcd759ed2b974`` (Python package 17.0.2) with
Isaac Sim 6.1.0.0 and Python 3.12.13. Use the checked-in SDK manifest/lock and
project overlay, rather than installing broad optional groups into an SDK.

From the benchmark source root, install a new SDK/runtime:

.. code-block:: bash

   uv python install 3.12.13
   uv run --no-project --python 3.12.13 python scripts/bootstrap_reproduction.py install --profile isaaclab3_ea_sim61
   export NAV_LAB="$PWD/.runtime/IsaacLab61"
   export NAV_PY="$PWD/.runtime/isaaclab3_ea_sim61/.venv/bin/python"
   "$NAV_PY" -m social_navigation.runtime_environment check --lab-root "$NAV_LAB"

Reuse a matching installed SDK with
``python3 scripts/bootstrap_reproduction.py prepare-runtime --profile
isaaclab3_ea_sim61 --lab-root /absolute/path/to/IsaacLab``. This creates a
project-owned overlay and preserves the SDK's source and binary providers.
Accept NVIDIA's EULA before launching Kit.

The root ``isaaclab`` extra names the new stack; ``isaaclab6`` identifies
historical Lab 12 / Sim 6.0 dependencies. These package versions are sourced from
pinned IsaacLab Git subdirectories for uv metadata resolution, and are not a
replacement for the SDK workspace lock. Simulator extras are mutually exclusive.
The frozen old SDK profile references the earlier project extra and cannot be
fresh-installed or reapplied against migrated source; reuse an already installed
matching legacy SDK only. Original old locks and receipts remain unchanged.

Existing Sim6.0 recordings retain their original environment identity. The new
runtime is installed and undergoing independent physical acceptance; dependency
or CPU checks alone do not establish 60-second gait or paper performance.

Genesis (Experimental)
~~~~~~~~~~~~~~~~~~~~~~

Genesis requires **Python 3.10**.

1. Create a conda environment:

   .. code-block:: bash

      conda create -n genesis python=3.10
      conda activate genesis

2. Install `Genesis <https://genesis-world.readthedocs.io/en/latest/index.html>`_

3. Install ProtoMotions and dependencies:

   .. code-block:: bash

      pip install -e /path/to/protomotions
      pip install -r /path/to/protomotions/requirements_genesis.txt

Newton (standalone historical environment)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default Lab3EA SDK already supplies Newton 1.5.2 and Warp 1.16. Reuse the
project SDK overlay to test that backend; do not install the root historical
``newton`` extra into it. The separate Newton 1.0 instructions below retain the
upstream standalone environment, not the benchmark's Sim6.1 profile.

Newton is a GPU-accelerated physics simulator built on NVIDIA Warp, available on PyPI.
For full installation details, see the `Newton Installation Guide <https://newton-physics.github.io/newton/1.0.0/guide/installation.html>`__.

**Requirements**: Python 3.10+ (3.11+ recommended), NVIDIA GPU (compute capability >= 5.0), driver 545+

1. Create a virtual environment:

   .. code-block:: bash

      python -m venv .venv_newton
      source .venv_newton/bin/activate

2. Install PyTorch and Newton:

   .. code-block:: bash

      pip install torch --index-url https://download.pytorch.org/whl/cu124
      pip install "newton[sim]==1.0.0"

   ``newton[examples]`` and importers can add a second ``usd-core`` provider. Keep that optional GUI environment separate from IsaacLab/Isaac Sim.

3. Install ProtoMotions and dependencies:

   .. code-block:: bash

      pip install -e /path/to/protomotions
      pip install -r /path/to/protomotions/requirements_newton.txt

.. note::

   On Python 3.10, ``imgui-bundle`` (a dependency of ``newton[examples]``) has no prebuilt
   wheel and compiles from source, which can take 10-20 minutes. Python 3.11+ has prebuilt
   wheels and installs instantly.

MuJoCo (CPU-only)
~~~~~~~~~~~~~~~~~

MuJoCo is a CPU-only backend for quick testing and debugging without GPU. It supports single environment only (``num_envs=1``).

**Requirements**: Python 3.10+, No GPU required

1. Create a conda environment:

   .. code-block:: bash

      conda create -n protomotions_mujoco python=3.10
      conda activate protomotions_mujoco

2. Install PyTorch CPU version (lighter, no CUDA needed):

   .. code-block:: bash

      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

3. Install ProtoMotions and dependencies:

   .. code-block:: bash

      pip install -e /path/to/protomotions
      pip install -r /path/to/protomotions/requirements_mujoco.txt

4. Run inference with MuJoCo:

   .. code-block:: bash

      python protomotions/inference_agent.py \
        --checkpoint data/pretrained_models/motion_tracker/g1-bones-deploy/last.ckpt \
        --motion-file data/motion_for_trackers/g1_bones_seed_mini.pt \
        --simulator mujoco \
        --num-envs 1

   This example uses the shipped G1 motion tracker and matching motion data. The
   checkpoint directory includes the required ``resolved_configs_inference.pt``
   file.

.. note::

   MuJoCo backend is intended for quick policy validation and debugging. For training or large-scale evaluation, use GPU-accelerated backends (IsaacGym, IsaacLab, Newton, Genesis).

Troubleshooting
---------------

IsaacLab Issues
~~~~~~~~~~~~~~~

**Torch Inductor Warning**

On smaller GPUs, IsaacLab evaluation may print a warning similar to:

.. code-block:: text

   Not enough SMs to use max_autotune_gemm mode

This is a non-fatal PyTorch performance warning. Evaluation can continue unless
it is followed by an actual traceback.

IsaacGym Issues
~~~~~~~~~~~~~~~

**libpython Error**

If you encounter ``libpython`` related errors, you need to set the ``LD_LIBRARY_PATH`` to your conda environment:

.. code-block:: bash

   # First, check your conda environment path
   conda info -e
   
   # Then set LD_LIBRARY_PATH (replace with your actual conda env path)
   export LD_LIBRARY_PATH=/path/to/conda/envs/your_env/lib:$LD_LIBRARY_PATH
   
   # For example:
   export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:$LD_LIBRARY_PATH

To make this permanent for only this conda environment, add activation hooks:

.. code-block:: bash

   mkdir -p "${CONDA_PREFIX}/etc/conda/activate.d" "${CONDA_PREFIX}/etc/conda/deactivate.d"
   cat > "${CONDA_PREFIX}/etc/conda/activate.d/isaacgym-libpython.sh" <<'EOF'
   export _OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
   export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
   EOF
   cat > "${CONDA_PREFIX}/etc/conda/deactivate.d/isaacgym-libpython.sh" <<'EOF'
   export LD_LIBRARY_PATH="${_OLD_LD_LIBRARY_PATH:-}"
   unset _OLD_LD_LIBRARY_PATH
   EOF

**Memory Issues**

If you run into memory issues during training:

.. code-block:: bash

   # Reduce number of environments in your training command
   --num-envs 1024

Next Steps
----------

After installation, proceed to the :doc:`quickstart` guide to train your first agent or run pre-trained models.
