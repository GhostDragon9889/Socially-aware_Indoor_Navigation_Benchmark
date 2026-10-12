Installation
============

The supported runtime is IsaacLab 3.0-EA / Isaac Sim 6.1. There are two setup methods.

External SDK
------------

.. code-block:: bash

   python3 scripts/bootstrap_reproduction.py configure --lab-root ../IsaacLab

Pinned uv installation
----------------------

.. code-block:: bash

   python3 scripts/bootstrap_reproduction.py install

Both methods use the SDK Python and locked project additions. Run project commands through ``scripts/run_external_python.py``. For resources, EULA, provider verification and editor setup, see the repository's ``docs/guides/installation.en.md``.
