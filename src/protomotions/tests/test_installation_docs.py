
from protomotions.repository import repository_root
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path


REPO_ROOT = repository_root(__file__)


def test_installation_documents_only_the_two_supported_sdk_methods():
    installation = (REPO_ROOT / "docs/source/getting_started/installation.rst").read_text()
    assert "bootstrap_reproduction.py configure" in installation
    assert "bootstrap_reproduction.py install" in installation
    assert "run_external_python.py" in installation
    assert "conda create" not in installation
    assert "pip install" not in installation
    assert "prepare-runtime" not in installation
