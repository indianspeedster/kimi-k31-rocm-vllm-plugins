import pathlib
import sys

import pytest

# the trainer's old import names (compat/) resolve without installing the package
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "compat"))


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs a ROCm GPU and the ROCm vLLM image")


def pytest_collection_modifyitems(config, items):
    try:
        import torch

        has_gpu = torch.cuda.is_available() and torch.version.hip is not None
    except ImportError:
        has_gpu = False
    if has_gpu:
        return
    skip = pytest.mark.skip(reason="no ROCm GPU")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
