import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs a ROCm GPU and the Kimi-K3.1 ROCm vLLM image")


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
