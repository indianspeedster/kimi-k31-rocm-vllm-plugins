import importlib
import pathlib

import pytest

try:
    import tomllib
except ImportError:  # Python 3.10
    tomllib = pytest.importorskip("tomli")

ROOT = pathlib.Path(__file__).resolve().parents[1]
PLUGINS = {"k31lora", "k31kdabi", "k31xlora", "k31dsabi"}


def entry_points():
    with open(ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)["project"]["entry-points"]["vllm.general_plugins"]


def test_every_plugin_is_an_entry_point_and_a_package():
    eps = entry_points()
    assert set(eps) == PLUGINS
    assert all(v == f"{k}:register" for k, v in eps.items())
    with open(ROOT / "pyproject.toml", "rb") as f:
        assert set(tomllib.load(f)["tool"]["setuptools"]["packages"]) == PLUGINS


@pytest.mark.parametrize("name", ["k31kdabi", "k31dsabi"])
def test_bi_only_plugins_do_nothing_outside_their_arm(monkeypatch, name):
    # register() must return before touching vLLM when VLLM_BATCH_INVARIANT is not 1.
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    mod = importlib.import_module(name)
    assert mod.register() is None
