import importlib
import pathlib

import pytest

try:
    import tomllib
except ImportError:  # Python 3.10
    tomllib = pytest.importorskip("tomli")

ROOT = pathlib.Path(__file__).resolve().parents[1]
COMPAT = {"k31lora", "k31kdabi", "k31dsabi", "k31xlora"}


def pyproject():
    with open(ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)


def test_one_entry_point():
    assert pyproject()["project"]["entry-points"]["vllm.general_plugins"] == {
        "vllm_rl_plugins": "vllm_rl_plugins:register"}


def test_every_package_is_listed():
    listed = set(pyproject()["tool"]["setuptools"]["packages"])
    on_disk = {".".join(p.parent.relative_to(ROOT).parts) for p in (ROOT / "vllm_rl_plugins").rglob("__init__.py")}
    assert listed == on_disk | COMPAT
    assert pyproject()["tool"]["setuptools"]["package-dir"] == {n: f"compat/{n}" for n in COMPAT}


@pytest.mark.parametrize("old, new", [
    ("k31lora.bi_env", "vllm_rl_plugins.models.kimi_k31.env"),
    ("k31lora.bi_lora", "vllm_rl_plugins.common.bi_lora"),
    ("k31lora.lora_slice", "vllm_rl_plugins.common.lora_slice"),
])
def test_old_names_are_the_same_module_objects(old, new):
    assert importlib.import_module(old) is importlib.import_module(new)


def test_compat_files_point_at_existing_modules():
    for f in (ROOT / "compat").rglob("*.py"):
        for line in f.read_text().splitlines():
            if "import_module(\"vllm_rl_plugins" in line:
                mod = line.split('import_module("')[1].split('"')[0]
                rel = pathlib.Path(*mod.split("."))
                assert (ROOT / f"{rel}.py").exists() or (ROOT / rel / "__init__.py").exists(), (f, mod)


@pytest.mark.parametrize("mod", ["vllm_rl_plugins.models.kimi_k31.rocm.kda"])
def test_bi_only_code_does_nothing_outside_its_arm(monkeypatch, mod):
    # register() must return before touching vLLM when VLLM_BATCH_INVARIANT is not 1.
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    assert importlib.import_module(mod).register() is None
