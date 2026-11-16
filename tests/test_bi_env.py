import pytest

from k31lora import bi_env


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in list(bi_env.PINNED_BI) + list(bi_env.RETIRED) + ["VLLM_BATCH_INVARIANT"]:
        monkeypatch.delenv(k, raising=False)


@pytest.mark.parametrize("x,want", [(1, True), (2, True), (0.5, True), (64 / 32, True), (3, False), (0, False), (-2, False), (1.5, False)])
def test_is_pow2(x, want):
    assert bi_env.is_pow2(x) is want


def test_unset_env_passes():
    bi_env.check("test")


@pytest.mark.parametrize("bi", ["0", "1"])
def test_retired_switch_accepts_only_its_fixed_value(monkeypatch, bi):
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", bi)
    monkeypatch.setenv("K31_BI_MOE_ENGINE", "dequant")
    bi_env.check("test")
    monkeypatch.setenv("K31_BI_MOE_ENGINE", "fused")
    with pytest.raises(RuntimeError, match="K31_BI_MOE_ENGINE"):
        bi_env.check("test")


def test_retired_switch_without_values_must_stay_unset(monkeypatch):
    monkeypatch.setenv("K31_KDA_BI_DUMP", "1")
    with pytest.raises(RuntimeError, match="K31_KDA_BI_DUMP"):
        bi_env.check("test")


def test_pinned_switch_under_bi(monkeypatch):
    monkeypatch.setenv("K31_BI_MOE", "0")
    bi_env.check("test")  # not pinned outside BI
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    with pytest.raises(RuntimeError, match="K31_BI_MOE"):
        bi_env.check("test")
    monkeypatch.setenv("K31_BI_MOE", "1")
    bi_env.check("test")


def test_kda_bi_needs_batch_invariance(monkeypatch):
    monkeypatch.setenv("K31_KDA_BI", "1")
    with pytest.raises(RuntimeError, match="K31_KDA_BI"):
        bi_env.check("test")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    bi_env.check("test")
