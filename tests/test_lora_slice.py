from types import SimpleNamespace

import pytest
import torch

from k31lora import lora_slice
from k31lora.lora_slice import EP_TAG, TP_TAG

E, R = 896, 8
LOCAL = E // R


def ep_spec(rank):
    return SimpleNamespace(ep_rank=rank, global_num_experts=E, local_num_experts=LOCAL)


def ep_adapter(rank, size=R, experts=E, expert_dim=LOCAL):
    return {
        EP_TAG: torch.tensor([rank, size, experts], dtype=torch.float32),
        "m.layers.1.mlp.experts.w1.lora_A.weight": torch.zeros(expert_dim, 4, 8),
        "m.layers.1.mlp.experts.shared.lora_A.weight": torch.zeros(1, 4, 8),
        "m.layers.1.self_attn.q_proj.lora_A.weight": torch.zeros(4, 8),
    }


def test_untagged_adapter_is_unchanged():
    w = {"a.lora_A.weight": torch.zeros(2, 2)}
    assert lora_slice.localize(w, adapter_manager=None) is w


def test_both_tags_rejected():
    w = {EP_TAG: torch.tensor([0, 8, 896]), TP_TAG: torch.tensor([0, 2])}
    with pytest.raises(RuntimeError, match="both"):
        lora_slice.localize(w, adapter_manager=None)


def test_tagged_adapter_needs_moe_lora_layers():
    with pytest.raises(RuntimeError, match="without MoE LoRA"):
        lora_slice.localize(ep_adapter(0), SimpleNamespace(modules={}))


@pytest.mark.parametrize("rank", [0, 3, 7])
def test_ep_block_matching_this_worker_is_stripped(rank):
    w = ep_adapter(rank)
    out = lora_slice.localize_ep(w, ep_spec(rank))
    assert EP_TAG not in out
    assert set(out) == set(w) - {EP_TAG}
    assert all(out[k] is w[k] for k in out)


def test_ep_through_localize_reads_the_worker_layout():
    moe = SimpleNamespace(
        w13_lora_b_stacked=None,
        base_layer=SimpleNamespace(expert_placement_strategy="linear"),
        moe_config=SimpleNamespace(moe_parallel_config=SimpleNamespace(enable_eplb=False)),
    )
    mgr = SimpleNamespace(modules={"x": moe}, moe_ep_load_spec=ep_spec(5))
    assert EP_TAG not in lora_slice.localize(ep_adapter(5), mgr)
    with pytest.raises(RuntimeError, match="placement mismatch"):
        lora_slice.localize(ep_adapter(4), mgr)


@pytest.mark.parametrize(
    "tag,spec",
    [((2, R, E), ep_spec(3)), ((3, 4, E), ep_spec(3)), ((3, R, 512), ep_spec(3))],
)
def test_ep_mismatch_raises(tag, spec):
    with pytest.raises(RuntimeError, match="placement mismatch"):
        lora_slice.localize_ep(ep_adapter(*tag), spec)


def test_ep_needs_expert_parallelism():
    with pytest.raises(RuntimeError, match="without expert parallelism"):
        lora_slice.localize_ep(ep_adapter(0), None)


@pytest.mark.parametrize("placement,eplb", [("round_robin", False), ("linear", True)])
def test_ep_needs_linear_placement_without_eplb(placement, eplb):
    with pytest.raises(RuntimeError, match="linear expert placement"):
        lora_slice.localize_ep(ep_adapter(0), ep_spec(0), placement, eplb)


def test_ep_wrong_expert_dim_raises():
    with pytest.raises(RuntimeError, match="expert dim"):
        lora_slice.localize_ep(ep_adapter(0, expert_dim=E), ep_spec(0))


def tp_moe(rank=1, size=2, shard=16, use_ep=False):
    return SimpleNamespace(tp_rank=rank, tp_size=size, _tp_shard_size=shard, use_ep=use_ep)


def tp_adapter(rank=1, size=2, shard=16):
    return {
        TP_TAG: torch.tensor([rank, size]),
        "l.experts.w1.lora_B.weight": torch.zeros(8, shard, 4),
        "l.experts.w3.lora_B.weight": torch.zeros(8, shard, 4),
        "l.experts.w2.lora_A.weight": torch.zeros(8, 4, shard),
        "l.experts.w1.lora_A.weight": torch.zeros(8, 4, 32),
    }


@pytest.fixture
def no_vllm_patch(monkeypatch):
    calls = []
    monkeypatch.setattr(lora_slice, "install_presliced_tp_passthrough", lambda: calls.append(1))
    return calls


def test_tp_shard_matching_this_worker_is_stripped(no_vllm_patch):
    out = lora_slice.localize_tp(tp_adapter(), [tp_moe(), tp_moe()])
    assert TP_TAG not in out and len(out) == 4
    assert no_vllm_patch == [1]


@pytest.mark.parametrize(
    "adapter,moes,match",
    [
        (tp_adapter(rank=0), [tp_moe()], "layout mismatch"),
        (tp_adapter(), [tp_moe(use_ep=True)], "layout mismatch"),
        (tp_adapter(rank=0, size=1), [tp_moe(rank=0, size=1)], "layout mismatch"),
        (tp_adapter(), [tp_moe(), tp_moe(shard=8)], "disagree"),
        (tp_adapter(shard=8), [tp_moe()], "expert shard"),
    ],
)
def test_tp_mismatch_raises(no_vllm_patch, adapter, moes, match):
    with pytest.raises(RuntimeError, match=match):
        lora_slice.localize_tp(adapter, moes)
    assert no_vllm_patch == []


@pytest.mark.parametrize(
    "bi,alpha,r,ok",
    [("1", 64, 32, True), ("1", 32, 32, True), ("1", 48, 32, False), ("1", None, 32, False), ("0", 48, 32, True)],
)
def test_expert_lora_scaling_is_a_power_of_two_under_bi(monkeypatch, bi, alpha, r, ok):
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", bi)
    req = SimpleNamespace(peft_config={"target_modules": ["q_proj", "experts"], "lora_alpha": alpha, "r": r})
    if ok:
        lora_slice._check_expert_scaling(req)
    else:
        with pytest.raises(RuntimeError, match="power-of-two"):
            lora_slice._check_expert_scaling(req)


def test_scaling_not_checked_without_expert_targets(monkeypatch):
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    lora_slice._check_expert_scaling(SimpleNamespace(peft_config={"target_modules": "q_proj", "lora_alpha": 3, "r": 2}))
