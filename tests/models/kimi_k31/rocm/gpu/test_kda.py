"""Kimi-K3.1 BI KDA on one GPU: the KDA chunk call gives a token the same bits however its sequence is cut or batched."""

import pytest
import torch

from .golden import check

pytestmark = pytest.mark.gpu

H, K = 12, 128  # Kimi-K3.1 KDA heads per rank at TP8, head_dim
LOWER = -5.0  # gate_lower_bound


@pytest.fixture(scope="module")
def kda(tmp_path_factory):
    import os

    os.environ.setdefault("K31_KDA_BI_CACHE", str(tmp_path_factory.mktemp("fla")))
    from vllm_rl_plugins.models.kimi_k31.rocm import kda as k31kdabi

    pytest.importorskip("fla.ops.kda")
    k31kdabi.pin_fla_autotune()
    return k31kdabi


def params(seed=100):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(H, generator=g) * 0.5).cuda(), (torch.randn(H * K, generator=g) * 0.5).cuda()


def inputs(T, seed):
    g = torch.Generator().manual_seed(seed)
    q, k, v, gate = (torch.randn(1, T, H, K, generator=g).bfloat16().cuda() for _ in range(4))
    beta = torch.randn(1, T, H, generator=g).bfloat16().cuda()
    return (q, k, v, gate, beta) + params()


def run(kda, x, lens, init=None, final=False):
    q, k, v, gate, beta, A_log, dt_bias = x
    cu_cpu = torch.tensor([0] + torch.tensor(lens).cumsum(0).tolist(), dtype=torch.long)
    return kda.kda_chunk(q, k, v, gate, beta, A_log, dt_bias, LOWER, cu_cpu.cuda(), init=init, final=final, cu_cpu=cu_cpu)


def cut(x, a, b):
    return tuple(t[:, a:b] if t.dim() >= 3 else t for t in x)


def test_prefix_bits_do_not_depend_on_sequence_length(kda):
    # every prefix length 1..T, packed as separate sequences in one call (the decode tail is recomputed this way)
    T = 256
    x = inputs(T, 2)  # an input where the FLA sub-chunk fix changes bits, so the golden catches its loss
    full, _ = run(kda, x, [T])
    lens = list(range(1, T + 1))
    prefixes = tuple(torch.cat([t[:, :n] for n in lens], 1) if t.dim() >= 3 else t for t in x)
    out, _ = run(kda, prefixes, lens)
    off = 0
    bad = []
    for n in lens:
        if not torch.equal(out[0, off : off + n], full[0, :n]):
            bad.append(n)
        off += n
    assert not bad, f"prefix lengths with different bits: {bad[:20]}"
    check("kdabi_chunk_T256", full)


def test_packed_sequences_match_each_alone(kda):
    # the rollout runs every request's window in one call
    a, b = inputs(150, 1), inputs(90, 2)
    packed = tuple(torch.cat([u, w], 1) if u.dim() >= 3 else u for u, w in zip(a, b))
    out, _ = run(kda, packed, [150, 90])
    oa, _ = run(kda, a, [150])
    ob, _ = run(kda, b, [90])
    assert torch.equal(out[0, :150], oa[0]) and torch.equal(out[0, 150:], ob[0])


@pytest.mark.parametrize("split", [64, 128])
def test_chunk_aligned_continuation_matches_one_pass(kda, split):
    T = 200
    x = inputs(T, 3)
    full, h_full = run(kda, x, [T], final=True)
    o1, h1 = run(kda, cut(x, 0, split), [split], final=True)
    o2, h2 = run(kda, cut(x, split, T), [T - split], init=h1, final=True)
    assert torch.equal(torch.cat([o1, o2], 1), full)
    assert torch.equal(h2, h_full)


def test_conv_silu_last_matches_rows_on_gpu(kda):
    g = torch.Generator().manual_seed(4)
    C, W, L = 3 * H * K, 4, 33
    ext = torch.randn(W - 1 + L, C, generator=g).bfloat16().cuda()
    w = torch.randn(C, W, generator=g).cuda()
    rows = kda.conv_silu_rows(ext, w)
    last = kda.conv_silu_last(ext.unfold(0, W, 1).permute(0, 2, 1).contiguous(), w)
    assert torch.equal(last, rows[W - 1 :])
    check("kdabi_conv_silu", rows)


def test_prefix_case_that_needs_the_fla_fix(kda):
    # FLA's sub-chunk kernel takes its gate reference row from the rows left in the sequence; without the plugin's fix
    # this bf16 input (the model's per-rank shape) gives prefix length 132 other bits than the full sequence.
    T = 160
    g = torch.Generator().manual_seed(0)
    q, k, v, gate = (torch.randn(1, T, H, K, generator=g) for _ in range(4))
    beta = torch.randn(1, T, H, generator=g)
    A_log, dt_bias = torch.randn(H, generator=g) * 0.5, torch.randn(H * K, generator=g) * 0.5
    x = tuple(t.bfloat16().cuda() for t in (q, k, v, gate, beta)) + (A_log.cuda(), dt_bias.cuda())
    full, _ = run(kda, x, [T])
    lens = list(range(1, T + 1))
    out, _ = run(kda, tuple(torch.cat([t[:, :n] for n in lens], 1) if t.dim() >= 3 else t for t in x), lens)
    offs = [0] + list(torch.tensor(lens).cumsum(0).tolist())
    bad = [n for n, o in zip(lens, offs) if not torch.equal(out[0, o : o + n], full[0, :n])]
    assert not bad, f"prefix lengths with different bits: {bad}"
    check("kdabi_prefix_T160", full)
