"""Kimi-K3.1 BI DSA kernels on one GPU: the deterministic indexer top-k and the batch-invariant MQA logits."""

import pytest
import torch

from .golden import check

pytestmark = pytest.mark.gpu

TOPK = 2048  # Kimi-K3.1 index_topk
H, D = 32, 128  # index_num_heads, index_head_dim


@pytest.fixture(scope="module")
def ix():
    return pytest.importorskip("vllm_rl_plugins.models.kimi_k31.rocm.dsa.indexer")


def ref_topk(logits, starts, ends, k):
    out = torch.full((logits.shape[0], k), -1, dtype=torch.int32)
    for r in range(logits.shape[0]):
        s, e = int(starts[r]), int(ends[r])
        seg = logits[r, s:e].double().cpu()
        cols = torch.arange(e - s)
        # by score, ties to the larger column
        order = sorted(range(e - s), key=lambda c: (-seg[c].item(), -c))[:k]
        sel = torch.sort(cols[order]).values
        out[r, : sel.numel()] = sel.to(torch.int32)
    return out


def run_topk(ix, logits, starts, ends):
    out = torch.empty(logits.shape[0], TOPK, dtype=torch.int32, device="cuda")
    ix.dsv4_det_topk_per_row(logits, starts, ends, out, logits.shape[0], TOPK)
    return out


def topk_case(seed):
    g = torch.Generator().manual_seed(seed)
    n = 9000
    # few distinct values: many ties at the cutoff; no -0.0 or NaN
    logits = (torch.randint(0, 600, (7, n), generator=g).float() / 8 - 30).cuda()
    # rows: full, in-register (<= 4096 cols), shorter than top-k, exactly top-k, empty, streamed with offset, one column
    starts = torch.tensor([0, 100, 0, 3000, 50, 1500, 8999], dtype=torch.int32)
    ends = torch.tensor([9000, 4000, 1500, 3000 + TOPK, 50, 9000, 9000], dtype=torch.int32)
    return logits, starts.cuda(), ends.cuda()


@pytest.mark.parametrize("seed", [0, 1])
def test_topk_matches_reference(ix, seed):
    logits, starts, ends = topk_case(seed)
    got = run_topk(ix, logits, starts, ends).cpu()
    assert torch.equal(got, ref_topk(logits, starts, ends, TOPK))
    check(f"dsabi_topk_{seed}", got)


def test_topk_row_depends_only_on_its_row(ix):
    logits, starts, ends = topk_case(0)
    full = run_topk(ix, logits, starts, ends)
    for r in range(logits.shape[0]):
        alone = run_topk(ix, logits[r : r + 1].clone(), starts[r : r + 1], ends[r : r + 1])
        assert torch.equal(alone[0], full[r]), r
    perm = torch.tensor([6, 2, 0, 5, 1, 4, 3], device="cuda")
    assert torch.equal(run_topk(ix, logits[perm], starts[perm], ends[perm]), full[perm])


def mqa_case(seed, n=3000):
    from vllm.platforms import current_platform

    g = torch.Generator().manual_seed(seed)
    fp8 = current_platform.fp8_dtype()
    q = (torch.randn(6, H, D, generator=g) * 2).to(fp8).cuda()
    k = (torch.randn(n, D, generator=g) * 2).to(fp8).cuda()
    k_scale = (torch.rand(n, generator=g) * 0.1 + 0.01).cuda()
    w = (torch.randn(6, H, generator=g) * 0.1).cuda()
    ks = torch.tensor([0, 0, 500, 1000, 0, 2999], dtype=torch.int32).cuda()
    ke = torch.tensor([1, 700, 2900, 3000, 1700, 3000], dtype=torch.int32).cuda()
    return q, k, k_scale, w, ks, ke


def valid(logits, ks, ke):
    return [logits[r, int(ks[r]) : int(ke[r])] for r in range(logits.shape[0])]


def ref_logits(q, k, k_scale, w):
    # fp64 from the same fp8 values: sum_h w[h] * relu(q[h] . k * k_scale)
    score = torch.einsum("mhd,nd->mhn", q.double(), k.double()) * k_scale.double()
    return (score.relu() * w.double()[:, :, None]).sum(1)


def test_mqa_logits_match_reference(ix):
    q, k, s, w, ks, ke = mqa_case(0)
    got = ix._bi_fp8_mqa_logits(q, (k, s), w, ks, ke)
    ref = ref_logits(q, k, s, w)
    for a, b in zip(valid(got, ks, ke), valid(ref, ks, ke)):
        torch.testing.assert_close(a.double(), b, rtol=1e-4, atol=1e-4)
    check("dsabi_mqa_logits_0", *valid(got, ks, ke))


def test_mqa_logits_row_depends_only_on_its_row(ix):
    q, k, s, w, ks, ke = mqa_case(1)
    full = valid(ix._bi_fp8_mqa_logits(q, (k, s), w, ks, ke), ks, ke)
    for r in range(q.shape[0]):
        alone = ix._bi_fp8_mqa_logits(q[r : r + 1], (k, s), w[r : r + 1], ks[r : r + 1], ke[r : r + 1])
        assert torch.equal(valid(alone, ks[r : r + 1], ke[r : r + 1])[0], full[r]), r
    # more keys after every row's range (a longer cache) and the rows reversed
    q2, k2, s2, w2, _, _ = mqa_case(1, n=7000)
    k2[:3000], s2[:3000] = k, s
    rev = torch.arange(q.shape[0] - 1, -1, -1, device="cuda")
    out = valid(ix._bi_fp8_mqa_logits(q[rev], (k2, s2), w[rev], ks[rev], ke[rev]), ks[rev], ke[rev])
    for i, r in enumerate(rev.tolist()):
        assert torch.equal(out[i], full[r]), r
