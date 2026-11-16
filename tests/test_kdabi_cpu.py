import pytest
import torch
import torch.nn.functional as F

import k31kdabi


@pytest.mark.parametrize("C,W,L", [(16, 4, 9), (256, 4, 1)])
def test_conv_silu_last_matches_the_last_row(C, W, L):
    # Bitwise on the GPU (tests/gpu); CPU SiLU rounds differently by vector length.
    g = torch.Generator().manual_seed(0)
    ext = torch.randn(W - 1 + L, C, generator=g).bfloat16()
    w = torch.randn(C, W, generator=g)
    rows = k31kdabi.conv_silu_rows(ext, w)
    windows = ext.unfold(0, W, 1).permute(0, 2, 1)  # [rows, W, C]
    last = k31kdabi.conv_silu_last(windows.contiguous(), w)
    torch.testing.assert_close(last, rows[W - 1 :], rtol=1e-6, atol=1e-6)


def test_conv_silu_rows_matches_a_causal_conv():
    g = torch.Generator().manual_seed(1)
    C, W, T = 8, 4, 12
    ext = torch.randn(W - 1 + T, C, generator=g).bfloat16()
    w = torch.randn(C, W, generator=g)
    ref = F.silu(F.conv1d(ext.float().T[None], w[:, None, :], groups=C)[0].T)
    torch.testing.assert_close(k31kdabi.conv_silu_rows(ext, w)[W - 1 :], ref, rtol=1e-5, atol=1e-5)


def test_gated_rms_norm():
    g = torch.Generator().manual_seed(2)
    x, gate = torch.randn(5, 3, 128, generator=g).bfloat16(), torch.randn(5, 3, 128, generator=g).bfloat16()
    w = torch.randn(128, generator=g).bfloat16()
    out = k31kdabi.gated_rms_norm(x, gate, w, 1e-6)
    ref = F.rms_norm(x.float(), (128,), eps=1e-6) * w.float() * torch.sigmoid(gate.float())
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
