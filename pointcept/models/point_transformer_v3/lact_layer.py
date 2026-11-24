"""
Helper functions of the LaCT fast-weight update, used by the TTT layers of
PT-v3-LaCT-Dec and PT-v3-LaCT-Temporal.

Copied from LaCT (https://github.com/a1600012888/LaCT),
minimal_implementations/causal_lact_with_sliding_window_attn.py,
Copyright (c) 2025 Tianyuan Zhang, Hao Tan, MIT License (see THIRD_PARTY_NOTICES.md).
Modified for PoinTTT: only l2_norm, zeropower_via_newtonschulz5 and inv_softplus (for a
Python float) are kept, without torch.compile.
"""

import math
import torch


def l2_norm(x: torch.Tensor):
    """
    x: [b, l, d]
    """
    x_type = x.dtype
    ret = x / (x.norm(dim=-1, keepdim=True) + 1e-5)  # norm will upcast to float32
    return ret.type(x_type)


def zeropower_via_newtonschulz5(G):
    """
    This is an updated version of the zeropower_via_newtonschulz5 function in here:
    https://github.com/KellerJordan/modded-nanogpt/blob/master/train_gpt_medium.py#L26
    The code is modified from https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py#L49, which contains the original muon implementation.
    Major change: G is [b, d, d] rather than [d, d]
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    Args:
        G: [b, d, d] (square: the fast weights are head_dim x head_dim)
    Returns:
        X: [b, d, d]
    FLOPS: 30 * b * d^3
    """
    assert len(G.shape) == 3
    X = G.bfloat16()
    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(1, 2), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for a, b, c in [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ]:
        A = X @ X.transpose(1, 2)
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    return X


def inv_softplus(x):
    return x + math.log(-math.expm1(-x))
