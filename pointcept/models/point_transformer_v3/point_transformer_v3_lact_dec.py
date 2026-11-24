"""
PTv3-LaCT: Chunk-and-run PTv3 with TTT (Test-Time Training) layers.

Each scene is Hilbert-sorted and split into chunks of `chunk_size` points.
TTT layers live after selected encoder/decoder stages. Their fast weights
are stored as (init + delta) where init is a trainable nn.Parameter and
delta is a detached running buffer that tracks the fast-weight drift.

Train / Eval (identical structure):
  1. Reset delta to 0 (w_cur = w_init)
  2. L→R warm-up: no-grad forward over all chunks, evolves delta
  3. R→L prediction: per-chunk forward (with grad in train), continues
     from L→R's end state (no reset between). In training each chunk
     does its own backward; in eval R→L outputs are the final features.

Gradients of the fast-weight writes (`lact_write_grad`):
  By default (False) every write runs under torch.no_grad() and delta is a
  detached buffer: the task loss reaches a TTT layer only through its reads.
  The K/V rows of `to_qkv` and `lr_proj` therefore receive no gradient
  (stop-gradient through the memory update). With lact_write_grad=True each
  LaCT layer's parameters are also differentiated through all of that layer's
  writes in the scene, in training only. The features entering the writes stay constants (no
  gradient reaches the backbone or upstream LaCT layers through a write).
  Forward values, fast-weight states and evaluation are bitwise identical to
  False. See TTTLayer for the mechanism and its memory cost. Training with
  lact_write_grad requires fp32 (enable_amp=False); under autocast it raises.

The fast-weight update of TTTLayer (SwiGLU fast weights, Newton-Schulz orthogonalised
update) is adapted from LaCT (https://github.com/a1600012888/LaCT, Copyright (c) 2025
Tianyuan Zhang, Hao Tan, MIT License; see THIRD_PARTY_NOTICES.md).
"""

import itertools
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint

from pointcept.models.builder import MODELS
from pointcept.models.utils.structure import Point
from pointcept.models.utils.serialization import hilbert_encode
from pointcept.models.modules import PointModule

from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import PointTransformerV3
from pointcept.models.point_transformer_v3.lact_layer import (
    l2_norm, inv_softplus, zeropower_via_newtonschulz5,
)


_WRITE_GRAD_AMP_ERROR = (
    "lact_write_grad cannot be trained under autocast (enable_amp=True); set "
    "enable_amp=False. Autocast caches the low-precision copy of to_qkv.weight "
    "made in the no-grad L->R pass, so the recorded writes would lose their "
    "path to to_qkv (with write checkpointing the flush fails with a "
    "CheckpointError, without it the K/V write gradient is silently dropped).")


def _autocast_enabled():
    """True inside an active CUDA or CPU autocast region."""
    return torch.is_autocast_enabled("cuda") or torch.is_autocast_enabled("cpu")


def validate_write_grad(value):
    """Check a write-gradient mode: False or True (see TTTLayer)."""
    if value is False or value is True:
        return value
    raise ValueError(f"lact_write_grad must be False or True, got {value!r}")


class TTTLayer(nn.Module):
    """Test-Time Training layer using LaCT's SwiGLU fast weight update.

    Fast weights are stored as (w_init + w_delta):
      - w_init: trainable nn.Parameter (the "slow" initial weight)
      - w_delta: detached buffer tracking drift from init across chunks
    Forward always reads `w_init + w_delta` so gradient flows to w_init
    on every chunk, while delta is updated in-place under no_grad.

    forward(x) does:
      1. Inference: output = w1_cur @ (silu(w0_cur @ q) * (w2_cur @ q))
      2. Update:   compute Δw from (k, v), add to delta buffer, renormalize.

    Call reset_fast_weights() to zero the delta (w_cur = w_init).

    write_grad selects what the task loss differentiates through the update:
      False  (default) the update runs under no_grad, so the K/V rows of
             to_qkv and lr_proj receive no gradient, pre_norm learns only
             through q, and w*_init only through the read (straight-through).
      True   in training mode every update is also recorded as an isolated
             autograd graph of this layer's write parameters (pre_norm,
             to_qkv, lr_proj, w0/w1/w2_init) and of the previous state. The
             update input x is detached, so the backbone and upstream layers
             get no gradient through it. Each grad-enabled read uses fresh
             leaf copies of the current state, so a chunk's ordinary
             backward stops at them and frees its activations as before;
             flush_write_gradients() then sends the collected state adjoints
             through the whole write history once. Outputs, the delta
             buffers and eval are bitwise those of False (the recorded update
             runs the same ops on the same inputs).
    With checkpoint_write (default) only each update's input x and the
    states are kept and the update is recomputed in the flush; memory grows
    with the number of updates in a scene (x is N x dim per update).
    Training callers must call flush_write_gradients() after the last chunk
    backward of a scene, before the next reset or optimizer step
    (PointTransformerV3LaCTDec.grad_accum_forward does this). Recording a
    write raises under autocast: train write-gradient runs in fp32.
    """

    def __init__(self, dim, head_dim=64, base_lr=1e-2,
                 write_grad=False, checkpoint_write=True):
        super().__init__()
        self.write_grad = validate_write_grad(write_grad)
        self.checkpoint_write = bool(checkpoint_write)
        # Transient autograd state of the write history: plain attributes,
        # never part of the state_dict.
        self._write_state = None         # current state deltas (graph tensors)
        self._write_state_live = False   # False: a detached copy, no write recorded yet
        self._write_reads = []           # (state, read leaves, live) per grad-enabled read
        self._write_inputs = []          # (detached write input, its version counter)
        self.head_dim = head_dim
        self.num_heads = dim // head_dim

        self.pre_norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, 3 * dim, bias=False)

        self.lr_dim = 1
        self.lr_proj = nn.Linear(dim, self.lr_dim * 3 * self.num_heads, bias=False)
        self.base_lr_inv = inv_softplus(base_lr)

        d_h = head_dim

        # Initial (learned) "slow" weights — trainable params
        self.w0_init = nn.Parameter(torch.randn(self.num_heads, d_h, head_dim) / (head_dim ** 0.5))
        self.w1_init = nn.Parameter(torch.randn(self.num_heads, head_dim, d_h) / (d_h ** 0.5))
        self.w2_init = nn.Parameter(torch.randn(self.num_heads, d_h, head_dim) / (head_dim ** 0.5))

        # Running drift from init. Forward always reads (init + delta) so grad flows
        # to w_init on EVERY chunk, not just the first one after reset. delta itself
        # is a detached buffer, updated in-place under no_grad.
        self.register_buffer('w0_delta', torch.zeros_like(self.w0_init.data))
        self.register_buffer('w1_delta', torch.zeros_like(self.w1_init.data))
        self.register_buffer('w2_delta', torch.zeros_like(self.w2_init.data))

        self.o_norm = nn.LayerNorm(head_dim)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        # Zero-init residual gate (Flamingo / LoRA-style). At init the TTT
        # contribution to `point.feat` is exactly 0, so plugging this layer on
        # top of a pretrained backbone does not perturb its features. Gradient
        # flows through the gate on the first step; once gate becomes non-zero,
        # gradient starts to reach the TTT internals too.
        # Forward uses tanh(gate) so the effective scale is bounded to [-1, 1]
        # and cannot blow up even if the raw parameter drifts large.
        self.residual_gate = nn.Parameter(torch.zeros(1))

    def reset_fast_weights(self):
        """Reset fast weight drift to zero (w_cur = w_init)."""
        self.w0_delta.zero_()
        self.w1_delta.zero_()
        self.w2_delta.zero_()
        self.clear_write_graph()

    # ---- differentiable write history (write_grad) ----
    def clear_write_graph(self):
        """Release write history and read adjoints without changing memory values."""
        self._write_state = None
        self._write_state_live = False
        self._write_reads.clear()
        self._write_inputs.clear()

    def _write_params(self):
        """Parameters entering an update, in _write_graph_fn's argument order."""
        return (self.pre_norm.weight, self.pre_norm.bias, self.to_qkv.weight,
                self.lr_proj.weight, self.w0_init, self.w1_init, self.w2_init)

    def _records_writes(self):
        """Whether this forward records its updates in the write history."""
        return self.write_grad and self.training and any(
            p.requires_grad for p in self._write_params())

    def _write_features(self, x, ln_w, ln_b, qkv_w, lr_w):
        """k, v and per-token learning rates of one update (ops of forward())."""
        H, Dk = self.num_heads, self.head_dim
        x_normed = F.layer_norm(x, self.pre_norm.normalized_shape, ln_w, ln_b,
                                self.pre_norm.eps)
        ttt_qkv = F.silu(F.linear(x_normed, qkv_w))
        _, ttt_k, ttt_v = rearrange(
            ttt_qkv, "n (qkv h d) -> qkv h n d", qkv=3, h=H, d=Dk)
        ttt_k = l2_norm(ttt_k.unsqueeze(0)).squeeze(0)
        with torch.autocast(device_type="cuda", enabled=False):
            lr = F.linear(x_normed, lr_w)
        lr = F.softplus(lr.float() + self.base_lr_inv)
        lr0, lr1, lr2 = rearrange(
            lr, "n (h lrs d) -> lrs h n d", lrs=3, h=H, d=self.lr_dim)
        return ttt_k, ttt_v, lr0, lr1, lr2

    def _fast_weight_grads(self, w0, w1, w2, ttt_k, ttt_v, lr0, lr1, lr2):
        """Raw (not orthogonalised) fast-weight update of one write (ops of forward())."""
        ki_t = ttt_k.transpose(1, 2)  # (H, Dk, N)
        vi = ttt_v.transpose(1, 2)  # (H, Dk, N)
        gate_before_act = torch.bmm(w0, ki_t)
        hidden_before_mul = torch.bmm(w2, ki_t)
        hidden = F.silu(gate_before_act) * hidden_before_mul

        dhidden = torch.bmm(w1.transpose(1, 2), vi)
        dhidden_before_mul = dhidden * F.silu(gate_before_act)
        dgate = dhidden * hidden_before_mul
        sigma = torch.sigmoid(gate_before_act)
        dgate_before_act = dgate * sigma * (1 + gate_before_act * (1 - sigma))

        dw1 = torch.bmm(vi, (hidden.transpose(1, 2) * lr1).type_as(vi))
        dw0 = torch.bmm(dgate_before_act, (ttt_k * lr0).type_as(dgate_before_act))
        dw2 = torch.bmm(dhidden_before_mul, (ttt_k * lr2).type_as(dhidden_before_mul))
        return dw0, dw1, dw2

    def _write_graph_fn(self, x, ln_w, ln_b, qkv_w, lr_w, w0_init, w1_init, w2_init,
                        delta0, delta1, delta2):
        """One update as a pure function of its inputs; returns the new deltas.

        Same ops, order and GEMM shapes (the full packed to_qkv projection) as
        the no_grad update in forward(), so its values are bitwise those of
        the default update; differentiable in every argument.
        """
        ttt_k, ttt_v, lr0, lr1, lr2 = self._write_features(x, ln_w, ln_b, qkv_w, lr_w)
        w0_for_update = w0_init + delta0
        w1_for_update = w1_init + delta1
        w2_for_update = w2_init + delta2
        dw0, dw1, dw2 = self._fast_weight_grads(
            w0_for_update, w1_for_update, w2_for_update, ttt_k, ttt_v, lr0, lr1, lr2)
        dw0 = zeropower_via_newtonschulz5(dw0)
        dw1 = zeropower_via_newtonschulz5(dw1)
        dw2 = zeropower_via_newtonschulz5(dw2)

        w0_new = w0_for_update + dw0
        w1_new = w1_for_update + dw1
        w2_new = w2_for_update + dw2
        w0_norm = w0_for_update.norm(dim=2, keepdim=True)
        w1_norm = w1_for_update.norm(dim=2, keepdim=True)
        w2_norm = w2_for_update.norm(dim=2, keepdim=True)
        w0_new = w0_new / (w0_new.norm(dim=2, keepdim=True) + 1e-5) * w0_norm
        w1_new = w1_new / (w1_new.norm(dim=2, keepdim=True) + 1e-5) * w1_norm
        w2_new = w2_new / (w2_new.norm(dim=2, keepdim=True) + 1e-5) * w2_norm
        return w0_new - w0_init, w1_new - w1_init, w2_new - w2_init

    def _write_input(self, x):
        """Detached update input, remembered with its version for the flush check."""
        x = x.detach()
        self._write_inputs.append((x, x._version))
        return x

    def _run_write(self, fn, *args):
        """Run a pure update function with autograd on (also inside a no_grad
        warm-up: only this isolated graph is kept, never the backbone's)."""
        if _autocast_enabled():
            self.clear_write_graph()   # the scene's history is unusable; leave none behind
            raise RuntimeError(_WRITE_GRAD_AMP_ERROR)
        with torch.enable_grad():
            if self.checkpoint_write:
                return checkpoint(fn, *args, use_reentrant=False,
                                  preserve_rng_state=False)
            return fn(*args)

    def _current_write_state(self):
        if self._write_state is None:
            self._write_state = tuple(
                delta.detach().clone()
                for delta in (self.w0_delta, self.w1_delta, self.w2_delta))
            self._write_state_live = False
        return self._write_state, self._write_state_live

    def _read_leaves(self):
        """Fresh leaves holding the current state for one grad-enabled read."""
        state, live = self._current_write_state()
        leaves = tuple(d.detach().clone().requires_grad_(True) for d in state)
        self._write_reads.append((state, leaves, live))
        return leaves

    def _record_write(self, x):
        """Recorded replacement of the no_grad update: same values in w*_delta."""
        state, _ = self._current_write_state()
        deltas = self._run_write(
            self._write_graph_fn, self._write_input(x), *self._write_params(), *state)
        self._write_state, self._write_state_live = deltas, True
        with torch.no_grad():
            for buffer, delta in zip((self.w0_delta, self.w1_delta, self.w2_delta), deltas):
                buffer.copy_(delta.detach())

    def flush_write_gradients(self):
        """Propagate the recorded read adjoints through the write history once.

        Call after every prediction loss of the scene has run backward, before
        the next reset and the optimizer step. Shared history nodes are
        traversed once. The history is released in all cases.
        """
        try:
            roots, adjoints = {}, {}
            for state, leaves, live in self._write_reads:
                if not live:
                    continue  # state before any recorded update: no write parameter in it
                for delta, leaf in zip(state, leaves):
                    if leaf.grad is None:
                        continue  # this read did not reach a loss
                    if delta.grad_fn is None:
                        raise RuntimeError(
                            "TTTLayer write history has no autograd graph: an update "
                            "was recorded outside torch.enable_grad()")
                    key = id(delta)
                    if key in roots:
                        adjoints[key] = adjoints[key] + leaf.grad
                    else:
                        roots[key], adjoints[key] = delta, leaf.grad
            for x, version in self._write_inputs:
                if x._version != version:
                    raise RuntimeError(
                        "a TTTLayer update input was modified in place after the "
                        "update; its write-history gradient would be wrong")
            if roots:
                torch.autograd.backward(
                    list(roots.values()), grad_tensors=[adjoints[k] for k in roots])
        finally:
            self.clear_write_graph()

    def forward(self, point):
        """Process point features. Updates fast weights in-place."""
        feat = point.feat
        offset = point.offset
        starts = [0] + offset.cpu().tolist()
        H, Dk = self.num_heads, self.head_dim
        record = self._records_writes()

        outputs = []
        for b in range(len(starts) - 1):
            s, e = int(starts[b]), int(starts[b + 1])
            x = feat[s:e]  # (N, D)
            x_normed = self.pre_norm(x)

            # Project to q, k, v
            ttt_qkv = F.silu(self.to_qkv(x_normed))
            ttt_q, ttt_k, ttt_v = rearrange(
                ttt_qkv, "n (qkv h d) -> qkv h n d",
                qkv=3, h=H, d=Dk)
            ttt_q, ttt_k = l2_norm(ttt_q.unsqueeze(0)).squeeze(0), l2_norm(ttt_k.unsqueeze(0)).squeeze(0)
            # ttt_q: (H, N, Dk), ttt_k: (H, N, Dk), ttt_v: (H, N, Dk)

            with torch.autocast(device_type="cuda", enabled=False):
                lr = self.lr_proj(x_normed)
            lr = F.softplus(lr.float() + self.base_lr_inv)
            lr0, lr1, lr2 = rearrange(
                lr, "n (h lrs d) -> lrs h n d", lrs=3, h=H, d=self.lr_dim)

            # ---- Inference with current fast weights ----
            # w_cur = w_init + w_delta. Grad flows through w_init on EVERY chunk.
            # w_delta is a detached running offset tracking the fast update drift.
            w0_cur = self.w0_init + self.w0_delta
            w1_cur = self.w1_init + self.w1_delta
            w2_cur = self.w2_init + self.w2_delta
            if record and torch.is_grad_enabled():
                # Same values; the leaves collect this read's state adjoint
                # for flush_write_gradients().
                read_leaves = self._read_leaves()
                w0_cur = self.w0_init + read_leaves[0]
                w1_cur = self.w1_init + read_leaves[1]
                w2_cur = self.w2_init + read_leaves[2]

            # w0: (H, d_h, Dk), q: (H, N, Dk)
            qi = ttt_q.transpose(1, 2)  # (H, Dk, N)
            h = torch.bmm(w2_cur, qi)  # (H, d_h, N)
            gate = F.silu(torch.bmm(w0_cur, qi))
            output_b = torch.bmm(w1_cur, gate * h)  # (H, Dk, N)
            output_b = output_b.transpose(1, 2)  # (H, N, Dk)
            output_b = self.o_norm(output_b)
            output_b = rearrange(output_b, "h n d -> n (h d)")
            outputs.append(self.o_proj(output_b))

            if record:
                self._record_write(x)
                continue

            # ---- Update fast weights using k, v ----
            with torch.no_grad():
                ki_t = ttt_k.transpose(1, 2)  # (H, Dk, N)
                vi = ttt_v.transpose(1, 2)  # (H, Dk, N)

                # Use current weights (either init or buffer) for the update
                w0_for_update = w0_cur.detach()
                w1_for_update = w1_cur.detach()
                w2_for_update = w2_cur.detach()

                gate_before_act = torch.bmm(w0_for_update, ki_t)
                hidden_before_mul = torch.bmm(w2_for_update, ki_t)
                hidden = F.silu(gate_before_act) * hidden_before_mul

                dhidden = torch.bmm(w1_for_update.transpose(1, 2), vi)
                dhidden_before_mul = dhidden * F.silu(gate_before_act)
                dgate = dhidden * hidden_before_mul
                sigma = torch.sigmoid(gate_before_act)
                dgate_before_act = dgate * sigma * (1 + gate_before_act * (1 - sigma))

                dw1 = torch.bmm(vi, (hidden.transpose(1, 2) * lr1).type_as(vi))
                dw0 = torch.bmm(dgate_before_act, (ttt_k * lr0).type_as(dgate_before_act))
                dw2 = torch.bmm(dhidden_before_mul, (ttt_k * lr2).type_as(dhidden_before_mul))

                dw0 = zeropower_via_newtonschulz5(dw0)
                dw1 = zeropower_via_newtonschulz5(dw1)
                dw2 = zeropower_via_newtonschulz5(dw2)

                w0_new = w0_for_update + dw0
                w1_new = w1_for_update + dw1
                w2_new = w2_for_update + dw2

                w0_norm = w0_for_update.norm(dim=2, keepdim=True)
                w1_norm = w1_for_update.norm(dim=2, keepdim=True)
                w2_norm = w2_for_update.norm(dim=2, keepdim=True)

                w0_new = w0_new / (w0_new.norm(dim=2, keepdim=True) + 1e-5) * w0_norm
                w1_new = w1_new / (w1_new.norm(dim=2, keepdim=True) + 1e-5) * w1_norm
                w2_new = w2_new / (w2_new.norm(dim=2, keepdim=True) + 1e-5) * w2_norm

                # Store as drift from init (w_delta = w_new - w_init). Forward
                # will reconstruct via (w_init + w_delta), flowing grad to w_init.
                self.w0_delta.copy_(w0_new - self.w0_init.detach())
                self.w1_delta.copy_(w1_new - self.w1_init.detach())
                self.w2_delta.copy_(w2_new - self.w2_init.detach())

        out = torch.cat(outputs, dim=0)
        # tanh-wrapped gate keeps the effective scale in [-1, 1] regardless of
        # how large the underlying gate parameter drifts.
        point.feat = point.feat + torch.tanh(self.residual_gate) * out
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point


@MODELS.register_module("PT-v3-LaCT-Dec")
class PointTransformerV3LaCTDec(PointModule):
    """PTv3 with TTT layers after selected encoder/decoder stages.

    Chunk-and-run: scene Hilbert-sorted, split into `chunk_size` chunks.
    Each chunk runs through PTv3(+TTT). TTT fast weight drift evolves
    across chunks via L→R warm-up then R→L prediction (no reset between).

    When `train_with_ttt=False`, TTT layers are skipped entirely in both
    train and eval — the model is pure chunk-wise PTv3.
    """

    def __init__(
        self,
        lact_enc_stages,        # encoder stages followed by a TTT layer
        lact_dec_stages,        # decoder stages followed by a TTT layer
        chunk_size=10240,
        lact_head_dim=64,
        lact_base_lr=1e-2,
        train_with_ttt=False,   # if False, TTT layers are skipped in both train
                                # and eval (pure chunk-wise PTv3).
        lact_write_grad=False,    # Training only. False = fast-weight writes
                                  # under stop-gradient (K/V rows of to_qkv and
                                  # lr_proj get no gradient). True =
                                  # differentiate each LaCT layer's parameters
                                  # through all of its writes in the scene;
                                  # the write inputs (backbone
                                  # features) stay constants. Forward and eval
                                  # are identical for both values. True trains
                                  # in fp32 only (enable_amp=False).
        lact_write_grad_checkpoint=True,  # keep only each write's input and
                                  # recompute the write in the flush.
        **ptv3_kwargs,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.train_with_ttt = train_with_ttt
        self.lact_write_grad = validate_write_grad(lact_write_grad)
        self.lact_write_grad_checkpoint = bool(lact_write_grad_checkpoint)
        if self.lact_write_grad and not train_with_ttt:
            warnings.warn("lact_write_grad has no effect with train_with_ttt=False")

        self.ptv3 = PointTransformerV3(**ptv3_kwargs)

        enc_channels = ptv3_kwargs.get('enc_channels', (32, 64, 128, 256, 512))
        dec_channels = ptv3_kwargs.get('dec_channels', (64, 64, 128, 256))
        self.lact_enc_stages = lact_enc_stages
        self.lact_dec_stages = lact_dec_stages

        # Encoder TTT layers
        self.enc_ttt_layers = nn.ModuleDict()
        for s in lact_enc_stages:
            self.enc_ttt_layers[str(s)] = TTTLayer(
                dim=enc_channels[s], head_dim=lact_head_dim,
                base_lr=lact_base_lr,
                write_grad=self.lact_write_grad,
                checkpoint_write=self.lact_write_grad_checkpoint)

        # Decoder TTT layers
        self.dec_ttt_layers = nn.ModuleDict()
        for s in lact_dec_stages:
            self.dec_ttt_layers[str(s)] = TTTLayer(
                dim=dec_channels[s], head_dim=lact_head_dim,
                base_lr=lact_base_lr,
                write_grad=self.lact_write_grad,
                checkpoint_write=self.lact_write_grad_checkpoint)

    def _reset_all_ttt(self):
        for ttt in self.enc_ttt_layers.values():
            ttt.reset_fast_weights()
        for ttt in self.dec_ttt_layers.values():
            ttt.reset_fast_weights()

    def _flush_all_write_gradients(self):
        for ttt in itertools.chain(
            self.enc_ttt_layers.values(), self.dec_ttt_layers.values()
        ):
            ttt.flush_write_gradients()

    def _clear_all_write_graphs(self):
        for ttt in itertools.chain(
            self.enc_ttt_layers.values(), self.dec_ttt_layers.values()
        ):
            ttt.clear_write_graph()

    def _check_write_grad_forward(self):
        """Training with lact_write_grad must go through grad_accum_forward."""
        if self.training and self.train_with_ttt and self.lact_write_grad:
            raise RuntimeError(
                "lact_write_grad training requires grad_accum_forward "
                "(StreamingSegmentorV2) so that each scene's write history is "
                "flushed before the optimizer step")

    def sort_points_for_chunking(self, grid_coord):
        """Argsort of the 3D Hilbert code: the point order that is cut into chunks."""
        return hilbert_encode(grid_coord, depth=16).argsort()

    def _run_one_chunk(self, chunk_data, use_ttt=True):
        """Run one chunk through PTv3 encoder+decoder, optionally with TTT layers."""
        # Force BN to eval mode during every chunked forward. Two reasons:
        #   1) A chunk can collapse to 1 voxel at deep encoder stages after
        #      stride-2 downsampling → BN batch-stat computation crashes.
        #   2) With nw chunks per iter, per-chunk BN running stats update ~nw
        #      times per iter → running stats drift wildly. Freezing them here
        #      keeps training/eval behavior consistent (both use frozen stats).
        bn_switched = {}
        for name, m in self.ptv3.named_modules():
            if isinstance(m, nn.BatchNorm1d) and m.training:
                bn_switched[name] = True
                m.eval()

        point = Point(chunk_data)
        point.serialization(order=self.ptv3.order, shuffle_orders=self.ptv3.shuffle_orders)
        point.sparsify()
        point = self.ptv3.embedding(point)

        # Encoder with optional TTT after each stage
        for s_name, enc_stage in self.ptv3.enc.named_children():
            point = enc_stage(point)
            s_idx = int(s_name.replace("enc", ""))
            if use_ttt and s_idx in self.lact_enc_stages:
                point = self.enc_ttt_layers[str(s_idx)](point)

        # Decoder with optional TTT after each stage
        for s_name, dec_stage in self.ptv3.dec.named_children():
            point = dec_stage(point)
            s_idx = int(s_name.replace("dec", ""))
            if use_ttt and s_idx in self.lact_dec_stages:
                point = self.dec_ttt_layers[str(s_idx)](point)

        for name, m in self.ptv3.named_modules():
            if name in bn_switched:
                m.train()

        return point

    def grad_accum_forward(self, data_dict, seg_head, criteria):
        """Chunk-wise forward + backward for memory-efficient training.

        Called by StreamingSegmentorV2 in training mode. Each chunk runs its
        own forward → seg_head → loss → backward; activations are freed after
        each backward so peak memory is O(chunk_size), not O(total_points).
        With lact_write_grad the scene's isolated write history is kept as
        well (checkpointing stores write inputs and recomputes the writes)
        and flushed into .grad after the scene's last chunk backward, so the
        caller's optimizer step and any gradient sync see the full gradient.
        lact_write_grad training must run in fp32: it raises under autocast.

        use_ttt=False: per-chunk independent (no cross-chunk dependency).
        use_ttt=True : L→R no-grad pass evolves fast weights; then R→L per chunk
                       forward+backward, starting from L→R's end state (NO reset
                       between — matches eval's bidirectional behavior).
        """
        if (self.lact_write_grad and self.train_with_ttt and self.training
                and _autocast_enabled()):
            raise RuntimeError(_WRITE_GRAD_AMP_ERROR)
        try:
            return self._grad_accum_forward(data_dict, seg_head, criteria)
        except BaseException:
            # Release a partially recorded write history (lact_write_grad).
            self._clear_all_write_graphs()
            raise

    def _grad_accum_forward(self, data_dict, seg_head, criteria):
        device = next(self.parameters()).device
        feat = data_dict["feat"]
        coord = data_dict["coord"]
        gc = data_dict["grid_coord"]
        offset = data_dict["offset"]
        seg = data_dict["segment"]
        W = self.chunk_size

        starts = [0] + offset.cpu().tolist()
        total_loss = 0.0
        total_nw = 0

        for b in range(len(starts) - 1):
            s, e = int(starts[b]), int(starts[b + 1])
            Ns = e - s
            gc_b = gc[s:e]
            sorted_idx = self.sort_points_for_chunking(gc_b)

            chunk_ranges = [(j, min(j + W, Ns)) for j in range(0, Ns, W)]
            nw = len(chunk_ranges)
            total_nw += nw

            def _make_chunk(cs, ce):
                idx = sorted_idx[cs:ce] + s
                return dict(
                    feat=feat[idx],
                    coord=coord[idx],
                    grid_coord=gc[idx],
                    offset=torch.tensor([ce - cs], device=device),
                    segment=seg[idx],
                )

            if self.train_with_ttt:
                # L→R no-grad pass: evolves fast weight drift buffers.
                self._reset_all_ttt()
                with torch.no_grad():
                    for i in range(nw):
                        cs, ce = chunk_ranges[i]
                        _ = self._run_one_chunk(_make_chunk(cs, ce), use_ttt=True)

                # R→L training pass — NO reset: starts from L→R's end state.
                iter_order = range(nw - 1, -1, -1)
            else:
                iter_order = range(nw)

            for i in iter_order:
                cs, ce = chunk_ranges[i]
                cdata = _make_chunk(cs, ce)
                chunk_point = self._run_one_chunk(cdata, use_ttt=self.train_with_ttt)
                logits = seg_head(chunk_point.feat)
                loss_c = criteria(logits, cdata["segment"])
                loss_c.backward()
                total_loss += loss_c.detach().item()
                del chunk_point, logits, loss_c, cdata

            if self.lact_write_grad and self.train_with_ttt:
                # All of this scene's reads have run backward: send their state
                # adjoints through its write history before the next reset.
                self._flush_all_write_gradients()

        # Return per-chunk mean loss for logging (does NOT affect the gradients
        # that were already accumulated chunk-by-chunk above).
        return total_loss / max(total_nw, 1)

    def forward(self, data_dict):
        self._check_write_grad_forward()
        device = next(self.parameters()).device
        feat = data_dict["feat"]
        coord = data_dict["coord"]
        gc = data_dict["grid_coord"]
        offset = data_dict["offset"]

        starts = [0] + offset.cpu().tolist()
        all_feats = []

        for b in range(len(starts) - 1):
            s, e = int(starts[b]), int(starts[b + 1])
            Ns = e - s
            gc_b = gc[s:e]
            sorted_idx = self.sort_points_for_chunking(gc_b)
            inv_idx = torch.empty_like(sorted_idx)
            inv_idx[sorted_idx] = torch.arange(Ns, device=device)

            W = self.chunk_size
            chunk_ranges = [(j, min(j + W, Ns)) for j in range(0, Ns, W)]
            nw = len(chunk_ranges)

            def _make_chunk(cs, ce):
                idx = sorted_idx[cs:ce] + s
                return dict(feat=feat[idx], coord=coord[idx], grid_coord=gc[idx],
                            offset=torch.tensor([ce - cs], device=device))

            # TTT is used iff train_with_ttt flag is set — eval follows training
            # mode so non-TTT models aren't polluted by untrained TTT layers.
            use_ttt = self.train_with_ttt

            if not use_ttt:
                # Pure chunk-wise PTv3 — no cross-chunk state, no L→R/R→L.
                combined = []
                for i in range(nw):
                    cs, ce = chunk_ranges[i]
                    chunk_point = self._run_one_chunk(_make_chunk(cs, ce), use_ttt=False)
                    combined.append(chunk_point.feat)
                sorted_feat = torch.cat(combined, dim=0)
                all_feats.append(sorted_feat[inv_idx])
            else:
                bwd_outputs = [None] * nw
                # ---- L→R warm-up over all chunks: no_grad, evolves drift. ----
                self._reset_all_ttt()
                with torch.no_grad():
                    for i in range(nw):
                        cs, ce = chunk_ranges[i]
                        _ = self._run_one_chunk(_make_chunk(cs, ce), use_ttt=True)
                # ---- R→L prediction (NO reset; continues from L→R end). ----
                for i in range(nw - 1, -1, -1):
                    cs, ce = chunk_ranges[i]
                    chunk_point = self._run_one_chunk(_make_chunk(cs, ce), use_ttt=True)
                    bwd_outputs[i] = chunk_point.feat.detach()

                sorted_feat = torch.cat(bwd_outputs, dim=0)
                all_feats.append(sorted_feat[inv_idx])

        point = Point(data_dict)
        point.feat = torch.cat(all_feats, dim=0)
        return point

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        ptv3_prefix = prefix + "ptv3."
        has_ptv3 = any(k.startswith(ptv3_prefix) for k in state_dict)
        if not has_ptv3:
            keys_to_remap = [k for k in list(state_dict.keys()) if k.startswith(prefix)]
            for k in keys_to_remap:
                suffix = k[len(prefix):]
                if any(suffix.startswith(p) for p in ['enc.', 'dec.', 'embedding.']):
                    state_dict[ptv3_prefix + suffix] = state_dict.pop(k)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)
