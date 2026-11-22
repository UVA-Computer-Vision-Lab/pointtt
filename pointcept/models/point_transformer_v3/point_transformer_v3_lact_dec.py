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

The fast-weight update of TTTLayer (SwiGLU fast weights, Newton-Schulz orthogonalised
update) is adapted from LaCT (https://github.com/a1600012888/LaCT, Copyright (c) 2025
Tianyuan Zhang, Hao Tan, MIT License; see THIRD_PARTY_NOTICES.md).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from pointcept.models.builder import MODELS
from pointcept.models.utils.structure import Point
from pointcept.models.utils.serialization import hilbert_encode
from pointcept.models.modules import PointModule

from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import PointTransformerV3
from pointcept.models.point_transformer_v3.lact_layer import (
    l2_norm, inv_softplus, zeropower_via_newtonschulz5,
)


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
    """

    def __init__(self, dim, head_dim=64, inter_multi=1.0, base_lr=1e-2,
                 use_muon=True, grad_clip_val=None, gate_init=0.0):
        super().__init__()
        self.grad_clip_val = grad_clip_val
        self.dim = dim
        while dim % head_dim != 0 and head_dim > 1:
            head_dim -= 1
        self.head_dim = head_dim
        self.num_heads = dim // head_dim

        self.pre_norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, 3 * dim, bias=False)

        self.lr_dim = 1
        self.lr_proj = nn.Linear(dim, self.lr_dim * 3 * self.num_heads, bias=False)
        self.base_lr_inv = inv_softplus(base_lr)

        d_h = int(head_dim * inter_multi)

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

        self.use_muon = use_muon

        self.o_norm = nn.LayerNorm(head_dim)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        # Zero-init residual gate (Flamingo / LoRA-style). At init the TTT
        # contribution to `point.feat` is exactly 0, so plugging this layer on
        # top of a pretrained backbone does not perturb its features. Gradient
        # flows through the gate on the first step; once gate becomes non-zero,
        # gradient starts to reach the TTT internals too.
        # Forward uses tanh(gate) so the effective scale is bounded to [-1, 1]
        # and cannot blow up even if the raw parameter drifts large.
        # Output gate: the layer adds tanh(residual_gate) * output. Default 0.0.
        self.residual_gate = nn.Parameter(torch.full((1,), float(gate_init)))

        # Optional per-parameter gradient clamping for TTT internals. Each
        # backward call (one per chunk in chunk-wise training) passes each
        # parameter's grad through this element-wise clamp before it is
        # accumulated into `.grad`. Caps any single chunk's contribution so a
        # pathological sample cannot dominate the TTT update.
        if self.grad_clip_val is not None:
            clip_val = float(self.grad_clip_val)

            def _clamp(g):
                return g.clamp(-clip_val, clip_val)

            for n, p in self.named_parameters():
                if n != "residual_gate":
                    p.register_hook(_clamp)

    def reset_fast_weights(self):
        """Reset fast weight drift to zero (w_cur = w_init)."""
        self.w0_delta.zero_()
        self.w1_delta.zero_()
        self.w2_delta.zero_()

    def forward(self, point):
        """Process point features. Updates fast weights in-place."""
        feat = point.feat
        offset = point.offset
        starts = [0] + offset.cpu().tolist()
        device = feat.device
        H, Dk = self.num_heads, self.head_dim

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

            # w0: (H, d_h, Dk), q: (H, N, Dk)
            qi = ttt_q.transpose(1, 2)  # (H, Dk, N)
            h = torch.bmm(w2_cur, qi)  # (H, d_h, N)
            gate = F.silu(torch.bmm(w0_cur, qi))
            output_b = torch.bmm(w1_cur, gate * h)  # (H, Dk, N)
            output_b = output_b.transpose(1, 2)  # (H, N, Dk)
            output_b = self.o_norm(output_b)
            output_b = rearrange(output_b, "h n d -> n (h d)")
            outputs.append(self.o_proj(output_b))

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

                if self.use_muon:
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
        chunk_size=10240,
        lact_head_dim=64,
        lact_base_lr=1e-2,
        lact_use_muon=True,
        lact_enc_stages=None,   # which encoder stages get TTT, None = all
        lact_dec_stages=None,   # which decoder stages get TTT, None = all
        freeze_backbone=True,
        train_with_ttt=False,   # if False, TTT layers are skipped in both train
                                # and eval (pure chunk-wise PTv3).
        lact_grad_clip_val=None,  # per-chunk element-wise grad clamp on TTT
                                  # internal params. None = no extra clip.
        lact_inter_multi=1.0,     # SwiGLU intermediate dim multiplier for TTT.
                                  # >1 widens w0/w1/w2 internal hidden dim.
        lact_gate_init=0.0,       # initial value of each TTT layer's residual
                                  # gate. 0.0 = LoRA-style no-op at init;
                                  # >0 engages TTT from step 0 so its
                                  # internals receive gradient and co-adapt.
        chunk_order="3d",         # "3d": full 3D Hilbert over (x, y, z).
                                  # "xy": 2D Hilbert over (x, y) with z as
                                  #       tiebreak → chunks are vertical
                                  #       columns (prisms over the full z range).
        ttt_span=None,            # Eval-time TTT context length, in chunks.
                                  # None = full scene (default — TTT state
                                  # spans all chunks of one scene). Setting
                                  # e.g. 2 makes TTT reset every 2 chunks, so
                                  # cross-chunk context is capped at that span.
        **ptv3_kwargs,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.ttt_span = ttt_span
        self.train_with_ttt = train_with_ttt
        assert chunk_order in ("3d", "xy"), f"bad chunk_order: {chunk_order}"
        self.chunk_order = chunk_order

        self.ptv3 = PointTransformerV3(**ptv3_kwargs)

        enc_channels = ptv3_kwargs.get('enc_channels', (32, 64, 128, 256, 512))
        dec_channels = ptv3_kwargs.get('dec_channels', (64, 64, 128, 256))
        num_enc_stages = len(enc_channels)
        num_dec_stages = len(dec_channels)

        if lact_enc_stages is None:
            lact_enc_stages = list(range(num_enc_stages))
        if lact_dec_stages is None:
            lact_dec_stages = list(range(num_dec_stages))
        self.lact_enc_stages = lact_enc_stages
        self.lact_dec_stages = lact_dec_stages

        # Encoder TTT layers
        self.enc_ttt_layers = nn.ModuleDict()
        for s in lact_enc_stages:
            self.enc_ttt_layers[str(s)] = TTTLayer(
                dim=enc_channels[s], head_dim=lact_head_dim,
                base_lr=lact_base_lr, use_muon=lact_use_muon,
                grad_clip_val=lact_grad_clip_val,
                inter_multi=lact_inter_multi, gate_init=lact_gate_init)

        # Decoder TTT layers
        self.dec_ttt_layers = nn.ModuleDict()
        for s in lact_dec_stages:
            self.dec_ttt_layers[str(s)] = TTTLayer(
                dim=dec_channels[s], head_dim=lact_head_dim,
                base_lr=lact_base_lr, use_muon=lact_use_muon,
                grad_clip_val=lact_grad_clip_val,
                inter_multi=lact_inter_multi, gate_init=lact_gate_init)

        if freeze_backbone:
            for p in self.ptv3.parameters():
                p.requires_grad = False

    def _reset_all_ttt(self):
        for ttt in self.enc_ttt_layers.values():
            ttt.reset_fast_weights()
        for ttt in self.dec_ttt_layers.values():
            ttt.reset_fast_weights()

    def sort_points_for_chunking(self, grid_coord):
        """Return argsort indices that arrange points for chunk-wise forward.

        "3d"  — 3D Hilbert over (x, y, z). Adjacent chunk points are
                3D-spatially close.
        "xy"  — 2D Hilbert over (x, y) only, with z used only as an in-column
                ordering. Each chunk becomes a vertical prism, preserving
                trees that span the full z range.
        """
        if self.chunk_order == "3d":
            return hilbert_encode(grid_coord, depth=16).argsort()
        # "xy": zero out the z column to get 2D Hilbert (same code works
        # because the encoder interleaves bits — with z=0, only the x/y bits
        # contribute). Ties on xy are broken by the original z (stable sort).
        gc_xy = grid_coord.clone()
        gc_xy[:, 2] = 0
        primary = hilbert_encode(gc_xy, depth=16)  # int64 per point
        # stable argsort on primary preserves input order among ties; we want
        # z-monotonic order within a column, so sort by z first (stable), then
        # by primary — the latter dominates.
        z_sorted = grid_coord[:, 2].argsort(stable=True)
        primary_rearranged = primary[z_sorted]
        outer = primary_rearranged.argsort(stable=True)
        return z_sorted[outer]

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
        if not self.ptv3.cls_mode:
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

        use_ttt=False: per-chunk independent (no cross-chunk dependency).
        use_ttt=True : L→R no-grad pass evolves fast weights; then R→L per chunk
                       forward+backward, starting from L→R's end state (NO reset
                       between — matches eval's bidirectional behavior).
        """
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
            # Hilbert-based chunking; "3d" = 3D Hilbert, "xy" = vertical prisms.
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

        # Return per-chunk mean loss for logging (does NOT affect the gradients
        # that were already accumulated chunk-by-chunk above).
        return total_loss / max(total_nw, 1)

    def forward(self, data_dict):
        device = next(self.parameters()).device
        feat = data_dict["feat"]
        coord = data_dict["coord"]
        gc = data_dict["grid_coord"]
        offset = data_dict["offset"]
        N = feat.shape[0]

        starts = [0] + offset.cpu().tolist()
        all_feats = []

        for b in range(len(starts) - 1):
            s, e = int(starts[b]), int(starts[b + 1])
            Ns = e - s
            gc_b = gc[s:e]
            # Hilbert-based chunking; "3d" = 3D Hilbert, "xy" = vertical prisms.
            sorted_idx = self.sort_points_for_chunking(gc_b)
            inv_idx = torch.empty_like(sorted_idx)
            inv_idx[sorted_idx] = torch.arange(Ns, device=device)

            W = self.chunk_size
            chunk_ranges = [(j, min(j + W, Ns)) for j in range(0, Ns, W)]
            nw = len(chunk_ranges)

            def _make_chunk(cs, ce):
                idx = sorted_idx[cs:ce] + s
                d = dict(feat=feat[idx], coord=coord[idx], grid_coord=gc[idx],
                         offset=torch.tensor([ce - cs], device=device))
                if "segment" in data_dict:
                    d["segment"] = data_dict["segment"][idx]
                return d

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
                # Process chunks in non-overlapping windows of `span` chunks.
                # ttt_span=None → one window over all chunks (full-scene TTT).
                # ttt_span=K → TTT state resets every K chunks, capping the
                # cross-chunk context length.
                span = self.ttt_span if self.ttt_span is not None else nw
                span = max(1, min(span, nw))
                detach_fn = (lambda t: t.detach()) if not self.training else (lambda t: t)
                bwd_outputs = [None] * nw

                for win_start in range(0, nw, span):
                    win_end = min(win_start + span, nw)
                    # ---- L→R warm-up within window: no_grad, evolves drift. ----
                    self._reset_all_ttt()
                    with torch.no_grad():
                        for i in range(win_start, win_end):
                            cs, ce = chunk_ranges[i]
                            _ = self._run_one_chunk(_make_chunk(cs, ce), use_ttt=True)
                    # ---- R→L pred within window (NO reset; continues from L→R end). ----
                    for i in range(win_end - 1, win_start - 1, -1):
                        cs, ce = chunk_ranges[i]
                        chunk_point = self._run_one_chunk(_make_chunk(cs, ce), use_ttt=True)
                        bwd_outputs[i] = detach_fn(chunk_point.feat)

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
            keys_to_remap = [k for k in list(state_dict.keys())
                             if k.startswith(prefix) and not k.startswith(prefix + "ttt_")]
            for k in keys_to_remap:
                suffix = k[len(prefix):]
                if any(suffix.startswith(p) for p in ['enc.', 'dec.', 'embedding.']):
                    state_dict[ptv3_prefix + suffix] = state_dict.pop(k)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)
