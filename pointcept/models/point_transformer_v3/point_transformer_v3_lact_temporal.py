"""
PTv3-LaCT-Temporal: chunk by clip with PER-BATCH-SLOT TTT fast weights.

Key differences vs PT-v3-LaCT-Dec (spatial parent):
  - Chunks are temporal clips (sorted by `clip_idx`), not Hilbert spatial slices.
  - TTT layer (`BatchedTTTLayer`) keeps an independent fast-weight delta per
    batch slot, so multiple sequences in one GPU forward DON'T contaminate
    each other's TTT state.
  - ``grad_accum_forward`` outer-loops over clips and inner-batches all B
    sequences' clip-k points into one PTv3 forward — gives real intra-GPU
    parallelism instead of serial per-sequence iteration.

Use with ``StreamingSegmentorV2`` for the per-clip backward training loop.

The per-slot fast-weight update of BatchedTTTLayer is adapted from LaCT
(https://github.com/a1600012888/LaCT, Copyright (c) 2025 Tianyuan Zhang, Hao Tan,
MIT License; see THIRD_PARTY_NOTICES.md).

Author: Wentao
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from pointcept.models.builder import MODELS
from pointcept.models.utils.structure import Point

from .point_transformer_v3_lact_dec import (
    PointTransformerV3LaCTDec,
    TTTLayer as _BaseTTTLayer,
)
from .lact_layer import l2_norm, zeropower_via_newtonschulz5

class BatchedTTTLayer(_BaseTTTLayer):
    """TTT layer with one independent ``w*_delta`` slot per batch sample.

    The original lact_dec TTTLayer keeps a single shared delta buffer; if a
    forward sees offset of length>1 (multiple batch samples) it loops over
    them but writes/reads the SAME delta — slot 0's evolution leaks into
    slot 1. We allocate ``(max_batch_size, H, ...)`` deltas indexed by slot.

    Args:
        max_batch_size: pre-allocated number of independent delta slots.
            Must be ≥ batch_size_per_gpu used at runtime.
    """

    def __init__(self, *args, max_batch_size: int = 8, **kwargs):
        super().__init__(*args, **kwargs)
        H = self.num_heads
        Dk = self.head_dim
        d_h = self.w0_init.shape[1]

        self.max_batch_size = max_batch_size
        # When False, forward() computes outputs with the current fast weights
        # but SKIPS the delta update — used to freeze the adapted state during
        # eval output passes so every clip sees identical full-context weights.
        self.update_enabled = True
        # When True, forward() runs the causal recurrence per chunk:
        #   W_t = Update(W_{t-1}, H_t)  THEN  H~_t = Apply(W_t, H_t)
        # i.e. the delta update is applied BEFORE the output is computed, so
        # chunk t is predicted with state that includes its own update but
        # nothing from the future. Default False = apply-then-update.
        self.update_first = False
        # Replace the (H, ...) buffers from the base ctor with batched ones.
        self.register_buffer('w0_delta',
                              torch.zeros(max_batch_size, H, d_h, Dk),
                              persistent=False)
        self.register_buffer('w1_delta',
                              torch.zeros(max_batch_size, H, Dk, d_h),
                              persistent=False)
        self.register_buffer('w2_delta',
                              torch.zeros(max_batch_size, H, d_h, Dk),
                              persistent=False)

    def reset_fast_weights(self):
        self.w0_delta.zero_()
        self.w1_delta.zero_()
        self.w2_delta.zero_()

    def forward(self, point):
        feat = point.feat
        offset = point.offset
        # slot boundaries as Python ints: supplied by _build_mega_chunk ("offset_list") so that
        # no device-to-host copy is needed here; otherwise they are read from `offset`.
        starts = point.get("offset_list", None)
        if starts is None or len(starts) != len(offset) + 1:
            starts = [0] + offset.cpu().tolist()
        H, Dk = self.num_heads, self.head_dim

        outputs = []
        for b in range(len(starts) - 1):
            if b >= self.max_batch_size:
                raise RuntimeError(
                    f"BatchedTTTLayer: batch slot {b} exceeds "
                    f"max_batch_size={self.max_batch_size}; bump it in config."
                )
            s, e = int(starts[b]), int(starts[b + 1])
            x = feat[s:e]
            x_normed = self.pre_norm(x)

            ttt_qkv = F.silu(self.to_qkv(x_normed))
            ttt_q, ttt_k, ttt_v = rearrange(
                ttt_qkv, "n (qkv h d) -> qkv h n d", qkv=3, h=H, d=Dk)
            ttt_q = l2_norm(ttt_q.unsqueeze(0)).squeeze(0)
            ttt_k = l2_norm(ttt_k.unsqueeze(0)).squeeze(0)

            with torch.autocast(device_type="cuda", enabled=False):
                lr = self.lr_proj(x_normed)
            lr = F.softplus(lr.float() + self.base_lr_inv)
            lr0, lr1, lr2 = rearrange(
                lr, "n (h lrs d) -> lrs h n d", lrs=3, h=H, d=self.lr_dim)

            # Per-slot delta. Grad still flows to w_init through the addition.
            w0_cur = self.w0_init + self.w0_delta[b]
            w1_cur = self.w1_init + self.w1_delta[b]
            w2_cur = self.w2_init + self.w2_delta[b]

            def _delta_update(w0_d, w1_d, w2_d, b=b, ttt_k=ttt_k, ttt_v=ttt_v,
                              lr0=lr0, lr1=lr1, lr2=lr2):
                """Fast-weight update from this chunk's (k, v): writes delta[b].
                Update math always uses the PASSED (pre-update) weights, i.e.
                W_t = Update(W_{t-1}, H_t) regardless of call order."""
                with torch.no_grad():
                    ki_t = ttt_k.transpose(1, 2)
                    vi = ttt_v.transpose(1, 2)

                    gate_before_act = torch.bmm(w0_d, ki_t)
                    hidden_before_mul = torch.bmm(w2_d, ki_t)
                    hidden = F.silu(gate_before_act) * hidden_before_mul

                    dhidden = torch.bmm(w1_d.transpose(1, 2), vi)
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

                    w0_new = w0_d + dw0
                    w1_new = w1_d + dw1
                    w2_new = w2_d + dw2

                    w0_norm = w0_d.norm(dim=2, keepdim=True)
                    w1_norm = w1_d.norm(dim=2, keepdim=True)
                    w2_norm = w2_d.norm(dim=2, keepdim=True)

                    w0_new = w0_new / (w0_new.norm(dim=2, keepdim=True) + 1e-5) * w0_norm
                    w1_new = w1_new / (w1_new.norm(dim=2, keepdim=True) + 1e-5) * w1_norm
                    w2_new = w2_new / (w2_new.norm(dim=2, keepdim=True) + 1e-5) * w2_norm

                    self.w0_delta[b].copy_(w0_new - self.w0_init.detach())
                    self.w1_delta[b].copy_(w1_new - self.w1_init.detach())
                    self.w2_delta[b].copy_(w2_new - self.w2_init.detach())

            if self.update_first and self.update_enabled:
                # Causal online form: W_t = Update(W_{t-1}, H_t), then Apply
                # with W_t. Re-read the updated delta so the output below uses
                # state that includes this chunk (and nothing from the future).
                _delta_update(w0_cur.detach(), w1_cur.detach(), w2_cur.detach())
                w0_cur = self.w0_init + self.w0_delta[b]
                w1_cur = self.w1_init + self.w1_delta[b]
                w2_cur = self.w2_init + self.w2_delta[b]

            qi = ttt_q.transpose(1, 2)
            h = torch.bmm(w2_cur, qi)
            gate = F.silu(torch.bmm(w0_cur, qi))
            output_b = torch.bmm(w1_cur, gate * h).transpose(1, 2)
            output_b = self.o_norm(output_b)
            output_b = rearrange(output_b, "h n d -> n (h d)")
            outputs.append(self.o_proj(output_b))

            # Default order: apply first (above), then update slot b's delta.
            if not self.update_first and self.update_enabled:
                _delta_update(w0_cur.detach(), w1_cur.detach(), w2_cur.detach())

        out = torch.cat(outputs, dim=0)
        point.feat = point.feat + torch.tanh(self.residual_gate) * out
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point


@MODELS.register_module("PT-v3-LaCT-Temporal")
class PointTransformerV3LaCTTemporal(PointTransformerV3LaCTDec):
    """PTv3-LaCT for HOI4D sequences.

    Chunks = temporal clips (sorted by clip_idx).
    Per-batch-slot TTT delta → batch_size_per_gpu > 1 is correct AND faster
    (mega-chunk over all slots' clip-k points = one bigger PTv3 forward
    per clip instead of B serial smaller ones).
    """

    def __init__(self, max_batch_size_per_gpu: int = 8,
                 ttt_online_mode: bool = False,
                 ttt_online_view_avg: bool = False,
                 ttt_online_extra_views: int = 0, **kwargs):
        super().__init__(**kwargs)
        # Evaluation modes (TTT layers enabled, train_with_ttt=True):
        #   default: no-grad L→R warm-up pass over all clips of a sequence, then
        #     an R→L output pass that continues from the warm-up state.
        #   ttt_online_mode: causal streaming inference. One chronological pass
        #     over the chunks; per chunk t the fast weights are updated FIRST
        #     (W_t = Update(W_{t-1}, H_t)) and the prediction is computed with
        #     W_t. No warm-up pass, no access to future chunks; the state is
        #     reset only at sample (= video) start.
        self.ttt_online_mode = ttt_online_mode
        #   ttt_online_view_avg: predict each chunk twice — once with W_{t-1}
        #     (updates frozen) and once with W_t (update-then-apply) — and
        #     average the feature views. Both views are causal.
        self.ttt_online_view_avg = ttt_online_view_avg
        #   ttt_online_extra_views: N additional frozen-state forwards of the
        #     current chunk (each re-randomizes the serialization order),
        #     averaged in as extra views. Causal — same data, more views.
        self.ttt_online_extra_views = max(0, int(ttt_online_extra_views))

        # Replace the TTT layers built by the parent ctor with batched ones.
        H_dim = kwargs.get('lact_head_dim', 64)
        base_lr = kwargs.get('lact_base_lr', 1e-2)
        use_muon = kwargs.get('lact_use_muon', True)
        grad_clip = kwargs.get('lact_grad_clip_val', None)
        inter_multi = kwargs.get('lact_inter_multi', 1.0)
        gate_init = kwargs.get('lact_gate_init', 0.0)
        enc_channels = kwargs.get('enc_channels', (32, 64, 128, 256, 512))
        dec_channels = kwargs.get('dec_channels', (64, 64, 128, 256))

        new_enc = nn.ModuleDict()
        for s in self.lact_enc_stages:
            new_enc[str(s)] = BatchedTTTLayer(
                dim=enc_channels[s], head_dim=H_dim,
                base_lr=base_lr, use_muon=use_muon,
                grad_clip_val=grad_clip, inter_multi=inter_multi,
                gate_init=gate_init,
                max_batch_size=max_batch_size_per_gpu,
            )
        self.enc_ttt_layers = new_enc

        new_dec = nn.ModuleDict()
        for s in self.lact_dec_stages:
            new_dec[str(s)] = BatchedTTTLayer(
                dim=dec_channels[s], head_dim=H_dim,
                base_lr=base_lr, use_muon=use_muon,
                grad_clip_val=grad_clip, inter_multi=inter_multi,
                gate_init=gate_init,
                max_batch_size=max_batch_size_per_gpu,
            )
        self.dec_ttt_layers = new_dec

        self.max_batch_size_per_gpu = max_batch_size_per_gpu

    def _set_ttt_updates(self, enabled: bool):
        for layer in self.enc_ttt_layers.values():
            layer.update_enabled = enabled
        for layer in self.dec_ttt_layers.values():
            layer.update_enabled = enabled

    def _set_ttt_update_first(self, update_first: bool):
        for layer in self.enc_ttt_layers.values():
            layer.update_first = update_first
        for layer in self.dec_ttt_layers.values():
            layer.update_first = update_first

    # ---- mega-chunk helpers ----
    def _gather_batch_offsets(self, data_dict):
        offset = data_dict["offset"]
        starts = [0] + offset.cpu().tolist()
        return [(int(starts[b]), int(starts[b + 1])) for b in range(len(starts) - 1)]

    def _all_unique_clips(self, batch_offsets, clip_idx_full):
        all_clips = set()
        for s, e in batch_offsets:
            ci_b = clip_idx_full[s:e]
            all_clips.update(torch.unique(ci_b).cpu().tolist())
        return sorted(all_clips)

    def _prepare_clip_index(self, batch_offsets, clip_idx_full):
        """Per-slot point indices grouped by clip, computed once per sample batch.

        Replaces one ``nonzero()`` (a device-to-host synchronisation) per slot per chunk
        and per pass with one stable sort per slot per batch. For clip k of slot b the
        selected indices are the same, in the same (ascending) order, as
        ``(clip_idx[s:e] == k).nonzero()``, so the chunks are identical.
        Returns (unique_clips, cache) with cache["slots"][b] = (sorted global indices,
        [cumulative counts]) on the host and cache["counts"] / cache["offsets"] = (n_clip, B)
        tensors on the device, so that a chunk's ``offset`` is a view and no host->device copy
        (which would wait for the stream) is issued per chunk."""
        n_clip = int(clip_idx_full.max().item()) + 1 if clip_idx_full.numel() else 0
        slots, counts_dev = [], []
        present = torch.zeros(n_clip, dtype=torch.bool)
        for s, e in batch_offsets:
            ci_b = clip_idx_full[s:e]
            _, order = torch.sort(ci_b, stable=True)
            counts = torch.bincount(ci_b, minlength=n_clip)
            counts_dev.append(counts)
            counts_host = counts.cpu()
            present |= counts_host > 0
            slots.append((order + s, [0] + counts_host.cumsum(0).tolist()))
        counts_dev = torch.stack(counts_dev, 1)                       # (n_clip, B)
        cache = dict(slots=slots, counts=counts_dev, offsets=counts_dev.cumsum(1))
        unique_clips = present.nonzero(as_tuple=True)[0].tolist()
        return unique_clips, cache

    def _build_mega_chunk(self, data_dict, k, batch_offsets, clip_idx_full,
                          want_segment=True, index_cache=None):
        """Concat clip-k points from every batch slot. ``offset`` segments
        keep them separate so TTT/PTv3 know batch boundaries."""
        device = data_dict["feat"].device
        keys = ["feat", "coord", "grid_coord"]
        if want_segment and "segment" in data_dict:
            keys.append("segment")
        per_slot_idx = []
        per_slot_sizes = []

        for b, (s, e) in enumerate(batch_offsets):
            if index_cache is not None:
                order, cum = index_cache["slots"][b]
                global_idx = order[cum[k]:cum[k + 1]]
            else:
                ci_b = clip_idx_full[s:e]
                local = (ci_b == k).nonzero(as_tuple=True)[0]
                global_idx = local + s
            per_slot_idx.append(global_idx)
            per_slot_sizes.append(int(global_idx.shape[0]))

        # one gather per key instead of one per key and slot (same rows, same order)
        all_idx = per_slot_idx[0] if len(per_slot_idx) == 1 else torch.cat(per_slot_idx, 0)
        chunk = {key: data_dict[key][all_idx] for key in keys}
        offset_list = [0]
        for n in per_slot_sizes:
            offset_list.append(offset_list[-1] + n)
        if index_cache is not None:
            # views of per-batch tensors: no host->device copy per chunk
            chunk["offset"] = index_cache["offsets"][k]
            repeats = index_cache["counts"][k]
        else:
            chunk["offset"] = torch.tensor(offset_list[1:], device=device)
            repeats = torch.tensor(per_slot_sizes, device=device)
        # slot boundaries on the host (BatchedTTTLayer / SerializedPooling read them instead of
        # copying ``offset`` back) and the per-point slot id, so that Point() does not have to
        # derive it with a synchronising repeat_interleave.
        chunk["offset_list"] = offset_list
        chunk["batch"] = torch.repeat_interleave(
            torch.arange(len(per_slot_sizes), device=device, dtype=torch.long),
            repeats, output_size=offset_list[-1],
        )
        return chunk, per_slot_idx

    # ---- training entry point ----
    def grad_accum_forward(self, data_dict, seg_head, criteria):
        if "clip_idx" not in data_dict:
            raise KeyError("PT-v3-LaCT-Temporal needs per-point `clip_idx`.")
        clip_idx_full = data_dict["clip_idx"]
        batch_offsets = self._gather_batch_offsets(data_dict)
        B = len(batch_offsets)
        if B > self.max_batch_size_per_gpu:
            raise RuntimeError(
                f"batch_size_per_gpu={B} > max_batch_size_per_gpu="
                f"{self.max_batch_size_per_gpu}; bump max_batch_size_per_gpu."
            )

        unique_clips, index_cache = self._prepare_clip_index(batch_offsets, clip_idx_full)
        nw = len(unique_clips)
        # the per-chunk losses are summed on the device; one .item() per sample batch
        # instead of one per chunk (each .item() is a full device synchronisation)
        total_loss = torch.zeros((), dtype=torch.float64, device=clip_idx_full.device)

        if self.train_with_ttt:
            # L→R warm-up under no_grad over all clips → evolves delta.
            self._reset_all_ttt()
            with torch.no_grad():
                for k in unique_clips:
                    chunk, _ = self._build_mega_chunk(
                        data_dict, k, batch_offsets, clip_idx_full,
                        index_cache=index_cache)
                    _ = self._run_one_chunk(chunk, use_ttt=True)
            iter_order = list(reversed(unique_clips))
        else:
            iter_order = list(unique_clips)

        # R→L (or single-pass) per-clip forward + backward.
        for k in iter_order:
            chunk, _ = self._build_mega_chunk(
                data_dict, k, batch_offsets, clip_idx_full, index_cache=index_cache)
            chunk_point = self._run_one_chunk(chunk, use_ttt=self.train_with_ttt)
            logits = seg_head(chunk_point.feat)
            loss_c = criteria(logits, chunk["segment"])
            loss_c.backward()
            total_loss += loss_c.detach().double()
            del chunk_point, logits, loss_c, chunk

        return total_loss.item() / max(nw, 1)

    # ---- eval entry point ----
    def forward(self, data_dict):
        if "clip_idx" not in data_dict:
            raise KeyError("PT-v3-LaCT-Temporal needs per-point `clip_idx`.")
        clip_idx_full = data_dict["clip_idx"]
        feat_in = data_dict["feat"]
        device = feat_in.device
        N = feat_in.shape[0]

        batch_offsets = self._gather_batch_offsets(data_dict)
        unique_clips, index_cache = self._prepare_clip_index(batch_offsets, clip_idx_full)
        out_feats = None

        def _scatter(out_feats_, feat_k_, per_slot_idx):
            pos = 0
            for idx_t in per_slot_idx:
                n_b = idx_t.shape[0]
                out_feats_[idx_t] = feat_k_[pos:pos + n_b]
                pos += n_b

        if not self.train_with_ttt:
            for k in unique_clips:
                chunk, per_slot_idx = self._build_mega_chunk(
                    data_dict, k, batch_offsets, clip_idx_full,
                    want_segment=False, index_cache=index_cache)
                chunk_point = self._run_one_chunk(chunk, use_ttt=False)
                if out_feats is None:
                    out_feats = torch.empty(
                        (N, chunk_point.feat.shape[1]),
                        dtype=chunk_point.feat.dtype, device=device)
                _scatter(out_feats, chunk_point.feat, per_slot_idx)
        elif self.ttt_online_mode:
            # Causal streaming inference: single chronological pass; chunk t
            # is predicted with W_t = Update(W_{t-1}, H_t). State is reset at
            # sample start (= video start), never within the video, and no
            # future chunk is ever touched.
            detach_fn = (lambda t: t.detach()) if not self.training else (lambda t: t)
            self._reset_all_ttt()
            self._set_ttt_update_first(True)
            try:
                for k in unique_clips:
                    chunk, per_slot_idx = self._build_mega_chunk(
                        data_dict, k, batch_offsets, clip_idx_full,
                        want_segment=False, index_cache=index_cache)
                    feat_prev = None
                    if self.ttt_online_view_avg:
                        # View 1: predict chunk t with W_{t-1} (updates off).
                        # Causal — uses only past state + current input.
                        self._set_ttt_updates(False)
                        self._set_ttt_update_first(False)
                        cp_prev = self._run_one_chunk(chunk, use_ttt=True)
                        feat_prev = detach_fn(cp_prev.feat)
                        self._set_ttt_updates(True)
                        self._set_ttt_update_first(True)
                    # View 2 (or only view): update-then-apply with W_t.
                    chunk_point = self._run_one_chunk(chunk, use_ttt=True)
                    feat_k = detach_fn(chunk_point.feat)
                    views_sum, n_views = feat_k, 1
                    if feat_prev is not None:
                        views_sum = views_sum + feat_prev
                        n_views += 1
                    if self.ttt_online_extra_views > 0:
                        # Extra frozen-state forwards of the SAME chunk — each
                        # re-randomizes serialization order (shuffle_orders).
                        self._set_ttt_updates(False)
                        self._set_ttt_update_first(False)
                        for _ in range(self.ttt_online_extra_views):
                            cp_x = self._run_one_chunk(chunk, use_ttt=True)
                            views_sum = views_sum + detach_fn(cp_x.feat)
                            n_views += 1
                        self._set_ttt_updates(True)
                        self._set_ttt_update_first(True)
                    feat_k = views_sum / n_views
                    if out_feats is None:
                        out_feats = torch.empty(
                            (N, feat_k.shape[1]),
                            dtype=feat_k.dtype, device=device)
                    _scatter(out_feats, feat_k, per_slot_idx)
            finally:
                self._set_ttt_update_first(False)
        else:
            # L→R warm-up under no_grad over all clips (evolves the fast
            # weights), then the R→L output pass continuing from that state.
            detach_fn = (lambda t: t.detach()) if not self.training else (lambda t: t)
            self._reset_all_ttt()
            with torch.no_grad():
                for k in unique_clips:
                    chunk, _ = self._build_mega_chunk(
                        data_dict, k, batch_offsets, clip_idx_full,
                        want_segment=False, index_cache=index_cache)
                    _ = self._run_one_chunk(chunk, use_ttt=True)
            for k in reversed(unique_clips):
                chunk, per_slot_idx = self._build_mega_chunk(
                    data_dict, k, batch_offsets, clip_idx_full,
                    want_segment=False, index_cache=index_cache)
                chunk_point = self._run_one_chunk(chunk, use_ttt=True)
                feat_k = detach_fn(chunk_point.feat)
                if out_feats is None:
                    out_feats = torch.empty(
                        (N, feat_k.shape[1]),
                        dtype=feat_k.dtype, device=device)
                _scatter(out_feats, feat_k, per_slot_idx)

        if out_feats is None:
            out_feats = feat_in

        point = Point(data_dict)
        point.feat = out_feats
        return point
