"""CPU tests of the lact_write_grad option of PT-v3-LaCT-Dec / PT-v3-LaCT-Temporal.

    python tests/test_write_grad.py          # or: python -m pytest tests/test_write_grad.py

1. float64: the gradient accumulated by grad_accum_forward with lact_write_grad=True equals an independent
   reference that runs the whole scene (or temporal mega-batch) in ONE autograd graph: the fast-weight state is a
   differentiable chain, every write takes its input features detached, the inner fast-weight gradient is taken
   by torch.autograd.grad(create_graph=True), and the sum of the chunk losses is backpropagated once.
2. float64 central finite differences for a single TTT layer on fixed inputs (all parameter tensors).
3. float32 (bf16 Newton-Schulz): lact_write_grad=True leaves chunk outputs, losses, fast-weight buffers, eval
   outputs, state_dict keys and every gradient outside the write path bitwise unchanged; K/V rows of to_qkv and
   lr_proj receive nonzero gradients; the write history is released after every scene.
4. lact_write_grad=True refuses to train under autocast (enable_amp=True); the default is unaffected.
Small random models on the CPU; needs spconv (CPU build is enough), no GPU.
"""
import contextlib
import os
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import pointcept.models  # noqa: E402,F401
from pointcept.models.builder import build_model  # noqa: E402
import pointcept.models.point_transformer_v3.point_transformer_v3_lact_dec as lact_dec  # noqa: E402
import pointcept.models.point_transformer_v3.point_transformer_v3_lact_temporal as lact_temporal  # noqa: E402

NS_COEF = [(4.0848, -6.8946, 2.9270), (3.9505, -6.3029, 2.6377), (3.7418, -5.5913, 2.3037),
           (2.8769, -3.1427, 1.2046), (2.8366, -3.0525, 1.2012)]


# ----------------------------------------------------------------------------------------- helpers
def ns_native(G):
    """Newton-Schulz in G's own dtype (the layer casts to bf16); used for the float64 checks only."""
    X = G.transpose(1, 2) if G.size(1) > G.size(2) else G
    X = X / (X.norm(dim=(1, 2), keepdim=True) + 1e-7)
    for a, b, c in NS_COEF:
        A = X @ X.transpose(1, 2)
        X = a * X + (b * A + c * A @ A) @ X
    return X.transpose(1, 2) if G.size(1) > G.size(2) else X


@contextlib.contextmanager
def float64_exact():
    """Newton-Schulz in float64 and lr.float() a no-op on float64 tensors (test only)."""
    saved = (lact_dec.zeropower_via_newtonschulz5, lact_temporal.zeropower_via_newtonschulz5, torch.Tensor.float)
    lact_dec.zeropower_via_newtonschulz5 = lact_temporal.zeropower_via_newtonschulz5 = ns_native
    to_float = saved[2]
    torch.Tensor.float = lambda t, *a, **k: t if t.dtype == torch.float64 else to_float(t, *a, **k)
    try:
        yield
    finally:
        lact_dec.zeropower_via_newtonschulz5, lact_temporal.zeropower_via_newtonschulz5, torch.Tensor.float = saved


def tiny(temporal=False, dtype=torch.float64, seed=0, **kw):
    cfg = dict(type="PT-v3-LaCT-Temporal" if temporal else "PT-v3-LaCT-Dec", chunk_size=96, lact_head_dim=8,
               lact_base_lr=1e-2, lact_use_muon=True, lact_enc_stages=[0, 1], lact_dec_stages=[0],
               freeze_backbone=False, train_with_ttt=True, lact_gate_init=0.5, in_channels=6,
               order=("z", "z-trans"), stride=(2,), enc_depths=(1, 1), enc_channels=(16, 32), enc_num_head=(2, 4),
               enc_patch_size=(32, 32), dec_depths=(1,), dec_channels=(16,), dec_num_head=(2,),
               dec_patch_size=(32,), drop_path=0.0, shuffle_orders=False, enable_flash=False)
    if temporal:
        cfg["max_batch_size_per_gpu"] = 4
    cfg.update(kw)
    torch.manual_seed(seed)
    model = build_model(cfg)
    import spconv.pytorch as spconv
    for m in model.modules():
        if isinstance(m, spconv.conv.SparseConvolution):
            m.algo = spconv.ConvAlgo.Native
        if hasattr(m, "shuffle_orders"):
            m.shuffle_orders = False
    head = nn.Linear(16, 3)
    return model.to(dtype), head.to(dtype)


def cpu_eval(model):
    """eval(), but spconv layers keep training=True (their CPU build cannot fuse the bias in eval)."""
    import spconv.pytorch as spconv
    model.eval()
    for m in model.modules():
        if isinstance(m, spconv.conv.SparseConvolution):
            m.train()


def scenes(sizes, dtype, seed=0):
    g = torch.Generator().manual_seed(seed)
    parts = []
    for n in sizes:
        sd = max(8, int(round((n * 3.5) ** (1 / 3))))
        cells = torch.randperm(sd ** 3, generator=g)[:n]
        gc = torch.stack([cells // (sd * sd), (cells // sd) % sd, cells % sd], 1).long()
        parts.append(((gc.to(dtype) + torch.rand(n, 3, generator=g, dtype=dtype)) * 0.05, gc,
                      torch.randn(n, 6, generator=g, dtype=dtype), (gc[:, 0] * 3 // sd + (gc[:, 2] > sd // 2)) % 3))
    return dict(coord=torch.cat([p[0] for p in parts]), grid_coord=torch.cat([p[1] for p in parts]),
                feat=torch.cat([p[2] for p in parts]), segment=torch.cat([p[3] for p in parts]),
                offset=torch.tensor(torch.tensor(sizes).cumsum(0).tolist()))


def sequences(n_clips, pts, dtype, seed=0):
    g = torch.Generator().manual_seed(seed)
    cols = {k: [] for k in ("coord", "grid_coord", "feat", "segment", "clip_idx")}
    sizes = []
    for n in pts:
        for c in range(n_clips):
            cells = torch.randperm(1000, generator=g)[:n]
            gc = torch.stack([cells // 100, (cells // 10) % 10, cells % 10], 1).long()
            cols["coord"].append((gc.to(dtype) + torch.rand(n, 3, generator=g, dtype=dtype)) * 0.05)
            cols["grid_coord"].append(gc)
            cols["feat"].append(torch.randn(n, 6, generator=g, dtype=dtype))
            cols["segment"].append((gc[:, 0] * 3 // 10 + c) % 3)
            cols["clip_idx"].append(torch.full((n,), c, dtype=torch.long))
        sizes.append(n * n_clips)
    data = {k: torch.cat(v) for k, v in cols.items()}
    data["offset"] = torch.tensor(torch.tensor(sizes).cumsum(0).tolist())
    return data


def layers(model):
    return list(model.enc_ttt_layers.values()) + list(model.dec_ttt_layers.values())


def grads(model, head):
    out = {f"m.{n}": p.grad for n, p in model.named_parameters()}
    out.update({f"h.{n}": p.grad for n, p in head.named_parameters()})
    return {k: (None if v is None else v.detach().clone()) for k, v in out.items()}


def train_step(model, head, data):
    model.train()
    for p in list(model.parameters()) + list(head.parameters()):
        p.grad = None
    model.grad_accum_forward(data, head, lambda logits, y: F.cross_entropy(logits, y))
    assert all(l.pending_write_reads == 0 and not l._write_inputs for l in layers(model))
    return grads(model, head)


# ----------------------------------------------------------------------------------------- reference
def _heads(t, H, Dk):
    return t.reshape(t.shape[0], H, Dk).transpose(0, 1)


def _fast_net(W0, W1, W2, z):
    a = torch.einsum("hij,hnj->hni", W0, z)
    b = torch.einsum("hij,hnj->hni", W2, z)
    return torch.einsum("hij,hnj->hni", W1, F.silu(a) * b)


def ref_read(l, x, S):
    D, H, Dk = l.dim, l.num_heads, l.head_dim
    W = [l.w0_init + S[0], l.w1_init + S[1], l.w2_init + S[2]]
    xn = F.layer_norm(x, (D,), l.pre_norm.weight, l.pre_norm.bias, l.pre_norm.eps)
    q = _heads(F.silu(xn @ l.to_qkv.weight.T)[:, :D], H, Dk)
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-5)
    o = F.layer_norm(_fast_net(*W, q), (Dk,), l.o_norm.weight, l.o_norm.bias, l.o_norm.eps)
    return o.transpose(0, 1).reshape(x.shape[0], D) @ l.o_proj.weight.T


def ref_write(l, x, S):
    D, H, Dk = l.dim, l.num_heads, l.head_dim
    xn = F.layer_norm(x.detach(), (D,), l.pre_norm.weight, l.pre_norm.bias, l.pre_norm.eps)
    qkv = F.silu(xn @ l.to_qkv.weight.T)
    k = _heads(qkv[:, D:2 * D], H, Dk)
    k = k / (k.norm(dim=-1, keepdim=True) + 1e-5)
    v = _heads(qkv[:, 2 * D:], H, Dk)
    lr = F.softplus(xn @ l.lr_proj.weight.T + l.base_lr_inv).reshape(-1, H, 3).permute(2, 1, 0)
    inits = (l.w0_init, l.w1_init, l.w2_init)
    W = [i + s for i, s in zip(inits, S)]
    new = []
    for r in range(3):
        g = torch.autograd.grad((lr[r].unsqueeze(-1) * _fast_net(*W, k) * v).sum(), W[r], create_graph=True)[0]
        g = ns_native(g) if l.use_muon else g
        Wn = W[r] + g
        new.append(Wn / (Wn.norm(dim=2, keepdim=True) + 1e-5) * W[r].norm(dim=2, keepdim=True) - inits[r])
    return tuple(new)


def _zero(l):
    return tuple(torch.zeros_like(p) for p in (l.w0_init, l.w1_init, l.w2_init))


def _ref_forward(l, point):
    starts = point.get("offset_list", None)
    if starts is None or len(starts) != len(point.offset) + 1:
        starts = [0] + point.offset.tolist()
    outs = []
    for b, (s, e) in enumerate(zip(starts[:-1], starts[1:])):
        x, S = point.feat[s:e], l._ref_S.get(b) or _zero(l)
        if getattr(l, "update_first", False):
            S = ref_write(l, x, S)
        outs.append(ref_read(l, x, S))
        if not getattr(l, "update_first", False):
            S = ref_write(l, x, S)
        l._ref_S[b] = S
    point.feat = point.feat + torch.tanh(l.residual_gate) * torch.cat(outs, 0)
    point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
    return point


def reference(model, head, data, temporal):
    model.train()
    for p in list(model.parameters()) + list(head.parameters()):
        p.grad = None
    for l in layers(model):
        l.forward = types.MethodType(_ref_forward, l)
    total = 0
    try:
        if temporal:
            offs = model._gather_batch_offsets(data)
            clips, cache = model._prepare_clip_index(offs, data["clip_idx"])
            for l in layers(model):
                l._ref_S = {}
            chunks = [model._build_mega_chunk(data, k, offs, data["clip_idx"], index_cache=cache)[0] for k in clips]
            for c in chunks:
                model._run_one_chunk(c)
            for c in reversed(chunks):
                total = total + F.cross_entropy(head(model._run_one_chunk(c).feat), c["segment"])
        else:
            starts = [0] + data["offset"].tolist()
            for s, e in zip(starts[:-1], starts[1:]):
                order = model.sort_points_for_chunking(data["grid_coord"][s:e]) + s
                chunks = [order[j:j + model.chunk_size] for j in range(0, e - s, model.chunk_size)]
                chunks = [dict(feat=data["feat"][i], coord=data["coord"][i], grid_coord=data["grid_coord"][i],
                               offset=torch.tensor([len(i)]), segment=data["segment"][i]) for i in chunks]
                for l in layers(model):
                    l._ref_S = {}
                for c in chunks:
                    model._run_one_chunk(c)
                for c in reversed(chunks):
                    total = total + F.cross_entropy(head(model._run_one_chunk(c).feat), c["segment"])
        total.backward()
    finally:
        for l in layers(model):
            del l.forward
            del l._ref_S
    return grads(model, head)


def max_rel_err(a, b):
    worst = 0.0
    for n in b:
        x = torch.zeros_like(b[n]) if a[n] is None and b[n] is not None else a[n]
        y = torch.zeros_like(a[n]) if b[n] is None and a[n] is not None else b[n]
        if x is None:
            continue
        den = max(float(x.norm()), float(y.norm()))
        worst = max(worst, float((x - y).norm()) / den if den > 0 else 0.0)
    return worst


# ----------------------------------------------------------------------------------------- tests
def test_matches_whole_scene_reference():
    with float64_exact():
        for temporal in (False, True):
            for kw in (dict(), dict(lact_use_muon=False), dict(lact_gate_init=0.0)):
                model, head = tiny(temporal=temporal, lact_write_grad=True, **kw)
                data = sequences(4, [40, 30], torch.float64) if temporal else scenes([300, 70], torch.float64)
                err = max_rel_err(train_step(model, head, data), reference(model, head, data, temporal))
                assert err < 1e-10, (temporal, kw, err)


def test_finite_differences_single_layer():
    class Sparse:
        def replace_feature(self, f):
            return self
    torch.manual_seed(3)
    layer = lact_dec.TTTLayer(16, head_dim=8, gate_init=0.5, write_grad=True).double()
    head = nn.Linear(16, 3).double()
    g = torch.Generator().manual_seed(9)
    xs = [torch.randn(n, 16, generator=g, dtype=torch.float64) for n in (7, 9, 6, 8)]
    ys = [torch.randint(3, (len(x),), generator=g) for x in xs]

    def pt(x):
        return types.SimpleNamespace(feat=x, offset=torch.tensor([len(x)]), sparse_conv_feat=Sparse())

    def scene(train):
        layer.train(train)
        layer.reset_fast_weights()
        with torch.no_grad():
            for x in xs:
                layer(pt(x))
        tot = 0.0
        for x, y in reversed(list(zip(xs, ys))):
            loss = F.cross_entropy(head(layer(pt(x)).feat), y)
            if train:
                loss.backward()
            tot += float(loss)
        if train:
            layer.flush_write_gradients()
        return tot

    with float64_exact():
        scene(True)
        gd = torch.Generator().manual_seed(1)
        for n, p in layer.named_parameters():
            u = torch.randn(p.shape, generator=gd, dtype=p.dtype)
            u /= u.norm()
            p0 = p.detach().clone()
            with torch.no_grad():
                p.copy_(p0 + 1e-6 * u)
                lp = scene(False)
                p.copy_(p0 - 1e-6 * u)
                lm = scene(False)
                p.copy_(p0)
            fd, ad = (lp - lm) / 2e-6, float((p.grad * u).sum())
            assert abs(ad - fd) <= 1e-5 * max(abs(fd), 1e-12), (n, ad, fd)


def test_forward_unchanged_and_write_gradients_present():
    for temporal in (False, True):
        base, head0 = tiny(temporal=temporal, dtype=torch.float32)
        new, head1 = tiny(temporal=temporal, dtype=torch.float32, lact_write_grad=True)
        assert list(base.state_dict()) == list(new.state_dict())
        new.load_state_dict(base.state_dict(), strict=True)
        head1.load_state_dict(head0.state_dict())
        data = sequences(3, [30, 20], torch.float32) if temporal else scenes([250, 70], torch.float32)
        rec = {0: [], 1: []}
        for i, m in enumerate((base, new)):
            for l in layers(m):
                l.register_forward_hook(lambda mod, inp, out, i=i: rec[i].append(
                    [getattr(mod, f"w{j}_delta").clone() for j in range(3)] + [out.feat.detach().clone()]))
        ga, gb = train_step(base, head0, data), train_step(new, head1, data)
        assert len(rec[0]) == len(rec[1]) and all(
            torch.equal(x, y) for r0, r1 in zip(rec[0], rec[1]) for x, y in zip(r0, r1))
        for n in ga:
            write = "ttt_layers" in n and n.endswith(("pre_norm.weight", "pre_norm.bias", "to_qkv.weight",
                                                      "lr_proj.weight", "w0_init", "w1_init", "w2_init"))
            if n.endswith("to_qkv.weight"):
                d = ga[n].shape[1]
                assert torch.equal(ga[n][:d], gb[n][:d]) and ga[n][d:].count_nonzero() == 0
                assert gb[n][d:2 * d].norm() > 0 and gb[n][2 * d:].norm() > 0
            elif n.endswith("lr_proj.weight"):
                assert ga[n] is None and gb[n] is not None and gb[n].norm() > 0
            elif not write:
                assert (ga[n] is None and gb[n] is None) or torch.equal(ga[n], gb[n]), n
        for m in (base, new):
            cpu_eval(m)
        with torch.no_grad():
            assert torch.equal(base(dict(data)).feat, new(dict(data)).feat)
        new.train()
        try:
            new(dict(data))
            raise AssertionError("forward() in training mode must require grad_accum_forward")
        except RuntimeError:
            pass


def test_autocast_refused():
    """Write-gradient training is fp32 only: under autocast (enable_amp=True) grad_accum_forward raises before any
    work, and so does recording a write in a TTT layer (every write path goes through it). The default is unaffected."""
    for temporal in (False, True):
        model, head = tiny(temporal=temporal, dtype=torch.float32, lact_write_grad=True)
        data = sequences(3, [30, 20], torch.float32) if temporal else scenes([250, 70], torch.float32)
        model.train()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            try:
                model.grad_accum_forward(data, head, lambda logits, y: F.cross_entropy(logits.float(), y))
                raise AssertionError("lact_write_grad under autocast must raise")
            except RuntimeError as e:
                assert "autocast" in str(e), e
        assert all(p.grad is None for p in list(model.parameters()) + list(head.parameters()))
        assert all(l.pending_write_reads == 0 and not l._write_inputs for l in layers(model))

    class Sparse:
        def replace_feature(self, f):
            return self
    x = torch.randn(12, 16)
    for write_grad in (True, False):
        torch.manual_seed(0)
        layer = lact_dec.TTTLayer(16, head_dim=8, gate_init=0.5, write_grad=write_grad).train()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            try:
                layer(types.SimpleNamespace(feat=x, offset=torch.tensor([12]), sparse_conv_feat=Sparse()))
                assert not write_grad, "recording a write under autocast must raise"
            except RuntimeError as e:
                assert write_grad and "autocast" in str(e), e
        assert not layer._write_inputs


if __name__ == "__main__":
    torch.set_num_threads(4)
    for t in (test_matches_whole_scene_reference, test_finite_differences_single_layer,
              test_forward_unchanged_and_write_gradients_present, test_autocast_refused):
        t()
        print("PASS", t.__name__)
