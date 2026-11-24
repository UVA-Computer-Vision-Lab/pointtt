#!/usr/bin/env python
"""Environment check for the PoinTTT code (run by scripts/setup_env.sh at the end).

    PYTHONNOUSERSITE=1 python scripts/smoke_test_env.py --expect-torch 2.5.0               # full check, needs a GPU
    PYTHONNOUSERSITE=1 python scripts/smoke_test_env.py --expect-torch 2.5.0 --no-forward  # no GPU needed

Exit code 0 = all checks passed.
"""
import argparse
import importlib
import os
import sys
import tempfile
import time
import traceback
import warnings

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
RESULTS = []  # (status, name, detail)
EVAL_CONFIGS = ("configs/forainetv2/pointttt-forainetv2-eval.py", "configs/hoi4d/pointttt-hoi4d-eval.py")


def all_configs():
    """Every config under configs/ except the _base_ files, as paths relative to the repository."""
    out = []
    for d in sorted(os.listdir(os.path.join(REPO, "configs"))):
        if d.startswith("_") or not os.path.isdir(os.path.join(REPO, "configs", d)):
            continue
        out += [f"configs/{d}/{f}" for f in sorted(os.listdir(os.path.join(REPO, "configs", d))) if f.endswith(".py")]
    return out


def check(name):
    def deco(fn):
        def run(*a, **k):
            t = time.time()
            try:
                detail = fn(*a, **k)
                RESULTS.append(("PASS", name, f"{detail or ''} ({time.time() - t:.1f}s)"))
            except Exception as e:  # noqa: BLE001
                RESULTS.append(("FAIL", name, f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=6)}"))
            st, nm, det = RESULTS[-1]
            print(f"[{st}] {nm}: {det.splitlines()[0] if det else ''}", flush=True)
            if st == "FAIL":
                print("        " + "\n        ".join(det.splitlines()[1:]), flush=True)
        return run
    return deco


def synthetic_cloud(n_per_sample, n_cls, grid=0.05, extent=6.0, seed=0, device="cuda"):
    """Random voxelised point cloud in Pointcept's dict format."""
    import torch
    g = torch.Generator().manual_seed(seed)
    coords, gcs, offs, tot = [], [], [], 0
    for n in n_per_sample:
        c = torch.rand(n * 2, 3, generator=g) * extent
        gc = torch.unique(torch.floor(c / grid).long(), dim=0)[:n]
        coords.append((gc.float() + 0.5) * grid); gcs.append(gc); tot += gc.shape[0]; offs.append(tot)
    coord = torch.cat(coords)
    d = dict(coord=coord, grid_coord=torch.cat(gcs), feat=coord.clone(),
             segment=torch.randint(0, n_cls, (coord.shape[0],), generator=g), offset=torch.tensor(offs))
    return {k: v.to(device) for k, v in d.items()}


@check("python 3.11, user site-packages disabled")
def t_python():
    assert sys.version_info[:2] == (3, 11), f"python {sys.version.split()[0]} (the environment uses 3.11)"
    import site
    assert not site.ENABLE_USER_SITE, "user site-packages enabled -> export PYTHONNOUSERSITE=1"
    return sys.version.split()[0]


@check("torch build")
def t_torch(expect, need_gpu):
    import torch
    v = torch.__version__
    assert v.split("+")[0] == expect, f"torch {v}, expected {expect}"
    assert "+cu" in v, f"torch {v} is not a CUDA build"
    msg = f"torch {v} | cuda {torch.version.cuda}"
    if need_gpu:
        assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
        caps = sorted({torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())})
        msg += f" | {torch.cuda.device_count()} GPU(s) {torch.cuda.get_device_name(0)} | capabilities {caps}"
    return msg


@check("python packages")
def t_packages():
    import numpy as np
    assert int(np.__version__.split(".")[0]) == 2, f"numpy {np.__version__} (the environment uses 2.x)"
    out = [f"numpy {np.__version__}"]
    for m in ("h5py", "plyfile", "addict", "yapf", "timm", "einops", "spconv", "torch_scatter", "flash_attn",
              "pointops"):
        mod = importlib.import_module(m)
        out.append(f"{m} {getattr(mod, '__version__', '')}".strip())
    from timm.layers import DropPath  # noqa: F401
    from yapf.yapflib.yapf_api import FormatCode  # noqa: F401
    return ", ".join(out)


@check("pointcept import chain, registries, all configs")
def t_pointcept():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from pointcept.engines.defaults import default_config_parser
        from pointcept.engines.train import TRAINERS
        from pointcept.engines.hooks.builder import HOOKS
        from pointcept.models.builder import MODELS
        from pointcept.datasets.builder import DATASETS
    need = {TRAINERS: ["DefaultTrainer"],
            HOOKS: ["CheckpointLoader", "SemSegEvaluator", "IterationTimer", "InformationWriter", "CheckpointSaver"],
            MODELS: ["PT-v3m1", "PT-v3-LaCT-Dec", "PT-v3-LaCT-Temporal", "StreamingSegmentorV2", "DefaultSegmentorV2"],
            DATASETS: ["ForAINetV2Dataset", "HOI4DDataset"]}
    miss = [k for reg, keys in need.items() for k in keys if k not in reg.module_dict]
    assert not miss, f"not registered: {miss}"
    done = []
    for rel in all_configs():
        with tempfile.TemporaryDirectory(prefix="smoke_cfg_") as td:
            cfg = default_config_parser(os.path.join(REPO, rel), dict(save_path=td))
        hooks = [h["type"] for h in cfg.hooks if h["type"] not in HOOKS.module_dict]
        assert not hooks, f"{rel}: hooks not registered: {hooks}"
        assert cfg.model.backbone.type in MODELS.module_dict, f"{rel}: unknown backbone"
        done.append(f"{os.path.basename(rel)[:-3]}")
    return f"{len(done)} configs parse: " + ", ".join(done)


@check("spconv sparse convolution on GPU (default algorithm and Native)")
def t_spconv():
    import torch
    import spconv.pytorch as sp
    gc = torch.unique(torch.randint(0, 40, (4000, 3)), dim=0)
    idx = torch.cat([torch.zeros(len(gc), 1, dtype=torch.long), gc], 1).int().cuda()
    x = sp.SparseConvTensor(torch.randn(len(gc), 16).cuda(), idx, [40, 40, 40], 1)
    outs = []
    for algo in (None, sp.ConvAlgo.Native):
        torch.manual_seed(0)
        conv = sp.SubMConv3d(16, 32, 3, padding=1, bias=False, indice_key="k", algo=algo).cuda()
        outs.append(conv(x).features)
    d = (outs[0] - outs[1]).abs().max().item()
    assert d < 1e-3, f"default vs Native differ by {d}"
    return f"max |default - Native| = {d:.1e}"


@check("torch_scatter, flash_attn and pointops on GPU")
def t_cuda_ops():
    import torch
    import torch_scatter
    import flash_attn
    import pointops
    out = torch_scatter.segment_csr(torch.randn(10, 4).cuda(), torch.tensor([0, 3, 10]).cuda(), reduce="max")
    assert out.shape == (2, 4)
    n, h, c = 3000, 4, 64
    qkv = torch.randn(n, 3, h, c // h, device="cuda", dtype=torch.bfloat16)
    cu = torch.tensor([0, 1024, 2048, n], device="cuda", dtype=torch.int32)
    o = flash_attn.flash_attn_varlen_qkvpacked_func(qkv, cu, max_seqlen=1024)
    assert o.shape == (n, h, c // h) and torch.isfinite(o).all()
    msg = []
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):   # pointops allocates its outputs on the current device
            xyz = torch.rand(2000, 3, device=f"cuda:{i}")
            off = torch.tensor([1000, 2000], device=f"cuda:{i}")
            idx, dist = pointops.knn_query(8, xyz, off)
            assert idx.shape == (2000, 8) and (idx[:1000] < 1000).all() and (idx[1000:] >= 1000).all()
            a = xyz[:1000]
            ref = (a[:, None] - a[None]).norm(dim=-1).topk(8, largest=False).values
            err = (dist[:1000].sort(1).values - ref).abs().max().item()
            assert err < 1e-4, f"knn distances wrong on cuda:{i} (max err {err})"
        msg.append(f"pointops ok on sm_{''.join(map(str, torch.cuda.get_device_capability(i)))}")
    return f"flash_attn {flash_attn.__version__} | " + ", ".join(msg)


def _model_from_config(rel, **backbone_overrides):
    import torch
    from pointcept.utils.config import Config
    from pointcept.models import build_model
    cfg = Config.fromfile(os.path.join(REPO, rel))
    cfg.model.backbone.update(backbone_overrides)
    torch.manual_seed(0)
    return build_model(cfg.model).cuda().eval()


@check("PoinTTT spatial model (PT-v3-LaCT-Dec): evaluation forward over 3 chunks")
def t_lact_dec():
    import torch
    model = _model_from_config(EVAL_CONFIGS[0], chunk_size=4096)
    data = synthetic_cloud((9000,), n_cls=3)
    with torch.no_grad():
        out = model(dict(data))
    assert out["seg_logits"].shape == (data["coord"].shape[0], 3) and torch.isfinite(out["seg_logits"]).all()
    return f"logits {tuple(out['seg_logits'].shape)}"


@check("PoinTTT temporal model (PT-v3-LaCT-Temporal): online evaluation forward, 2 sequences x 3 clips")
def t_lact_temporal():
    import torch
    model = _model_from_config(EVAL_CONFIGS[1])
    data = synthetic_cloud((6000, 6000), n_cls=49, grid=0.02)
    n0 = int(data["offset"][0]); n1 = int(data["offset"][1]) - n0
    data["clip_idx"] = torch.cat([torch.arange(n0, device="cuda") * 3 // n0, torch.arange(n1, device="cuda") * 3 // n1])
    with torch.no_grad():
        out = model(dict(data))
    assert out["seg_logits"].shape == (data["coord"].shape[0], 49) and torch.isfinite(out["seg_logits"]).all()
    return f"logits {tuple(out['seg_logits'].shape)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expect-torch", required=True, choices=("2.5.0", "2.7.1"))
    ap.add_argument("--no-forward", action="store_true", help="skip everything that needs a GPU")
    args = ap.parse_args()
    t_python()
    t_torch(args.expect_torch, not args.no_forward)
    t_packages()
    t_pointcept()
    if not args.no_forward:
        t_spconv()
        t_cuda_ops()
        t_lact_dec()
        t_lact_temporal()
    bad = [r for r in RESULTS if r[0] == "FAIL"]
    print("-" * 100)
    print(f"{sum(r[0] == 'PASS' for r in RESULTS)} passed, {len(bad)} failed")
    if bad:
        print("SMOKE TEST FAILED: " + "; ".join(r[1] for r in bad))
        sys.exit(1)
    print("SMOKE TEST PASSED" + (" (without the GPU checks)" if args.no_forward else ""))


if __name__ == "__main__":
    main()
