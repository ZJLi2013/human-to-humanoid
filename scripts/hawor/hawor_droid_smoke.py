"""HaWoR masked DROID-SLAM `droid_backends` GPU functional smoke (ROCm / gfx942).

Exercises the ROCm-ported CUDA/HIP kernels on the AMD GPU:
  - droid_kernels.cu   (iproj / projmap / frame_distance)  <- P5 Eigen fill_n path
  - correlation_kernels.cu (corr_index_forward)            <- P4 .type()->.scalar_type()
  - altcorr_kernel.cu      (altcorr_forward)               <- P4 .type()->.scalar_type()
Plus the sibling ROCm-ported deps: lietorch (SE3) and torch_scatter (scatter_mean).

Emits a JSON summary to stdout (line-prefixed SMOKE_JSON=) for the runner to capture.
"""
import json
import sys
import traceback

import torch

RESULT = {"ops": {}, "env": {}}


def record(name, fn):
    try:
        out = fn()
        RESULT["ops"][name] = {"status": "PASS", **out}
        print(f"[smoke] {name}: PASS {out}")
    except Exception as e:  # noqa: BLE001
        RESULT["ops"][name] = {"status": "FAIL", "error": repr(e)}
        print(f"[smoke] {name}: FAIL {e!r}")
        traceback.print_exc()


def finite(t):
    return bool(torch.isfinite(t).all().item())


def main():
    assert torch.cuda.is_available(), "no ROCm/CUDA device visible"
    dev = "cuda"
    RESULT["env"] = {
        "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "device": torch.cuda.get_device_name(0),
    }
    print(f"[smoke] env: {RESULT['env']}")

    import droid_backends  # noqa: F401

    B, N, H, W = 1, 4, 48, 64
    # The droid_kernels geom ops (iproj/projmap/frame_distance) operate on the DROID
    # video buffers WITHOUT a batch dim: poses [N,7], disps [N,H,W], intrinsics [N,4].
    # SE3 pose data = translation(3)+quat(4); identity quat = (0,0,0,1); give frames a
    # small translation so edges are non-degenerate.
    poses = torch.zeros(N, 7, device=dev, dtype=torch.float32)
    poses[:, 6] = 1.0
    poses[:, 0] = torch.linspace(0.0, 0.3, N, device=dev)  # translate along x
    disps = (0.5 + torch.rand(N, H, W, device=dev)).contiguous()  # positive inverse-depth
    fx = fy = 200.0
    cx, cy = W / 2.0, H / 2.0
    # DROID shares one intrinsics vector across frames -> pass a 1-D [4] tensor.
    intrinsics = torch.tensor([fx, fy, cx, cy], device=dev).contiguous()
    ii = torch.tensor([0, 1, 2], device=dev, dtype=torch.long)
    jj = torch.tensor([1, 2, 3], device=dev, dtype=torch.long)

    # --- droid_kernels.cu (P5 Eigen path) ---
    def op_iproj():
        pts = droid_backends.iproj(poses, disps, intrinsics)
        torch.cuda.synchronize()
        return {"out_shape": list(pts.shape), "finite": finite(pts)}
    record("iproj", op_iproj)

    def op_projmap():
        out = droid_backends.projmap(poses, disps, intrinsics, ii, jj)
        coords = out[0]
        torch.cuda.synchronize()
        return {"out_shape": list(coords.shape), "finite": finite(coords)}
    record("projmap", op_projmap)

    def op_frame_distance():
        d = droid_backends.frame_distance(poses, disps, intrinsics, ii, jj, 0.3)
        torch.cuda.synchronize()
        return {"out_shape": list(d.shape), "finite": finite(d), "sample": d.flatten()[:4].tolist()}
    record("frame_distance", op_frame_distance)

    # --- correlation_kernels.cu (P4) via repo CorrBlock ---
    sys.path.insert(0, "droid_slam")

    def op_corr_index():
        from modules.corr import CorrBlock
        C, h, w = 128, H // 8, W // 8
        fmap1 = torch.randn(B, N, C, h, w, device=dev)
        fmap2 = torch.randn(B, N, C, h, w, device=dev)
        y, x = torch.meshgrid(torch.arange(h, device=dev).float(),
                              torch.arange(w, device=dev).float(), indexing="ij")
        coords = torch.stack([x, y], dim=-1).view(1, 1, h, w, 2).repeat(B, N, 1, 1, 1).contiguous()
        corr = CorrBlock(fmap1, fmap2, num_levels=2, radius=3)(coords)
        torch.cuda.synchronize()
        return {"out_shape": list(corr.shape), "finite": finite(corr)}
    record("corr_index_forward", op_corr_index)

    # --- altcorr_kernel.cu (P4) via repo AltCorrBlock ---
    def op_altcorr():
        from modules.corr import AltCorrBlock
        C, h, w = 128, H // 8, W // 8
        fmaps = torch.randn(B, N, C, h, w, device=dev)
        y, x = torch.meshgrid(torch.arange(h, device=dev).float(),
                              torch.arange(w, device=dev).float(), indexing="ij")
        coords = torch.stack([x, y], dim=-1).view(1, 1, h, w, 2).repeat(B, N, 1, 1, 1).contiguous()
        block = AltCorrBlock(fmaps, num_levels=2, radius=3)
        corr = block(coords, ii=torch.tensor([0, 1, 2, 3], device=dev),
                     jj=torch.tensor([0, 1, 2, 3], device=dev))
        torch.cuda.synchronize()
        return {"out_shape": list(corr.shape), "finite": finite(corr)}
    record("altcorr_forward", op_altcorr)

    # --- sibling ROCm-ported deps ---
    def op_lietorch():
        from lietorch import SE3
        g = SE3(poses)
        y = (g * g.inv()).log()
        torch.cuda.synchronize()
        return {"out_shape": list(y.shape), "finite": finite(y)}
    record("lietorch_SE3", op_lietorch)

    def op_scatter():
        from torch_scatter import scatter_mean
        src = torch.randn(64, device=dev)
        idx = torch.randint(0, 8, (64,), device=dev)
        out = scatter_mean(src, idx, dim=0)
        torch.cuda.synchronize()
        return {"out_shape": list(out.shape), "finite": finite(out)}
    record("torch_scatter", op_scatter)

    core = ["iproj", "projmap", "frame_distance"]
    corr = ["corr_index_forward", "altcorr_forward"]
    passed = {k: v["status"] == "PASS" for k, v in RESULT["ops"].items()}
    RESULT["core_pass"] = all(passed.get(k) for k in core)
    RESULT["corr_pass"] = all(passed.get(k) for k in corr)
    RESULT["overall_pass"] = RESULT["core_pass"] and RESULT["corr_pass"]
    print("SMOKE_JSON=" + json.dumps(RESULT))
    print("HAWOR_DROID_SMOKE_PASS" if RESULT["overall_pass"] else "HAWOR_DROID_SMOKE_FAIL")
    return 0 if RESULT["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
