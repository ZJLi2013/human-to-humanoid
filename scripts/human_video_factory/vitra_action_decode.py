#!/usr/bin/env python3
"""feature8 / Exp-8.0 glue: probe + decode VITRA's action chunk into a structured
hand trajectory (wrist 6D pose + finger joint rotations), the input the Do-as-I-Do
retarget stage-1 consumes.

Why "probe, don't guess": the exact action dim (51 / 102 / 192 seen in different notes)
and the wrist-vs-finger split depend on VITRA's ActionFeature layout + normalizer. This
script introspects that layout from the *loaded* model/normalizer instead of hardcoding,
then decodes a normalized action array accordingly.

Run inside the `vitra-rocm` image (same load path as scripts/vitra/run_vitra_rocm_smoke.sh):
  # probe only (no GPU inference): dump action/state feature layout + normalizer dims
  python vitra_action_decode.py --vitra_repo /path/to/VITRA --mode probe

  # full: load model, run one predict_action, decode, dump structured hand traj npz
  python vitra_action_decode.py --vitra_repo /path/to/VITRA --mode decode \
      --image examples/0002.jpg --out /out/hand_traj.npz

Output npz (decode mode): per predicted step
  wrist_pose (T,4,4)  world->wrist or cam->wrist per VITRA convention (recorded in meta)
  finger     (T,Kf)   finger joint values (angle repr, denormalized)
  meta.json  the discovered layout so downstream retarget glue reads slices by name
"""
import argparse
import json
import os
import sys


def _introspect_feature(feat_cls, name):
    """Dump whatever slice/layout info an ActionFeature/StateFeature exposes, defensively."""
    info = {"class": name, "attrs": {}}
    for a in dir(feat_cls):
        if a.startswith("__"):
            continue
        try:
            v = getattr(feat_cls, a)
        except Exception:
            continue
        if callable(v):
            continue
        # keep only JSON-friendly scalars / small containers
        if isinstance(v, (int, float, str, bool)):
            info["attrs"][a] = v
        elif isinstance(v, (list, tuple)) and len(v) <= 64:
            try:
                json.dumps(v)
                info["attrs"][a] = list(v)
            except Exception:
                info["attrs"][a] = str(v)[:200]
        elif isinstance(v, dict) and len(v) <= 64:
            try:
                json.dumps(v)
                info["attrs"][a] = v
            except Exception:
                info["attrs"][a] = {k: str(x)[:80] for k, x in v.items()}
    return info


def probe(vitra_repo):
    sys.path.insert(0, vitra_repo)
    import numpy as np
    from vitra.datasets.dataset_utils import ActionFeature, StateFeature

    report = {
        "action_feature": _introspect_feature(ActionFeature, "ActionFeature"),
        "state_feature": _introspect_feature(StateFeature, "StateFeature"),
    }
    # normalizer dims (needs a config; best-effort so probe still works offline)
    try:
        from vitra.utils.config_utils import load_config
        from vitra.utils.data_utils import load_normalizer
        repo = "VITRA-VLA/VITRA-VLA-3B"
        cfg = load_config(repo)
        cfg["model_load_path"] = repo
        cfg["statistics_path"] = repo
        nz = load_normalizer(cfg)
        report["normalizer"] = {
            "action_mean_dim": int(np.asarray(nz.action_mean).shape[0]),
            "state_mean_dim": int(np.asarray(nz.state_mean).shape[0]),
        }
    except Exception as e:
        report["normalizer_error"] = repr(e)
    print(json.dumps(report, indent=2))
    return report


def _split_action(arr, layout):
    """Split a (T,D) denormalized action into named segments using discovered layout.

    layout: dict name->[start,end] (half-open) discovered in probe. If absent, returns
    the raw array under 'raw' so the caller still gets *something* to inspect.
    """
    import numpy as np
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    out = {}
    if layout:
        for name, (s, e) in layout.items():
            if 0 <= s < e <= arr.shape[-1]:
                out[name] = arr[..., s:e]
    if not out:
        out["raw"] = arr
    return out


def decode(vitra_repo, image_path, out_path, layout_json=None):
    """Load model, run ONE predict_action, denormalize, split by layout, dump npz.

    Mirrors run_vitra_rocm_smoke.sh's load path. layout_json (from a prior probe) maps
    segment name -> [start,end]; if omitted, dumps raw denormalized action for inspection.
    """
    sys.path.insert(0, vitra_repo)
    import numpy as np
    import torch
    from PIL import Image
    from vitra.models.vla_builder import build_vla
    from vitra.utils.data_utils import resize_short_side_to_target, load_normalizer
    from vitra.datasets.human_dataset import pad_state_human, pad_action
    from vitra.utils.config_utils import load_config
    from vitra.datasets.dataset_utils import ActionFeature, StateFeature
    from huggingface_hub import hf_hub_download, list_repo_files

    repo = "VITRA-VLA/VITRA-VLA-3B"
    cfg = load_config(repo)
    cfg["model_load_path"] = repo
    cfg["statistics_path"] = repo
    model = build_vla(cfg)
    files = list_repo_files(repo)
    ckpts = [f for f in files if f.startswith("checkpoints/") and f.endswith(".pt")] \
        or [f for f in files if f.endswith(".pt")]
    sd = torch.load(hf_hub_download(repo, ckpts[0]), map_location="cpu")
    model.load_state_dict(sd, strict=False)
    model = model.cuda().eval()
    nz = load_normalizer(cfg)

    if image_path and os.path.exists(image_path):
        image = Image.open(image_path).convert("RGB")
    else:
        image = Image.fromarray((np.random.rand(480, 640, 3) * 255).astype("uint8"))
    image = np.array(resize_short_side_to_target(image, target=224))
    fov = torch.tensor([[np.deg2rad(60.0), np.deg2rad(60.0)]], dtype=torch.float32)

    state = np.zeros((nz.state_mean.shape[0],))
    state_mask = np.array([False, True], dtype=bool)
    action_mask = np.tile(np.array([[False, True]], dtype=bool), (model.chunk_size, 1))
    unified_state, unified_state_mask = pad_state_human(
        state=nz.normalize_state(state), state_mask=state_mask,
        action_dim=nz.action_mean.shape[0], state_dim=nz.state_mean.shape[0],
        unified_state_dim=StateFeature.ALL_FEATURES[1])
    _, unified_action_mask = pad_action(
        actions=None, action_mask=action_mask,
        action_dim=nz.action_mean.shape[0], unified_action_dim=ActionFeature.ALL_FEATURES[1])

    norm_action = model.predict_action(
        image=image, instruction="Left hand: None. Right hand: pick up the object.",
        current_state=unified_state.float().unsqueeze(0),
        current_state_mask=unified_state_mask.unsqueeze(0),
        action_mask_torch=unified_action_mask.unsqueeze(0),
        num_ddim_steps=10, cfg_scale=5.0, fov=fov, sample_times=1)
    norm_action = np.asarray(norm_action, dtype="float32")

    # VITRA pads the action to a unified dim (192); real content = first action_mean_dim (102).
    real_dim = int(np.asarray(nz.action_mean).shape[0])
    act = norm_action[..., :real_dim]

    # de-normalize (best-effort: try normalizer methods on the real-dim slice, else manual).
    denorm = None
    for m in ("denormalize_action", "unnormalize_action", "inverse_action", "unnorm_action"):
        if hasattr(nz, m):
            try:
                denorm = np.asarray(getattr(nz, m)(act), dtype="float32")
                break
            except Exception:
                pass
    if denorm is None:
        std = getattr(nz, "action_std", None)
        if std is None:
            std = getattr(nz, "action_scale", 1.0)
        denorm = act * np.asarray(std) + np.asarray(nz.action_mean)

    layout = json.loads(layout_json) if layout_json else None
    segs = _split_action(denorm, layout)
    meta = {
        "norm_action_shape": list(norm_action.shape),
        "real_dim": real_dim,
        "denorm_action_shape": list(np.asarray(denorm).shape),
        "action_finite": bool(np.isfinite(denorm).all()),
        "segments": {k: list(np.asarray(v).shape) for k, v in segs.items()},
        "layout_used": layout or "none (raw dumped) — run --mode probe to discover slices",
        "note": "wrist frame convention (cam-relative) per VITRA; confirm before retarget.",
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez(out_path, **{k: np.asarray(v) for k, v in segs.items()})
    with open(os.path.splitext(out_path)[0] + ".meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print(json.dumps(meta, indent=2))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vitra_repo", required=True)
    ap.add_argument("--mode", choices=["probe", "decode"], default="probe")
    ap.add_argument("--image", default=None)
    ap.add_argument("--out", default="/tmp/vitra_hand_traj.npz")
    ap.add_argument("--layout_json", default=None,
                    help='JSON like {"wrist":[0,6],"finger":[6,51]} from a prior probe')
    args = ap.parse_args()
    if args.mode == "probe":
        probe(args.vitra_repo)
    else:
        decode(args.vitra_repo, args.image, args.out, args.layout_json)


if __name__ == "__main__":
    main()
