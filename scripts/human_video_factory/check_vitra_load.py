#!/usr/bin/env python3
"""VITRA-side smoke for feature3 / P1: load an episode through the stock VITRA
FrameDataset and assemble ONE batch. Proves the episode layout is loadable without
forking VITRA. Works on both a real ssv2 crop and hawor_to_vitra.py output.

Signature matches vitra/datasets/dataset.py::FrameDataset (verified on banff):
  FrameDataset(dataset_folder, dataset_name='ssv2', processor=..., action_future_window_size=15,
               load_images=False, normalization=True, augmentation=False, data_type='human')
--load_images defaults off (skips decord/video) so pose+action assembly is validated alone.
"""
import argparse
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vitra_repo", required=True)
    ap.add_argument("--dataset_folder", required=True, help="dir containing Annotation/ (and Video/)")
    ap.add_argument("--dataset_name", default="ssv2")
    ap.add_argument("--pretrained", default="google/paligemma2-3b-mix-224")
    ap.add_argument("--load_images", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, args.vitra_repo)
    import numpy as np
    import torch
    from vitra.datasets.dataset import FrameDataset

    processor = None
    if args.load_images:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(args.pretrained)

    ds = FrameDataset(
        dataset_folder=args.dataset_folder,
        dataset_name=args.dataset_name,
        processor=processor,
        action_past_window_size=0,
        action_future_window_size=15,
        action_type="angle",
        use_rel=False,
        normalization=True,
        augmentation=False,
        load_images=args.load_images,
        data_type="human",
    )
    print(f"[ds] len={len(ds)}")

    sample = ds[0]
    print("[sample] keys:", sorted(sample.keys()))
    for k, v in sample.items():
        if isinstance(v, (torch.Tensor, np.ndarray)):
            print(f"  {k:22s} {tuple(v.shape)} {v.dtype}")
        elif isinstance(v, str):
            print(f"  {k:22s} str: {v[:70]!r}")
    print("\n[OK] episode loaded through stock VITRA FrameDataset.")


if __name__ == "__main__":
    main()
