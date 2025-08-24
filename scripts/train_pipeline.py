# scripts/train_pipeline.py
import argparse
import os
import torch
from torch.optim import AdamW

from sigrid.models.unet import build_unet
from sigrid.models.fcn import build_fcn
from sigrid.data.dataset import SIGridDataset
from sigrid.data.transforms import setup_transforms
from sigrid.train.trainer import make_loader, train_one_epoch

def parse_args():
    ap = argparse.ArgumentParser(description="Train UNet/FCN on SIGrid dataset.")
    ap.add_argument("--arch", default="unet", choices=["unet","fcn"])
    ap.add_argument("--dataset", default="CUB")
    ap.add_argument("--images", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--cache_root", default=None)
    ap.add_argument("--n_segments", type=int, default=500)
    ap.add_argument("--compactness", type=float, default=10.0)
    ap.add_argument("--grid", type=int, default=96)
    ap.add_argument("--features", default="avg")
    ap.add_argument("--model", default="full", choices=["full", "reduced"])
    ap.add_argument("--downsample", default="pool", choices=["pool", "stride"])
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()

def main():
    args = parse_args()
    os.environ.setdefault("SIGRID_CACHE", args.cache_root or f"./artifacts/{args.dataset}/cache")

    # Feature flags
    features = {k: True for k in args.features.split(",") if k}

    # Transforms (no unit-255 scaling by default, since SIGrid channels may be non-[0,255])
    channels = sum(1 for _ in features if features[_]) or 1  # rough fallback if needed
    spatial_t, norm_t = setup_transforms(channels=channels, normalize="none")

    # Dataset & loader
    ds = SIGridDataset(
        dataset_name=args.dataset,
        n_segments=args.n_segments,
        compactness=args.compactness,
        grid_size=args.grid,
        image_dir=args.images,
        mask_dir=args.masks,
        spatial_transform=spatial_t,
        norm_transform=norm_t,
        features=features,
        cache_root=args.cache_root,
    )
    loader = make_loader(ds, batch_size=args.batch_size, shuffle=True)

    # Model, optim
    if args.arch == "unet":
        model = build_unet(input_channels=channels, model=args.model, downsample=args.downsample).to(args.device)
    else:
        model = build_fcn(input_channels=channels, base=64).to(args.device)
    optim = AdamW(model.parameters(), lr=args.lr)

    # Train
    for epoch in range(1, args.epochs + 1):
        avg_loss = train_one_epoch(model, loader, optim, device=args.device, use_amp=not args.no_amp)
        print(f"[Epoch {epoch}/{args.epochs}] loss={avg_loss:.6f}")

if __name__ == "__main__":
    main()