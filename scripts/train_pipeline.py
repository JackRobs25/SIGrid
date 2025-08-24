# scripts/train_pipeline.py
import argparse, os, torch, warnings
from torch.optim import AdamW

from sigrid.models.unet import build_unet
from sigrid.models.fcn import build_fcn
from sigrid.data.dataset import SIGridDataset
from sigrid.data.transforms import setup_transforms
from sigrid.train.trainer import make_loader, train_one_epoch, evaluate

# Best-available default: CUDA > MPS > CPU
DEVICE_DEFAULT = (
    "cuda" if torch.cuda.is_available()
    else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    else "cpu"
)

def parse_args():
    ap = argparse.ArgumentParser(description="Train on SIGrid dataset.")
    ap.add_argument("--dataset", default="CUB")
    ap.add_argument("--images", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--cache_root", default=None)

    ap.add_argument("--n_segments", type=int, default=500)
    ap.add_argument("--compactness", type=float, default=10.0)
    ap.add_argument("--grid", type=int, default=96)
    ap.add_argument("--features", default="avg")

    ap.add_argument("--arch", default="unet", choices=["unet","fcn"])
    ap.add_argument("--fcn_backbone", default="vgg_light3",
                choices=["vgg_light3","vgg11","vgg13","vgg16","vgg19"])
    ap.add_argument("--fcn_pretrained", action="store_true",
                help="use torchvision VGG weights (only if channels==3 and not vgg_light3)")
    ap.add_argument("--model", default="full", choices=["full","reduced"])   # unet-only
    ap.add_argument("--downsample", default="pool", choices=["pool","stride"])  # unet-only

    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--no_amp", action="store_true")  
    ap.add_argument("--device", type=str, default=DEVICE_DEFAULT, help="Device to run on: 'cuda', 'mps', or 'cpu'. Defaults to best available."
)

    # eval options
    ap.add_argument("--eval_every", type=int, default=1, help="run eval every N epochs")
    ap.add_argument("--translate_cells", action="store_true", help="translate grid preds back to pixels for eval")
    ap.add_argument("--save_preds_dir", default=None, help="optional directory to dump a few predictions")
    ap.add_argument("--save_first_n", type=int, default=0, help="how many preds to save")

    return ap.parse_args()

def main():
    args = parse_args()
    # Validate requested device & set AMP policy
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        use_amp = not args.no_amp  # CUDA AMP is supported
    elif args.device == "mps":
        if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            raise RuntimeError("MPS requested but not available on this system.")
        use_amp = False  # no CUDA AMP on MPS
        if not args.no_amp:
            warnings.warn("AMP disabled on MPS (not supported). Running in FP32.", RuntimeWarning)
    else:
        # CPU
        use_amp = False
        if not args.no_amp:
            warnings.warn("AMP disabled on CPU. Running in FP32.", RuntimeWarning)
    
    os.environ.setdefault("SIGRID_CACHE", args.cache_root or f"./artifacts/{args.dataset}/cache")

    features = {k: True for k in args.features.split(",") if k}

    # provisional transforms; normalization="none" because SIGrid channels are not [0..255]
    spatial_t, norm_t = setup_transforms(channels=1, normalize="none")

    # Dataset
    ds_train = SIGridDataset(
        dataset_name=args.dataset,
        n_segments=args.n_segments,
        compactness=args.compactness,
        grid_size=args.grid,
        image_dir=args.images,
        mask_dir=args.masks,
        spatial_transform=spatial_t,      # augments on train
        norm_transform=norm_t,
        features=features,
        cache_root=args.cache_root,
    )
    # loader
    loader = make_loader(ds_train, batch_size=args.batch_size, shuffle=True)

    # derive channels from first cached sample
    channels = int(ds_train.sig_list[0].shape[0])

    # Model
    if args.arch == "unet":
        model = build_unet(input_channels=channels, model=args.model, downsample=args.downsample).to(args.device)
    else:
        model = build_fcn(
            input_channels=channels,
            n_class=1,
            backbone=args.fcn_backbone,
            pretrained=args.fcn_pretrained,
            requires_grad=True,
        ).to(args.device)

    optim = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Simple train loop with periodic eval on the *same* split.
    # If you want a held-out split, run generate_sigrids for train+test and point --images/--masks to the test dirs for eval.
    for e in range(1, args.epochs + 1):
        loss = train_one_epoch(model, loader, optim, device=args.device, use_amp=use_amp)
        print(f"[Epoch {e}/{args.epochs}] train_loss={loss:.6f}")

        if args.eval_every and (e % args.eval_every == 0):
            avg_loss, c_iou, c_f, c_acc, p_iou, p_f, p_acc = evaluate(
                model, loader, device=args.device,
                translate_cells_to_pixels=args.translate_cells,
                save_preds_dir=args.save_preds_dir,
                save_first_n=args.save_first_n,
            )
            print(f"[Eval e={e}] loss={avg_loss:.4f}  "
                  f"cell_acc={c_acc:.2f}%  cell_iou={c_iou:.2f}%  cell_fβ={c_f:.2f}%  "
                  f"pixel_acc={p_acc:.2f}%  pixel_iou={p_iou:.2f}%  pixel_fβ={p_f:.2f}%")

if __name__ == "__main__":
    main()