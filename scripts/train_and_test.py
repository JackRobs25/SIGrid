# scripts/train_pipeline.py
import argparse, os, torch, warnings
from torch.optim import AdamW
from albumentations import Compose
from torchinfo import summary

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
    ap.add_argument("--train_images", required=True)
    ap.add_argument("--train_masks", required=True)
    ap.add_argument("--test_images", required=True)
    ap.add_argument("--test_masks", required=True)
    ap.add_argument("--cache_root", default=None)

    ap.add_argument("--n_segments", type=int, default=500)
    ap.add_argument("--compactness", type=int, default=20)
    ap.add_argument("--grid", type=int, default=96)
    ap.add_argument("--features", default="avg")

    ap.add_argument("--arch", default="unet", choices=["unet","fcn"])
    ap.add_argument("--fcn_backbone", default="vgg_light3",
                choices=["vgg_light3","vgg11","vgg13","vgg16","vgg19"])
    ap.add_argument("--fcn_pretrained", action="store_true",
                help="use torchvision VGG weights (only if channels==3 and not vgg_light3)")
    ap.add_argument("--downsample", default="pool", choices=["pool","stride"])  # unet-only

    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--no_amp", action="store_true")  
    ap.add_argument("--device", type=str, default=DEVICE_DEFAULT, help="Device to run on: 'cuda', 'mps', or 'cpu'. Defaults to best available."
)

    # eval options
    ap.add_argument("--eval_every", type=int, default=5, help="run eval every N epochs")
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
    spatial_t, norm_t = setup_transforms(channels=1, normalize="unit255")

    # Dataset
    ds_train = SIGridDataset(
        dataset_name=args.dataset,
        n_segments=args.n_segments,
        compactness=args.compactness,
        grid_size=args.grid,
        image_dir=args.train_images,
        mask_dir=args.train_masks,
        spatial_transform=spatial_t,      # augments on train
        norm_transform=norm_t,
        features=features,
        cache_root=args.cache_root,
    )

    ds_test = SIGridDataset(
        dataset_name=args.dataset,
        n_segments=args.n_segments,
        compactness=args.compactness,
        grid_size=args.grid,
        image_dir=args.test_images,
        mask_dir=args.test_masks,
        spatial_transform=Compose([]),     
        norm_transform=norm_t,
        features=features,
        cache_root=args.cache_root,
    )
    # loader
    train_loader = make_loader(ds_train, batch_size=args.batch_size, shuffle=True)
    test_loader = make_loader(ds_test, batch_size=args.batch_size, shuffle=True)

    # derive channels from first cached sample
    channels = int(ds_train.sig_list[0].shape[0])

    # Model
    if args.arch == "unet":
        model = build_unet(input_channels=channels, downsample=args.downsample).to(args.device)
    else:
        model = build_fcn(
            input_channels=channels,
            n_class=1,
            backbone=args.fcn_backbone,
            pretrained=args.fcn_pretrained,
            requires_grad=True,
        ).to(args.device)

    summary(model, input_size=(1, channels, args.grid, args.grid))

    optim = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Simple train loop with periodic eval on the *same* split.
    # If you want a held-out split, run generate_sigrids for train+test and point --images/--masks to the test dirs for eval.
    for e in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optim, device=args.device, use_amp=use_amp)
        print(f"[Epoch {e}/{args.epochs}] train_loss={loss:.6f}")

        if args.eval_every and (e % args.eval_every == 0):
            avg_train_loss, c_train_iou, c_train_f, c_train_acc, p_train_iou, p_train_f, p_train_acc = evaluate(
                model, train_loader, device=args.device, testing=False,
                translate_cells_to_pixels=False,
                save_preds_dir=None,
                save_first_n=0,
            )
            avg_test_loss, c_test_iou, c_test_f, c_test_acc, p_test_iou, p_test_f, p_test_acc = evaluate(
                model, test_loader, device=args.device, testing=True,
                translate_cells_to_pixels=False,
                save_preds_dir=None,
                save_first_n=0,
            )
            print(f"[Eval e={e}] "
                f"Train: loss={avg_train_loss:.4f}, "
                f"cell_acc={c_train_acc:.2f}%, cell_iou={c_train_iou:.2f}%, cell_fβ={c_train_f:.2f}%, "
                f"pixel_acc={p_train_acc:.2f}%, pixel_iou={p_train_iou:.2f}%, pixel_fβ={p_train_f:.2f}% | "
                f"Test: loss={avg_test_loss:.4f}, "
                f"cell_acc={c_test_acc:.2f}%, cell_iou={c_test_iou:.2f}%, cell_fβ={c_test_f:.2f}%, "
                f"pixel_acc={p_test_acc:.2f}%, pixel_iou={p_test_iou:.2f}%, pixel_fβ={p_test_f:.2f}%")
    
    avg_test_loss, c_test_iou, c_test_f, c_test_acc, p_test_iou, p_test_f, p_test_acc = evaluate(
                model, test_loader, device=args.device, testing=True,
                translate_cells_to_pixels=True,
                save_preds_dir=args.save_preds_dir,
                save_first_n=args.save_first_n,
            )      
    print(f"[Eval e={e}] "
                f"Test: loss={avg_test_loss:.4f}, "
                f"cell_acc={c_test_acc:.2f}%, cell_iou={c_test_iou:.2f}%, cell_fβ={c_test_f:.2f}%, "
                f"pixel_acc={p_test_acc:.2f}%, pixel_iou={p_test_iou:.2f}%, pixel_fβ={p_test_f:.2f}%")
    

if __name__ == "__main__":
    main()