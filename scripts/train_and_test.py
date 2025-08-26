# scripts/train_pipeline.py
import argparse, os, torch, warnings
from torch.optim import AdamW
from albumentations import Compose
from torchinfo import summary
import copy
import time
from types import SimpleNamespace
from scripts.save_data import save_experiment
from calflops import calculate_flops

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
    ap.add_argument("--workdir", default=None, help="Directory to save model and metrics JSON (defaults to ./artifacts/<dataset>/results)")

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

    results_dir = args.workdir or f"./artifacts/{args.dataset}/results"
    os.makedirs(results_dir, exist_ok=True)

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

    # Feature flags from args.features
    feat_flags = {k: True for k in args.features.split(",") if k}
    avg_color = bool(feat_flags.get("avg", False) or feat_flags.get("avg_col", False))
    area = bool(feat_flags.get("area", False))
    width = bool(feat_flags.get("width", False))
    height = bool(feat_flags.get("height", False))
    compac = bool(feat_flags.get("compac", False) or feat_flags.get("compactness", False))
    solidity = bool(feat_flags.get("solidity", False))
    eccentricity = bool(feat_flags.get("eccentricity", False) or feat_flags.get("ecc", False))
    hu = bool(feat_flags.get("hu", False))

    # State container for saving at the end
    state = SimpleNamespace(
        dataset=args.dataset,
        model=("UNet" if args.arch == "unet" else "FCN"),
        downsample=(args.downsample if args.arch == "unet" else "NA"),
        n_segments=args.n_segments,
        compactness=args.compactness,
        SIGrid_channels=channels,
        dim=args.grid,
        avg_color=avg_color, area=area, width=width, height=height,
        compac=compac, solidity=solidity, eccentricity=eccentricity, hu=hu,
        learning_rate=args.lr, num_epochs=args.epochs, batch_size=args.batch_size,
        training_cell_iou=[], training_cell_accuracies=[], training_pixel_iou=[], training_pixel_accuracies=[],
        testing_cell_iou=[], testing_pixel_accuracies=[], testing_pixel_iou=[],
        test_cell_iou=None, test_cell_f=None, test_cell_accuracy=None,
        test_pixel_iou=None, test_pixel_f=None, test_pixel_accuracy=None,
        test_losses=[], train_losses=[],
        start_time=time.time(), end_time=None, flops=None,
        network=None
    )

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

    state.network = model
    # ---- SAFE TORCHINFO SUMMARY ----
    try:
        input_shape = (1, channels, args.grid, args.grid)
        model_for_summary = copy.deepcopy(model).to("cpu")  # isolate & keep summary on CPU
        summary(model_for_summary, input_size=input_shape)
        flops, macs, params = calculate_flops(model=model_for_summary, 
                                      input_shape=input_shape,
                                      output_as_string=True,
                                      output_precision=4)
        state.flops = flops
    finally:
        del model_for_summary

    optim = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Simple train loop with periodic eval on the *same* split.
    # If you want a held-out split, run generate_sigrids for train+test and point --images/--masks to the test dirs for eval.
    for e in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optim, device=args.device, use_amp=use_amp)
        state.train_losses.append(float(loss))
        print(f"[Epoch {e}/{args.epochs}] train_loss={loss:.6f}")

        if args.eval_every and (e % args.eval_every == 0):
            avg_train_loss, c_train_iou, c_train_f, c_train_acc, p_train_iou, p_train_f, p_train_acc = evaluate(
                model, train_loader, device=args.device, testing=False,
                translate_cells_to_pixels=False,
                save_preds_dir=None,
                save_first_n=0,
            )
            state.training_cell_iou.append(float(c_train_iou))
            state.training_cell_accuracies.append(float(c_train_acc))
            state.training_pixel_iou.append(float(p_train_iou))
            state.training_pixel_accuracies.append(float(p_train_acc))
            avg_test_loss, c_test_iou, c_test_f, c_test_acc, p_test_iou, p_test_f, p_test_acc = evaluate(
                model, test_loader, device=args.device, testing=True,
                translate_cells_to_pixels=False,
                save_preds_dir=None,
                save_first_n=0,
            )
            state.testing_cell_iou.append(float(c_test_iou))
            state.testing_pixel_accuracies.append(float(p_test_acc))
            state.testing_pixel_iou.append(float(p_test_iou))
            state.test_losses.append(float(avg_test_loss))
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
    
    # Record final test metrics
    state.test_cell_iou = float(c_test_iou)
    state.test_cell_f = float(p_test_f) if p_test_f is not None else None  # placeholder; keep schema
    state.test_cell_accuracy = float(c_test_acc)
    state.test_pixel_iou = float(p_test_iou)
    state.test_pixel_f = float(p_test_f)
    state.test_pixel_accuracy = float(p_test_acc)
    state.end_time = time.time()

    # Persist model + metrics JSON
    save_experiment(state, workdir=results_dir)
    

if __name__ == "__main__":
    main()