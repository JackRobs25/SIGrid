# scripts/generate_sigrids.py
import argparse
from sigrid.pipeline.sigrid_compute import generate_sigrids

def parse_args():
    ap = argparse.ArgumentParser(description="Generate SIGrids from an image+mask dataset.")
    ap.add_argument("--input", required=True, help="Path to images root (e.g., data/CUB/train_images)")
    ap.add_argument("--masks", required=True, help="Path to masks root (e.g., data/CUB/train_masks)")
    ap.add_argument("--output", required=True, help="Where to save cached SIGrids (e.g., artifacts/CUB/cache)")
    ap.add_argument("--dataset", default="CUB")
    ap.add_argument("--n_segments", type=int, default=500)
    ap.add_argument("--compactness", type=float, default=10.0)
    ap.add_argument("--grid", type=int, default=96)
    ap.add_argument("--features", default="avg", help="Comma list: avg,area,width,height,compac,solidity,eccentricity,hu")
    return ap.parse_args()

def main():
    args = parse_args()
    features = {k: True for k in args.features.split(",") if k}
    generate_sigrids(
        input_dir=args.input,
        masks_dir=args.masks,
        output_dir=args.output,
        dataset_name=args.dataset,
        n_segments=args.n_segments,
        compactness=args.compactness,
        grid=args.grid,
        features=features,
    )

if __name__ == "__main__":
    main()