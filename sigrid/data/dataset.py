# sigrid/data/dataset.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from skimage import io

# Import pipeline processors (you'll paste real implementations later)
from sigrid.pipeline.sigrid_compute import MergingImageProcessor, MaskProcessor

__all__ = ["SIGridDataset"]

class SIGridDataset(Dataset):
    """
    Precomputes & caches SIGrid + mask grids, then serves them as tensors.

    Returns per item:
      (sig_tensor[C,H,W], sig_mask_tensor[H,W], slic_2d[np.ndarray], sp_label_grid_2d[np.ndarray], index)
    """
    def __init__(
        self,
        dataset_name: str,
        n_segments: int,
        compactness: float,
        grid_size: int,
        image_dir: str,
        mask_dir: str,
        spatial_transform,
        norm_transform,
        features: dict,
        cache_root: str | None = None,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.n_segments = n_segments
        self.compactness = compactness
        self.grid_size = grid_size
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.spatial_transform = spatial_transform
        self.norm_transform = norm_transform

        # Feature flags (avg, area, width, height, compac, solidity, eccentricity, hu)
        self.features = {k: bool(v) for k, v in features.items()}

        self.mode = "train" if "train" in image_dir else "test"
        self.images = self._get_all_images(self.image_dir)

        # Build cache root
        default_root = os.path.join(".", "artifacts", self.dataset_name, "cache")
        self.cache_root = cache_root or os.getenv("SIGRID_CACHE", default_root)
        os.makedirs(self.cache_root, exist_ok=True)

        # Channel suffix for cache key
        channel_suffix = "_".join([k for k, v in self.features.items() if v]) or "none"
        suffix = f"{self.n_segments}_{self.compactness}_{self.grid_size}_{self.mode}_{channel_suffix}"

        self.sig_path = os.path.join(self.cache_root, f"sig_list_{suffix}.pt")
        self.sig_mask_path = os.path.join(self.cache_root, f"sig_mask_list_{suffix}.pt")
        self.slic_path = os.path.join(self.cache_root, f"slic_list_{suffix}.npz")
        self.map_path = os.path.join(self.cache_root, f"map_list_{suffix}.npz")

        # Try load cache
        all_exist = all(os.path.exists(p) for p in [self.sig_path, self.sig_mask_path, self.slic_path, self.map_path])
        if all_exist:
            print(f"✅ Loading cached SIGrid data: {self.sig_path}")
            self.sig_list = torch.load(self.sig_path)
            self.sig_mask_list = torch.load(self.sig_mask_path)

            slic_npz = np.load(self.slic_path)
            map_npz = np.load(self.map_path)
            self.slic_list = [slic_npz[k] for k in slic_npz.files]
            self.map_list = [map_npz[k] for k in map_npz.files]
        else:
            print("🚧 Cache not found. Computing SIGrids...")
            self._compute_and_cache()

    def _get_all_images(self, directory: str):
        image_paths = []
        if self.dataset_name == "CUB":
            for root, _, files in os.walk(directory):
                for file in files:
                    if file.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".npy")):
                        image_paths.append(os.path.join(root, file))
        else:
            for file in os.listdir(directory):
                if file.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".npy")):
                    image_paths.append(os.path.join(directory, file))
        return image_paths

    def _mask_path_for(self, img_path: str):
        mask_path = img_path
        if "train_images" in mask_path:
            mask_path = mask_path.replace("train_images", "train_masks")
        elif "test_images" in mask_path:
            mask_path = mask_path.replace("test_images", "test_masks")
        return mask_path.replace(".jpg", ".png")

    def _compute_and_cache(self):
        self.sig_list, self.sig_mask_list, self.slic_list, self.map_list = [], [], [], []
        total_superpixels = 0
        superpixels_left_behind = 0

        for img_path in tqdm(self.images, desc="Precomputing SIGrid data"):
            mask_path = self._mask_path_for(img_path)

            image = io.imread(img_path)
            mask = io.imread(mask_path)

            aug = self.spatial_transform(image=image, mask=mask)
            transformed_img = aug["image"]
            transformed_mask = aug["mask"]

            height, width = transformed_img.shape[:2]
            max_dim = max(height, width)

            # Build SIGrid via pipeline
            proc = MergingImageProcessor(
                image_path=img_path,
                n_segments=self.n_segments,
                compactness=self.compactness,
                merge_threshold=(max_dim / 80.0),
            )
            proc.img = transformed_img
            proc.compute_superpixels_and_merge(fast=True)
            discarded = proc.create_grid(
                self.grid_size,
                optimized=True,
                use_avg_col=self.features.get("avg", False),
                use_height=self.features.get("height", False),
                use_width=self.features.get("width", False),
                use_area=self.features.get("area", False),
                use_compac=self.features.get("compac", False),
                use_eccentricity=self.features.get("eccentricity", False),
                use_solidity=self.features.get("solidity", False),
                use_hu=self.features.get("hu", False),
            )
            sig = proc.grid
            slic = proc.segments_slic
            sp_map = proc.sp_label_grid

            total_superpixels += len(slic)
            superpixels_left_behind += discarded

            # Mask grid
            mp = MaskProcessor(mask_path, self.grid_size)
            mp.mask = transformed_mask
            mp.set_superpixels(slic)
            mp.create_grid()
            sig_mask = mp.grid

            # Normalize & to tensor
            norm = self.norm_transform(image=sig, mask=sig_mask)
            normed_sig = norm["image"]
            normed_mask = norm["mask"]

            self.sig_list.append(normed_sig)
            self.sig_mask_list.append(normed_mask)
            self.slic_list.append(slic)
            self.map_list.append(sp_map)

        print("verify SIGrid shape of first sample: ", self.sig_list[0].shape)
        print("Total number of superpixels: ", total_superpixels)
        if total_superpixels > 0:
            print("Superpixels left behind (%): ", 100.0 * superpixels_left_behind / total_superpixels)

        # Save cache
        torch.save(self.sig_list, self.sig_path)
        torch.save(self.sig_mask_list, self.sig_mask_path)
        np.savez(self.slic_path, *self.slic_list)
        np.savez(self.map_path, *self.map_list)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return (
            self.sig_list[index],
            self.sig_mask_list[index],
            self.slic_list[index],
            self.map_list[index],
            index,
        )