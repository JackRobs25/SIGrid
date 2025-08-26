# sigrid/pipeline/sigrid_compute.py
import os
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.segmentation import slic, mark_boundaries, relabel_sequential
from skimage.measure import regionprops, regionprops_table
from scipy.ndimage import center_of_mass
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

import torch
import torch.nn.functional as F
from torch_scatter import scatter
from fast_slic import Slic as FastSLIC

__all__ = [
    "hu_moments_from_labels_pytorch",
    "log_hu",
    "MergingImageProcessor",
    "MaskProcessor",
    "compute_fixed_n_for_dataset",
    "generate_sigrids",
]

def hu_moments_from_labels_pytorch(labels: torch.Tensor, eps=1e-12):
    """
    Compute 7 Hu moments per labeled region using only PyTorch scatter_add_.

    Args:
        labels: (H, W) int tensor of region IDs (arbitrary integers, no background required).
        eps:    numerical stability term.

    Returns:
        hu:       (K, 7) tensor of Hu invariants (per region, in the order of 'orig_ids').
        orig_ids: (K,) tensor of original label IDs corresponding to rows in 'hu'.
        centroids:(K, 2) tensor with (x̄, ȳ) per region.
    """
    assert labels.dim() == 2 and labels.dtype in (torch.int32, torch.int64), \
        "labels must be (H,W) int tensor"

    H, W = labels.shape
    device = labels.device

    # Make labels contiguous [0..K-1] while remembering original IDs
    orig_ids, inv = torch.unique(labels, sorted=True, return_inverse=True)
    K = orig_ids.numel()
    inv = inv.view(-1)  # (H*W,)

    # Pixel coordinates
    y, x = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )
    x = x.flatten()
    y = y.flatten()

    # Helper: 1D scatter-add into size-(K,) accumulator
    def scatter_sum(values):
        out = torch.zeros(K, device=device, dtype=torch.float32)
        out.scatter_add_(0, inv, values)
        return out

    ones = torch.ones_like(x, dtype=torch.float32)

    # Raw moments for centroids
    m00 = scatter_sum(ones)      # ∑ 1
    m10 = scatter_sum(x)         # ∑ x
    m01 = scatter_sum(y)         # ∑ y

    xbar = m10 / (m00 + eps)
    ybar = m01 / (m00 + eps)

    # Centered coordinates per pixel (broadcast via inv)
    xc = x - xbar[inv]
    yc = y - ybar[inv]

    # Central moments μ_pq = ∑(xc^p yc^q)
    def mu(p, q):
        vals = (xc**p) * (yc**q)
        return scatter_sum(vals)

    mu20 = mu(2,0); mu02 = mu(0,2); mu11 = mu(1,1)
    mu30 = mu(3,0); mu03 = mu(0,3); mu12 = mu(1,2); mu21 = mu(2,1)

    # Normalized central moments η_pq = μ_pq / μ_00^{1+(p+q)/2}; μ_00 == m00
    def eta(mu_pq, p, q):
        gamma = 1.0 + 0.5*(p+q)
        return mu_pq / (m00 + eps).pow(gamma)

    n20 = eta(mu20,2,0); n02 = eta(mu02,0,2); n11 = eta(mu11,1,1)
    n30 = eta(mu30,3,0); n03 = eta(mu03,0,3); n12 = eta(mu12,1,2); n21 = eta(mu21,2,1)

    # Hu's 7 invariants
    phi1 = n20 + n02
    phi2 = (n20 - n02)**2 + 4*(n11**2)
    phi3 = (n30 - 3*n12)**2 + (3*n21 - n03)**2
    phi4 = (n30 + n12)**2 + (n21 + n03)**2
    phi5 = (n30 - 3*n12)*(n30 + n12)*((n30 + n12)**2 - 3*(n21 + n03)**2) \
         + (3*n21 - n03)*(n21 + n03)*(3*(n30 + n12)**2 - (n21 + n03)**2)
    phi6 = (n20 - n02)*((n30 + n12)**2 - (n21 + n03)**2) \
         + 4*n11*(n30 + n12)*(n21 + n03)
    phi7 = (3*n21 - n03)*(n30 + n12)*((n30 + n12)**2 - 3*(n21 + n03)**2) \
         - (n30 - 3*n12)*(n21 + n03)*(3*(n30 + n12)**2 - (n21 + n03)**2)

    hu = torch.stack([phi1, phi2, phi3, phi4, phi5, phi6, phi7], dim=1)
    centroids = torch.stack([xbar, ybar], dim=1)
    return hu, orig_ids, centroids

def log_hu(hu, eps=1e-12):
    return torch.sign(hu) * torch.log10(torch.abs(hu) + eps)

class MergingImageProcessor:
    def __init__(self, image_path, n_segments, compactness, sigma=1, merge_threshold=5):
        self.image_path = image_path
        self.n_segments = n_segments
        self.compactness = compactness
        self.sigma = sigma
        self.merge_threshold = merge_threshold
        self.img = None
        self.segments_slic = None
        self.superpixel_centers = []
        self.grid = None
        self.sp_label_grid = None

    def load_image(self):
        self.img = io.imread(self.image_path)

    def compute_superpixels_and_merge(self, fast=True):
        # If grayscale, convert to 3-channel by stacking
        if len(self.img.shape) == 2 or (len(self.img.shape) == 3 and self.img.shape[2] == 1):
            self.img = np.stack([self.img.squeeze()] * 3, axis=-1)

        if fast:
            img_np = np.ascontiguousarray(self.img)
            slic_engine = FastSLIC(num_components=self.n_segments, compactness=self.compactness)
            segments = slic_engine.iterate(img_np) + 1  # start_label=1 for regionprops
        else:
            segments = slic(self.img, n_segments=self.n_segments, compactness=self.compactness,
                            sigma=self.sigma, start_label=1, channel_axis=-1)

        # Generate list of labels (skip label 0)
        labels = np.arange(1, segments.max() + 1)

        # Efficient centroids for initial labels
        ones = np.ones_like(segments, dtype=np.uint8)
        centers = center_of_mass(ones, labels=segments, index=labels)
        centers = np.array(centers)  # [N, 2]

        # Build adjacency by threshold on centroid distance
        dists = squareform(pdist(centers))
        A = (dists < self.merge_threshold)
        np.fill_diagonal(A, True)

        # Components
        G = csr_matrix(A)
        n_comp, comp = connected_components(G, directed=False)

        # Canonical label per component
        canon = np.zeros(n_comp, dtype=labels.dtype)
        for c in range(n_comp):
            canon[c] = labels[comp == c].min()

        # Map old->merged
        merged_labels = canon[comp]
        label_map = dict(zip(labels, merged_labels))

        # Relabel whole image via unique/inverse
        u, inv = np.unique(segments, return_inverse=True)
        mapped_u = np.array([label_map.get(val, val) for val in u], dtype=u.dtype)
        new_segments = mapped_u[inv].reshape(segments.shape)
        new_segments, _, _ = relabel_sequential(new_segments)

        self.segments_slic = new_segments

        # New centers
        new_labels = np.arange(1, new_segments.max() + 1)
        ones = np.ones_like(new_segments, dtype=np.uint8)
        new_centers = center_of_mass(ones, labels=new_segments, index=new_labels)
        new_centers = np.array(new_centers)
        self.superpixel_centers = [tuple(rc) for rc in new_centers]

    def show_sp_centres(self):
        if self.segments_slic is None:
            raise ValueError("You must compute superpixels first.")
        image_with_boundaries = mark_boundaries(self.img, self.segments_slic, color=(1, 1, 0), mode='thick')
        plt.figure(figsize=(10, 10))
        plt.imshow(image_with_boundaries)
        for center in self.superpixel_centers:
            y, x = center
            plt.plot(x, y, 'ro', markersize=5)
        plt.title('Superpixelation with Superpixel Centers')
        plt.axis('off')
        plt.show()

    def create_grid(
        self,
        fixed_n,
        use_avg_col=True,
        use_height=True,
        use_width=True,
        use_area=True,
        use_compac=True,
        use_eccentricity=True,
        use_solidity=True,
        use_hu=True
    ):
        if self.segments_slic is None:
            raise ValueError("You must compute superpixels first.")

        img_height, img_width = self.img.shape[:2]
        grid_height = np.ceil(img_height / fixed_n)
        grid_width = np.ceil(img_width / fixed_n)

        # Determine total channels
        channel_map = {
            'avg_col': (use_avg_col, 3),
            'area': (use_area, 1),
            'width': (use_width, 1),
            'height': (use_height, 1),
            'compactness': (use_compac, 1),
            'eccentricity': (use_eccentricity, 1),
            'solidity': (use_solidity, 1),
            'hu': (use_hu, 7)
        }
        total_channels = sum(size for flag, size in channel_map.values() if flag)

        self.grid = np.full((fixed_n, fixed_n, total_channels), -1, dtype=np.float32)
        self.sp_label_grid = np.full((fixed_n, fixed_n), -1, dtype=np.int32)

        # Feature slice index map
        feature_slices = {}
        idx = 0
        for name, (flag, size) in channel_map.items():
            if flag:
                feature_slices[name] = slice(idx, idx + size)
                idx += size

        labels_flat = torch.tensor(self.segments_slic.flatten(), dtype=torch.long)
        if use_avg_col:
            img_tensor = torch.tensor(self.img.reshape(-1, 3), dtype=torch.float32)
            avg_colors = scatter(img_tensor, labels_flat, dim=0, reduce='mean')

        if use_width or use_height:
            y_coords, x_coords = torch.meshgrid(
                torch.arange(img_height), torch.arange(img_width), indexing='ij'
            )
            x_coords = x_coords.flatten().to(torch.float32)
            y_coords = y_coords.flatten().to(torch.float32)

        if use_width:
            x_max = scatter(x_coords, labels_flat, dim=0, reduce='max')
            x_min = scatter(x_coords, labels_flat, dim=0, reduce='min')
            widths = (x_max - x_min) / img_width

        if use_height:
            y_max = scatter(y_coords, labels_flat, dim=0, reduce='max')
            y_min = scatter(y_coords, labels_flat, dim=0, reduce='min')
            heights = (y_max - y_min) / img_height

        if use_area:
            ones = torch.ones_like(labels_flat, dtype=torch.float32)
            areas = scatter(ones, labels_flat, dim=0, reduce='sum')
            total_pixels = img_height * img_width
            normalized_areas = areas / total_pixels

        # Label list, centroids
        labels = np.arange(1, self.segments_slic.max() + 1)
        ones_np = np.ones_like(self.segments_slic, dtype=np.uint8)
        centroids = center_of_mass(ones_np, labels=self.segments_slic, index=labels)
        centroids = np.array(centroids)
        centroids_y, centroids_x = centroids[:, 0], centroids[:, 1]

        grid_rows = np.clip((centroids_y // grid_height).astype(int), 0, fixed_n - 1)
        grid_cols = np.clip((centroids_x // grid_width).astype(int), 0, fixed_n - 1)

        self.sp_label_grid[grid_rows, grid_cols] = labels

        coords = np.stack([grid_rows, grid_cols], axis=1)
        _, counts = np.unique(coords, axis=0, return_counts=True)
        discarded = int(np.sum(counts - 1))

        if use_avg_col:
            self.grid[grid_rows, grid_cols, feature_slices['avg_col']] = avg_colors[labels].numpy()
        if use_area:
            self.grid[grid_rows, grid_cols, feature_slices['area']] = normalized_areas[labels].numpy().reshape(-1, 1)
        if use_width:
            self.grid[grid_rows, grid_cols, feature_slices['width']] = widths[labels].numpy().reshape(-1, 1)
        if use_height:
            self.grid[grid_rows, grid_cols, feature_slices['height']] = heights[labels].numpy().reshape(-1, 1)
        if use_hu:
            assignment = torch.tensor(self.segments_slic, dtype=torch.int32)
            hu, orig_ids, _centroids = hu_moments_from_labels_pytorch(assignment)
            self.grid[grid_rows, grid_cols, feature_slices['hu']] = hu.detach().cpu().numpy()

        if use_compac or use_solidity or use_eccentricity:
            regions = regionprops(self.segments_slic)
            for region in regions:
                y, x = region.centroid
                grid_row = min(int(y // grid_height), fixed_n - 1)
                grid_col = min(int(x // grid_width), fixed_n - 1)
                if use_compac:
                    normalised_compactness = ((region.perimeter ** 2) / (4 * np.pi * region.area)) * (region.area / (img_height * img_width)) if region.area > 0 else 0
                    self.grid[grid_row, grid_col, feature_slices['compactness']] = normalised_compactness
                if use_solidity:
                    self.grid[grid_row, grid_col, feature_slices['solidity']] = region.solidity
                if use_eccentricity:
                    self.grid[grid_row, grid_col, feature_slices['eccentricity']] = region.eccentricity

        return discarded

def compute_fixed_n_for_dataset(image_dirs, n_segments, compactness, merge_threshold=5, sigma=1):
    print("\n📏 Computing fixed grid size (n) from all images...")
    max_grid_size = 0
    for split, image_dir in image_dirs.items():
        for root, _, files in os.walk(image_dir):
            for fname in files:
                if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                    continue
                image_path = os.path.join(root, fname)
                try:
                    processor = MergingImageProcessor(image_path, n_segments, compactness, sigma, merge_threshold)
                    processor.load_image()
                    processor.compute_superpixels_and_merge()
                    centers = processor.superpixel_centers
                    height, width = processor.img.shape[:2]

                    def is_valid(n):
                        cell_h = height / n
                        cell_w = width / n
                        seen = set()
                        for y, x in centers:
                            r = int(y // cell_h)
                            c = int(x // cell_w)
                            if (r, c) in seen:
                                return False
                            seen.add((r, c))
                        return True

                    n = math.ceil(math.sqrt(len(centers)))
                    while not is_valid(n):
                        n += 1

                    max_grid_size = max(max_grid_size, n)
                    print(f"✅ {fname}: requires {n}x{n}")

                except Exception as e:
                    print(f"❌ Failed to process {fname}: {e}")

    print(f"\n📐 Final fixed grid size: {max_grid_size}x{max_grid_size}")
    return max_grid_size

class MaskProcessor:
    def __init__(self, mask_path, n):
        self.mask_path = mask_path
        self.mask = None
        self.segments_slic = None
        self.superpixel_centers = []
        self.grid = None
        self.n = n

    def load_mask(self):
        self.mask = io.imread(self.mask_path)

    def set_superpixels(self, segments_slic):
        self.segments_slic = segments_slic
        regions = regionprops(self.segments_slic)
        self.superpixel_centers = [region.centroid for region in regions]

    def show_sp_centres(self):
        if self.segments_slic is None:
            raise ValueError("You must set the superpixels first using set_superpixels().")
        image_with_boundaries = mark_boundaries(self.mask, self.segments_slic, color=(1, 0.1, 0.1), mode='thin')
        mask_height, mask_width = self.mask.shape[:2]
        n = self.n
        grid_height = np.ceil(mask_height / n)
        grid_width = np.ceil(mask_width / n)
        plt.figure(figsize=(10, 10))
        plt.imshow(self.mask, cmap='gray')
        plt.imshow(image_with_boundaries, alpha=0.6)
        for center in self.superpixel_centers:
            y, x = center
            plt.plot(x, y, color='yellow', marker='o', markersize=4, markeredgecolor='black', markeredgewidth=0.5)
        for i in range(n):
            for j in range(n):
                value = self.grid[i, j]
                top_left_y = i * grid_height
                top_left_x = j * grid_width
                if value == 1:
                    edge_color = 'green'; line_width = 3
                elif value == -1:
                    edge_color = 'lightgrey'; line_width = 1
                else:
                    edge_color = 'blue'; line_width = 2
                plt.gca().add_patch(
                    plt.Rectangle((top_left_x, top_left_y), grid_width, grid_height,
                                  fill=False, edgecolor=edge_color, linewidth=line_width)
                )
        plt.title('Mask with Transparent Superpixel Boundaries, Centers, and Highlighted Grid Cells')
        plt.axis('off')
        plt.show()

    def create_grid(self):
        if self.segments_slic is None:
            raise ValueError("You must set the superpixels first using set_superpixels().")
        mask_height, mask_width = self.mask.shape[:2]
        n = self.n
        grid_height = np.ceil(mask_height / n)
        grid_width = np.ceil(mask_width / n)
        self.grid = np.full((n, n), -1, dtype=np.int32)
        foreground_threshold = 0.8
        for region in regionprops(self.segments_slic):
            y, x = region.centroid
            label = region.label
            mask_pixels = self.mask[self.segments_slic == label]
            total_pixels = mask_pixels.size
            foreground_pixels = np.sum(mask_pixels > 127)
            foreground_ratio = foreground_pixels / total_pixels
            is_foreground = foreground_ratio >= foreground_threshold
            grid_row = min(int(y // grid_height), n - 1)
            grid_col = min(int(x // grid_width), n - 1)
            if np.all(self.grid[grid_row, grid_col] == -1):
                self.grid[grid_row, grid_col] = 1 if is_foreground else 0

    def visualize_superpixels_with_grid(self):
        if self.segments_slic is None or self.grid is None:
            raise ValueError("Superpixels must be computed and grid must be created first.")
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        image_with_boundaries = mark_boundaries(self.mask, self.segments_slic, color=(1, 1, 0), mode='thick')
        axes[0].imshow(self.mask, cmap="gray")
        axes[0].imshow(image_with_boundaries, alpha=0.6)
        for center in self.superpixel_centers:
            y, x = center
            axes[0].plot(x, y, 'ro', markersize=5)
        mask_height, mask_width = self.mask.shape[:2]
        grid_height = np.ceil(mask_height / self.n)
        grid_width = np.ceil(mask_width / self.n)
        for i in range(self.n):
            for j in range(self.n):
                value = self.grid[i, j]
                top_left_y = i * grid_height
                top_left_x = j * grid_width
                if value == 1:
                    edge_color, line_width = 'green', 3
                elif value == -1:
                    edge_color, line_width = 'red', 1
                else:
                    edge_color, line_width = 'blue', 1
                axes[0].add_patch(
                    plt.Rectangle((top_left_x, top_left_y), grid_width, grid_height,
                                  fill=False, edgecolor=edge_color, linewidth=line_width)
                )
        axes[0].set_title("Superpixel Boundaries, Centers, and Grid")
        axes[0].axis("off")

        output_image = np.zeros_like(self.mask, dtype=np.uint8)
        for region in regionprops(self.segments_slic):
            label = region.label
            y, x = map(int, region.centroid)
            grid_row = min(int(y // grid_height), self.n - 1)
            grid_col = min(int(x // grid_width), self.n - 1)
            if self.grid[grid_row, grid_col] == 1:
                output_image[self.segments_slic == label] = 255
        axes[1].imshow(output_image, cmap="gray")
        axes[1].set_title("Foreground (White) vs. Background (Black) Superpixels")
        axes[1].axis("off")
        plt.tight_layout()
        plt.show()

    def plot_grid(self):
        if self.grid is None:
            raise ValueError("You must create the grid first.")
        plot_grid = np.zeros(self.grid.shape + (3,), dtype=np.float32)
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                value = self.grid[i, j]
                if value == 1:
                    plot_grid[i, j] = [1, 1, 1]
                elif value == 0:
                    plot_grid[i, j] = [0, 0, 0]
                else:
                    plot_grid[i, j] = [0, 0, 0]  # gray could be [0.5,0.5,0.5]
        plt.figure(figsize=(10, 10))
        plt.imshow(plot_grid)
        plt.title('Mask Grid Representation (Foreground, Background, Empty)')
        plt.axis('off')
        plt.show()

# ---- Convenience helper to match the CLI ----
def generate_sigrids(
    input_dir: str,
    masks_dir: str,
    output_dir: str,
    dataset_name: str,
    n_segments: int,
    compactness: float,
    grid: int,
    features: dict,
):
    """
    One-shot generator invoked by scripts/generate_sigrids.py.
    It builds SIGridDataset-like caches by iterating the images once.
    """
    # We’ll reuse the SIGridDataset logic to ensure identical caching format:
    from sigrid.data.transforms import setup_transforms
    from sigrid.data.dataset import SIGridDataset

    os.makedirs(output_dir, exist_ok=True)
    os.environ["SIGRID_CACHE"] = output_dir

    channels_hint = max(1, sum(3 if k == "avg" and v else 7 if k == "hu" and v else 1 for k, v in features.items()))
    spatial_t, norm_t = setup_transforms(channels=channels_hint, normalize="none")

    # Instantiating the dataset will compute & save caches if missing
    _ = SIGridDataset(
        dataset_name=dataset_name,
        n_segments=n_segments,
        compactness=compactness,
        grid_size=grid,
        image_dir=input_dir,
        mask_dir=masks_dir,
        spatial_transform=spatial_t,
        norm_transform=norm_t,
        features=features,
        cache_root=output_dir,
    )
    print(f"✅ SIGrids written to: {output_dir}")