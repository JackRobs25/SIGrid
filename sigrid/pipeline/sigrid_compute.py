# sigrid/pipeline/sigrid_compute.py
"""
Temporary stub. Replace these classes/functions with your real implementations
from compute_sigrids.py after you paste that file.
"""

class MergingImageProcessor:
    def __init__(self, img_path, n_segments, compactness, merge_threshold):
        self.img_path = img_path
        self.n_segments = n_segments
        self.compactness = compactness
        self.merge_threshold = merge_threshold
        self.img = None
        self.grid = None
        self.segments_slic = None
        self.sp_label_grid = None

    def compute_superpixels_and_merge(self, fast=True):
        raise NotImplementedError("Paste real MergingImageProcessor from compute_sigrids.py")

    def create_grid(self, n, optimized=True, **feature_flags):
        raise NotImplementedError("Paste real create_grid from compute_sigrids.py")

class MaskProcessor:
    def __init__(self, mask_path, n):
        self.mask_path = mask_path
        self.n = n
        self.mask = None
        self.grid = None

    def set_superpixels(self, slic):
        raise NotImplementedError("Paste real set_superpixels from compute_sigrids.py")

    def create_grid(self):
        raise NotImplementedError("Paste real create_grid from compute_sigrids.py")

def generate_sigrids(input_dir, masks_dir, output_dir, dataset_name, n_segments, compactness, grid, features):
    """
    Optional 'one-shot' generator for reviewers.
    You can implement this by iterating the dataset and saving its cache.
    """
    raise NotImplementedError("Implement using your pipeline once compute_sigrids.py is provided.")