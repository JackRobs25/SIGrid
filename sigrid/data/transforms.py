# sigrid/data/transforms.py
import albumentations as A
from albumentations.pytorch import ToTensorV2

__all__ = ["setup_transforms"]

def setup_transforms(channels: int, normalize: str = "none", stats=None):
    """
    normalize:
      - "none": only ToTensorV2()
      - "unit255": Normalize(mean=[0]*C, std=[1]*C, max_pixel_value=255)
      - "zscore": requires 'stats' = {"mean": [...], "std": [...]}
    """
    spatial = A.Compose([
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
    ])

    if normalize == "unit255":
        norm = A.Compose([
            A.Normalize(mean=[0.0]*channels, std=[1.0]*channels, max_pixel_value=255.0),
            ToTensorV2(),
        ])
    elif normalize == "zscore" and stats is not None:
        norm = A.Compose([
            A.Normalize(mean=stats["mean"], std=stats["std"]),
            ToTensorV2(),
        ])
    else:
        norm = A.Compose([ToTensorV2()])

    return spatial, norm