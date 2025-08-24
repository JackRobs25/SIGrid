# sigrid/train/trainer.py
from typing import Optional, List, Tuple
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from sklearn.metrics import jaccard_score, fbeta_score
import torchvision.transforms.functional as TF

__all__ = ["train_one_epoch", "evaluate", "make_loader"]

def _collate_keep_maps(batch):
    # Keeps slic/map as Python lists (no giant stacking)
    sigs, masks, slics, maps, idxs = zip(*batch)
    sigs = torch.stack(sigs, dim=0)          # BxCxHxW
    masks = torch.stack(masks, dim=0)        # BxHxW
    idxs = torch.tensor(idxs, dtype=torch.long)
    return sigs, masks, list(slics), list(maps), idxs

def make_loader(dataset, batch_size: int = 8, shuffle: bool = True, num_workers: int = 4):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True, collate_fn=_collate_keep_maps)

@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    translate_cells_to_pixels: bool = False,
    save_preds_dir: Optional[str] = None,
    save_first_n: int = 0,
):
    """
    Returns:
      avg_loss,
      cell_iou(%), cell_fbeta(%), cell_acc(%),
      pixel_iou(%|= -1 if disabled), pixel_fbeta(%|= -1), pixel_acc(%|= -1)
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    cell_iou_scores: List[float] = []
    cell_f_scores: List[float] = []
    num_correct_cells = 0
    num_cells = 0

    pixel_iou_scores: List[float] = []
    pixel_f_scores: List[float] = []
    num_correct_pixels = 0
    num_pixels = 0

    # optional save dir
    if save_preds_dir:
        os.makedirs(save_preds_dir, exist_ok=True)
    saved = 0

    for sigs, sig_masks, slics, maps, idxs in loader:
        sigs = sigs.to(device)
        sig_masks = sig_masks.to(device).unsqueeze(1)  # Bx1xHxW
        mask_valid = (sig_masks != -1)

        logits = model(sigs)
        loss = F.binary_cross_entropy_with_logits(logits[mask_valid], sig_masks[mask_valid].float())
        total_loss += float(loss.detach().cpu())
        num_batches += 1

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()

        # cell metrics
        preds_flat = preds[mask_valid].cpu().flatten()
        y_flat = sig_masks[mask_valid].cpu().flatten()
        num_correct_cells += (preds_flat == y_flat).sum().item()
        num_cells += preds_flat.numel()

        # sanity
        assert np.isin(y_flat.numpy(), [0, 1]).all()
        assert np.isin(preds_flat.numpy(), [0, 1]).all()

        cell_iou_scores.append(jaccard_score(y_flat, preds_flat, average="macro"))
        cell_f_scores.append(fbeta_score(y_flat, preds_flat, beta=0.3))

        if translate_cells_to_pixels:
            # translate each sample in batch
            for b in range(sigs.size(0)):
                pred_b = preds[b, 0]  # HxW (grid)
                slic_b = np.asarray(slics[b], dtype=np.int64)
                map_b = np.asarray(maps[b], dtype=np.int64)

                # project grid predictions onto superpixel ids present in map grid
                active = (map_b != -1)
                label_ids = map_b[active]                   # superpixel label per active grid cell
                pred_cells = pred_b[active].to(torch.uint8) # 0/1 per active grid cell

                # Build lookup table: max label + 1
                lut = torch.zeros(int(slic_b.max()) + 1, dtype=torch.uint8, device=pred_cells.device)
                lut[label_ids] = pred_cells * 255

                translated_px = lut[slic_b]                 # HxW, values {0,255}
                bin_px = (translated_px // 255).to(torch.uint8).cpu().numpy()

                # Load GT mask path from dataset using the same rules as dataset._mask_path_for
                img_path = loader.dataset.images[int(idxs[b])]
                gt = _load_gt_mask_like_dataset(img_path, dataset_name=getattr(loader.dataset, "dataset_name", "CUB"))

                # binarize GT (like your code: >175)
                gt_bin = (gt > 175).astype(np.uint8)

                num_correct_pixels += (gt_bin == bin_px).sum().item()
                num_pixels += gt_bin.size

                pixel_iou_scores.append(jaccard_score(gt_bin.flatten(), bin_px.flatten(), average="macro"))
                pixel_f_scores.append(fbeta_score(gt_bin.flatten(), bin_px.flatten(), beta=0.3))

                if save_preds_dir and saved < save_first_n:
                    # save translated prediction and GT
                    TF.to_pil_image(torch.tensor(translated_px).cpu().to(torch.uint8)).save(
                        os.path.join(save_preds_dir, f"pred_{saved}.png")
                    )
                    TF.to_pil_image(torch.tensor(gt_bin * 255).cpu().to(torch.uint8)).save(
                        os.path.join(save_preds_dir, f"gt_{saved}.png")
                    )
                    saved += 1

    avg_loss = total_loss / max(1, num_batches)
    cell_acc = 100.0 * num_correct_cells / max(1, num_cells)
    cell_iou = 100.0 * float(np.mean(cell_iou_scores)) if cell_iou_scores else -1.0
    cell_f = 100.0 * float(np.mean(cell_f_scores)) if cell_f_scores else -1.0

    if translate_cells_to_pixels and pixel_iou_scores:
        pixel_acc = 100.0 * num_correct_pixels / max(1, num_pixels)
        pixel_iou = 100.0 * float(np.mean(pixel_iou_scores))
        pixel_f = 100.0 * float(np.mean(pixel_f_scores))
    else:
        pixel_acc = pixel_iou = pixel_f = -1.0

    model.train()
    return avg_loss, cell_iou, cell_f, cell_acc, pixel_iou, pixel_f, pixel_acc


def _load_gt_mask_like_dataset(img_path: str, dataset_name: str):
    """
    Mirrors the logic in SIGridDataset._mask_path_for so evaluation works with the same structure.
    """
    mask_path = img_path
    if "train_images" in mask_path:
        mask_path = mask_path.replace("train_images", "train_masks")
    elif "test_images" in mask_path:
        mask_path = mask_path.replace("test_images", "test_masks")
    mask_path = mask_path.replace(".jpg", ".png")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Ground truth mask not found: {mask_path}")
    from PIL import Image
    return np.array(Image.open(mask_path).convert("L"))  # grayscale
        

def train_one_epoch(model, loader, optimizer, device, use_amp: bool = True, scaler: Optional[GradScaler] = None):
    model.train()
    if use_amp and scaler is None:
        scaler = GradScaler()

    total_loss = 0.0
    num_batches = 0

    for sigs, sig_masks, _slics, _maps, _idx in loader:
        sigs = sigs.to(device)
        sig_masks = sig_masks.float().unsqueeze(1).to(device)  # BxHxW -> Bx1xHxW
        valid = (sig_masks != -1)

        with autocast(enabled=use_amp):
            logits = model(sigs)
            loss = F.binary_cross_entropy_with_logits(logits[valid], sig_masks[valid])

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.detach().cpu())
        num_batches += 1

    return total_loss / max(1, num_batches)