# sigrid/train/trainer.py
from typing import Optional
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

__all__ = ["train_one_epoch", "make_loader"]

def _collate_keep_maps(batch):
    # Keeps slic/map as Python lists (no giant stacking)
    sigs, masks, slics, maps, idxs = zip(*batch)
    sigs = torch.stack(sigs, dim=0)
    masks = torch.stack(masks, dim=0)  # HxW -> BxHxW
    idxs = torch.tensor(idxs, dtype=torch.long)
    return sigs, masks, list(slics), list(maps), idxs

def make_loader(dataset, batch_size: int = 8, shuffle: bool = True, num_workers: int = 4):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True, collate_fn=_collate_keep_maps)

def train_one_epoch(model, loader, optimizer, device, use_amp: bool = True, scaler: Optional[GradScaler] = None):
    model.train()
    if use_amp and scaler is None:
        scaler = GradScaler()

    total_loss = 0.0
    num_batches = 0

    for sigs, sig_masks, _slics, _maps, _idx in loader:
        sigs = sigs.to(device)
        sig_masks = sig_masks.float().unsqueeze(1).to(device)  # BxHxW -> Bx1xHxW

        with autocast(enabled=use_amp):
            logits = model(sigs)
            M = (sig_masks != -1)
            loss = F.binary_cross_entropy_with_logits(logits[M], sig_masks[M])

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