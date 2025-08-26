# scripts/save_data.py
import os
import json
import uuid
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Dict
import torch
import numpy as np

__all__ = ["save_experiment"]

def _to_serializable(x: Any):
    """Convert tensors/arrays to plain Python so json.dump won't choke."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return [_to_serializable(v) for v in x]
    if isinstance(x, (dict,)):
        return {k: _to_serializable(v) for k, v in x.items()}
    return x

def save_experiment(state, workdir: str, model_attr: str = "network") -> Dict[str, str]:
    """
    Save model weights and experiment metadata.

    Parameters
    ----------
    state : object or SimpleNamespace
        Must expose attributes used below (dataset, model, downsample, n_segments, compactness,
        SIGrid_channels, dim, avg_color, area, width, height, compac, solidity, eccentricity, hu,
        learning_rate, num_epochs, batch_size, training_* metrics, testing_* metrics, test_* metrics,
        test_losses, train_losses, start_time, end_time, flops, and a model under `model_attr`).
    workdir : str
        Directory to write files to (will be created if it doesn't exist).
    model_attr : str
        Attribute name on `state` that holds the torch.nn.Module (default: "network").

    Returns
    -------
    Dict[str, str]
        Paths of the saved JSON and model weights.
    """
    os.makedirs(workdir, exist_ok=True)

    # pull the model
    model = getattr(state, model_attr, None)
    if model is None:
        raise AttributeError(f"`state` has no attribute '{model_attr}' (expected your nn.Module there).")

    # timestamp + uid
    unique_id = uuid.uuid4().hex
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # file names
    n_segments = getattr(state, "n_segments", "NA")
    compactness = getattr(state, "compactness", "NA")
    channels = getattr(state, "SIGrid_channels", "NA")
    dataset = getattr(state, "dataset", "DATASET")

    json_name = f"experiment_results_{timestamp}_{unique_id}_{n_segments}_{compactness}_{channels}.json"
    model_name = f"model_{dataset}_{timestamp}_{unique_id}.pth"

    json_path = os.path.join(workdir, json_name)
    model_path = os.path.join(workdir, model_name)

    # build the experiment dict mirroring your original keys
    exp = {
        'Dataset': getattr(state, 'dataset', None),
        'Model': getattr(state, 'model', None),
        'Downsample': getattr(state, 'downsample', None),
        'N_segments': n_segments,
        'Compactness': compactness,
        'SIGrid Channels': channels,
        'SIGrid dimensions': getattr(state, 'dim', None),
        'Average Color': getattr(state, 'avg_color', None),
        'Area': getattr(state, 'area', None),
        'Width': getattr(state, 'width', None),
        'Height': getattr(state, 'height', None),
        'Compactness (shape descriptor)': getattr(state, 'compac', None),
        'Solidity': getattr(state, 'solidity', None),
        'Eccentricity': getattr(state, 'eccentricity', None),
        'Hu Moments: ': getattr(state, 'hu', None),
        'Learning Rate': getattr(state, 'learning_rate', None),
        'Number of Epochs': getattr(state, 'num_epochs', None),
        'Batch Size': getattr(state, 'batch_size', None),

        'Training Cell IoU': getattr(state, 'training_cell_iou', None),
        'Training Cell Accuracies': getattr(state, 'training_cell_accuracies', None),
        'Training Pixel IoU': getattr(state, 'training_pixel_accuracies', None),
        'Training Pixel Accuracies': getattr(state, 'training_pixel_iou', None),

        'Testing Cell IoU': getattr(state, 'testing_cell_iou', None),
        'Testing Pixel Accuracies': getattr(state, 'testing_pixel_accuracies', None),
        'Testing Cell IoU (dup)': getattr(state, 'testing_pixel_iou', None),  # kept to mirror your dict
        'Testing Pixel Accuracies (dup)': getattr(state, 'testing_pixel_accuracies', None),

        'Test Cell Iou': getattr(state, 'test_cell_iou', None),
        'Test Cell F Score': getattr(state, 'test_cell_f', None),
        'Test Cell Accuracy': getattr(state, 'test_cell_accuracy', None),
        'Test Pixel Iou': getattr(state, 'test_pixel_iou', None),
        'Test Pixel F Score': getattr(state, 'test_pixel_f', None),
        'Test Pixel Accuracy': getattr(state, 'test_pixel_accuracy', None),

        'Test losses': getattr(state, 'test_losses', None),
        'Training losses': getattr(state, 'train_losses', None),

        'Timestamp': str(datetime.now()),
        'Training Duration': (getattr(state, 'end_time', 0) - getattr(state, 'start_time', 0)),
        'Flops': getattr(state, 'flops', None),
        # 'Macs': getattr(state, 'macs', None),
        # 'Params': getattr(state, 'params', None),
    }

    # save model weights
    torch.save(model.state_dict(), model_path)
    print(f"Model parameters saved to {model_path}")

    # JSON-serialize (convert tensors/arrays first)
    with open(json_path, "w") as f:
        json.dump(_to_serializable(exp), f, indent=4)
    print(f"Experiment data saved to {json_path}")

    return {"json": json_path, "model": model_path}