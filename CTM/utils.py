import torch
import torch.nn.functional as F
import re
import os
import json
from typing import Optional

# Optional huggingface hub support
try:
    from huggingface_hub import HfApi, hf_hub_download
    _HF_AVAILABLE = True
except Exception:
    HfApi = None
    hf_hub_download = None
    _HF_AVAILABLE = False

def compute_decay(T, params, clamp_lims=(0, 15)):
    """
    This function computes exponential decays for learnable synchronisation
    interactions between pairs of neurons.
    """
    assert len(clamp_lims), 'Clamp lims should be length 2'
    assert type(clamp_lims) == tuple, 'Clamp lims should be tuple'

    indices = torch.arange(T-1, -1, -1, device=params.device).reshape(T, 1).expand(T, params.shape[0])
    out = torch.exp(-indices * torch.clamp(params, clamp_lims[0], clamp_lims[1]).unsqueeze(0))
    return out

def add_coord_dim(x, scaled=True):
    """
    Adds a final dimension to the tensor representing 2D coordinates.

    Args:
        tensor: A PyTorch tensor of shape (B, D, H, W).

    Returns:
        A PyTorch tensor of shape (B, D, H, W, 2) with the last dimension
        representing the 2D coordinates within the HW dimensions.
    """
    B, H, W = x.shape
    # Create coordinate grids
    x_coords = torch.arange(W, device=x.device, dtype=x.dtype).repeat(H, 1)  # Shape (H, W)
    y_coords = torch.arange(H, device=x.device, dtype=x.dtype).unsqueeze(-1).repeat(1, W)  # Shape (H, W)
    if scaled:
        x_coords /= (W-1)
        y_coords /= (H-1)
    # Stack coordinates and expand dimensions
    coords = torch.stack((x_coords, y_coords), dim=-1)  # Shape (H, W, 2)
    coords = coords.unsqueeze(0)  # Shape (1, 1, H, W, 2)
    coords = coords.repeat(B, 1, 1, 1)  # Shape (B, D, H, W, 2)
    return coords

def compute_normalized_entropy(logits, reduction='mean'):
    """
    Calculates the normalized entropy of a PyTorch tensor of logits along the
    final dimension.

    Args:
      logits: A PyTorch tensor of logits.

    Returns:
      A PyTorch tensor containing the normalized entropy values.
    """

    # Apply softmax to get probabilities
    preds = F.softmax(logits, dim=-1)

    # Calculate the log probabilities
    log_preds = torch.log_softmax(logits, dim=-1)

    # Calculate the entropy
    entropy = -torch.sum(preds * log_preds, dim=-1)

    # Calculate the maximum possible entropy
    num_classes = preds.shape[-1]
    max_entropy = torch.log(torch.tensor(num_classes, dtype=torch.float32))

    # Normalize the entropy
    normalized_entropy = entropy / max_entropy
    if len(logits.shape)>2 and reduction == 'mean':
        normalized_entropy = normalized_entropy.flatten(1).mean(-1)

    return normalized_entropy


def get_loss(predictions, certainties, targets, use_most_certain=True):
    """Certainty-based loss used across the repo.

    Returns (loss_tensor, where_most_certain_index_tensor)
    """
    losses = torch.nn.CrossEntropyLoss(reduction='none')(predictions,
                                                         torch.repeat_interleave(targets.unsqueeze(-1), predictions.size(-1), -1))

    loss_index_1 = losses.argmin(dim=1)
    # certainties shape expected (B,2,T)
    loss_index_2 = certainties[:, 1].argmax(-1)
    if not use_most_certain:
        loss_index_2[:] = -1

    batch_indexer = torch.arange(predictions.size(0), device=predictions.device)
    loss_minimum_ce = losses[batch_indexer, loss_index_1].mean()
    loss_selected = losses[batch_indexer, loss_index_2].mean()

    loss = (loss_minimum_ce + loss_selected) / 2
    return loss, loss_index_2


def calculate_accuracy(predictions, targets, where_most_certain):
    """Calculate accuracy at the selected internal tick index per-example."""
    B = predictions.size(0)
    device = predictions.device
    predictions_at_most_certain_internal_tick = predictions.argmax(1)[torch.arange(B, device=device), where_most_certain].detach().cpu().numpy()
    accuracy = (targets.detach().cpu().numpy() == predictions_at_most_certain_internal_tick).mean()
    return accuracy

def reshape_predictions(predictions, prediction_reshaper):
    B, T = predictions.size(0), predictions.size(-1)
    new_shape = [B] + prediction_reshaper + [T]
    rehaped_predictions = predictions.reshape(new_shape)
    return rehaped_predictions

def get_all_log_dirs(root_dir):
    folders = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if any(f.endswith(".pt") for f in filenames):
            folders.append(dirpath)
    return folders

def get_latest_checkpoint(log_dir):
    files = [f for f in os.listdir(log_dir) if re.match(r'checkpoint_\d+\.pt', f)]
    return os.path.join(log_dir, max(files, key=lambda f: int(re.search(r'\d+', f).group()))) if files else None

def get_latest_checkpoint_file(filepath, limit=300000):
    checkpoint_files = get_checkpoint_files(filepath)
    checkpoint_files = [
        f for f in checkpoint_files if int(re.search(r'checkpoint_(\d+)\.pt', f).group(1)) <= limit
    ]
    if not checkpoint_files:
        return None
    return checkpoint_files[-1]

def get_checkpoint_files(filepath):
    regex = r'checkpoint_(\d+)\.pt'
    files = [f for f in os.listdir(filepath) if re.match(regex, f)]
    files = sorted(files, key=lambda f: int(re.search(regex, f).group(1)))
    return [os.path.join(filepath, f) for f in files]

def load_checkpoint(checkpoint_path, device):
    # thin wrapper to keep backwards compatibility
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint


def save_checkpoint_local(checkpoint: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint, path)
    return path


def save_checkpoint_to_hf(local_checkpoint_path: str, repo_id: str, token: Optional[str] = None, commit_message: str = "Add checkpoint") -> str:
    """Upload a local checkpoint file to a Hugging Face repo (model repo).

    Returns the path in the HF repo or raises if huggingface_hub not available.
    """
    if not _HF_AVAILABLE:
        raise RuntimeError('huggingface_hub not available; install it to upload checkpoints')

    api = HfApi()
    # create repo if not exists
    try:
        api.create_repo(repo_id=repo_id, exist_ok=True, repo_type='model', token=token)
    except Exception:
        # ignore errors here; upload may still work if repo exists
        pass

    filename = os.path.basename(local_checkpoint_path)
    api.upload_file(path_or_fileobj=local_checkpoint_path, path_in_repo=filename, repo_id=repo_id, repo_type='model', token=token, commit_message=commit_message)
    return filename


def download_checkpoint_from_hf(repo_id: str, filename: str, local_dir: str = '.', token: Optional[str] = None) -> str:
    """Download a file from a HF model repo using hf_hub_download and return local path."""
    if not _HF_AVAILABLE:
        raise RuntimeError('huggingface_hub not available; install it to download checkpoints')
    out_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type='model', token=token, cache_dir=local_dir)
    return out_path

def get_model_args_from_checkpoint(checkpoint):
    if "args" in checkpoint:
        return(checkpoint["args"])
    else:
        raise ValueError("Checkpoint does not contain saved args.")

def get_accuracy_and_loss_from_checkpoint(checkpoint, device="cpu"):
    training_iteration = checkpoint.get('training_iteration', 0)
    train_losses = checkpoint.get('train_losses', [])
    test_losses = checkpoint.get('test_losses', [])
    train_accuracies = checkpoint.get('train_accuracies_most_certain', [])
    test_accuracies = checkpoint.get('test_accuracies_most_certain', [])
    return training_iteration, train_losses, test_losses, train_accuracies, test_accuracies