import torch
import numpy as np
from pathlib import Path
from typing import Any, Dict, List

from utils import image_to_tensor, run_model_inference
import vig_2


def load_model(model_variant: str, checkpoint_path: str, device: torch.device):
    model_fn = getattr(vig_2, model_variant)
    model = model_fn()

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Common checkpoint formats
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Remove possible "module." prefix from DataParallel checkpoints
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        cleaned_state_dict[new_key] = v

    missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)

    print("\n[Checkpoint loading]")
    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")
    if missing:
        print("First 10 missing keys:", missing[:10])
    if unexpected:
        print("First 10 unexpected keys:", unexpected[:10])

    model.to(device)
    model.eval()
    return model


def adapt_feature_tensor(x: torch.Tensor) -> torch.Tensor:
    """
    Convert a raw per-layer feature tensor into [N, D] for analysis.

    Common possible shapes:
    - [1, C, H, W]   -> [H*W, C]
    - [C, H, W]      -> [H*W, C]
    - [1, C, N, 1]   -> [N, C]
    - [1, N, C]      -> [N, C]
    - [N, C]         -> [N, C]
    """
    if not torch.is_tensor(x):
        raise TypeError(f"Expected torch.Tensor, got {type(x)}")

    x = x.detach().cpu()

    if x.ndim == 4:
        # Case: [B, C, H, W]
        if x.shape[0] == 1:
            if x.shape[-1] == 1:
                # [1, C, N, 1] -> [N, C]
                x = x.squeeze(0).squeeze(-1).transpose(0, 1).contiguous()
            else:
                # [1, C, H, W] -> [H*W, C]
                x = x.squeeze(0).permute(1, 2, 0).contiguous()
                x = x.view(-1, x.shape[-1])
        else:
            raise ValueError(f"Batch size > 1 not supported yet, shape={tuple(x.shape)}")

    elif x.ndim == 3:
        # Case: [C, H, W] -> [H*W, C]
        if x.shape[0] < x.shape[-1] and x.shape[1] == x.shape[2]:
            x = x.permute(1, 2, 0).contiguous()
            x = x.view(-1, x.shape[-1])
        # Case: [1, N, C] -> [N, C]
        elif x.shape[0] == 1:
            x = x.squeeze(0).contiguous()
        else:
            raise ValueError(f"Unhandled 3D feature shape: {tuple(x.shape)}")

    elif x.ndim == 2:
        # Its already [N, D]
        pass

    else:
        raise ValueError(f"Unhandled feature tensor shape: {tuple(x.shape)}")

    return x


def adapt_edge_tensor(x: Any) -> torch.Tensor:
    """
    Try to standardize raw graph connectivity into a CPU tensor.

    We do not force a single graph format yet; we only detach and save it.
    Later, after inspecting shapes, we can define a stricter adapter.

    Possible formats in graph models include:
    - [2, E]
    - [B, 2, E]
    - [2, N, k]
    - [B, 2, N, k]
    """
    if not torch.is_tensor(x):
        raise TypeError(f"Expected torch.Tensor for edge structure, got {type(x)}")

    return x.detach().cpu()



def summarize_adapted_feature(feat: torch.Tensor) -> Dict[str, Any]:
    # feat: [N, D]
    feat = feat.float()
    norms = torch.norm(feat, dim=1)

    pooled_mean = feat.mean(dim=0)
    pooled_max = feat.max(dim=0).values

    return {
        "num_nodes": int(feat.shape[0]),
        "feat_dim": int(feat.shape[1]),
        "feat_mean_norm": float(norms.mean().item()),
        "feat_std_norm": float(norms.std().item()),
        "feat_global_mean": float(feat.mean().item()),
        "feat_global_std": float(feat.std().item()),
        # Optional compact embeddings for later
        "pooled_mean": pooled_mean.cpu().numpy().astype(np.float16),
        "pooled_max": pooled_max.cpu().numpy().astype(np.float16),
    }

def summarize_adapted_edges(edge: torch.Tensor, num_nodes: int) -> Dict[str, Any]:
    edge = edge.cpu()

    summary = {
        "edge_shape": list(edge.shape),
        "num_edges": None,
        "avg_degree": None,
    }

    if edge.ndim == 2 and edge.shape[0] == 2:
        e = int(edge.shape[1])
        summary["num_edges"] = e
        summary["avg_degree"] = float(e / max(num_nodes, 1))
    elif edge.ndim == 3 and edge.shape[0] == 2:
        # ie: [2, N, k]
        n = int(edge.shape[1])
        k = int(edge.shape[2])
        e = n * k
        summary["num_edges"] = e
        summary["avg_degree"] = float(k)

    return summary


