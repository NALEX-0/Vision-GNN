import torch
import numpy as np
from typing import Any, Dict, List

# Prompt Entropy, LLM paper
def compute_entropy_single_layer(
    z: torch.Tensor,
    alpha: float = 1.0,
    eps: float = 1e-12,
) -> Dict[str, Any]:
    """
    Matrix-based entropy on the per-layer features

    Args:
        z: Tensor [N, D] (nodes x features)
        alpha: entropy order (1.0 = Shannon)
        eps: numerical stability
    """

    z = z.double() # better precision

    # Center features
    z = z - z.mean(dim=0, keepdim=True)

    N, D = z.shape

    # Build Gram /covariance matrix
    # use smaller matrix for efficiency, cause they both have same non-zero eigenvalues (?)
    # if N > D:
    #     mat = z.T @ z      # [D, D]  feature to feature
    # else:
    mat = z @ z.T      # [N, N]  node to node

    # Ensure symmetry
    mat = 0.5 * (mat + mat.T)

    trace = torch.trace(mat)

    # Edge case: zero variance
    if trace <= eps:
        return {
            "entropy": 0.0,
            "normalized_entropy": 0.0,
            "effective_rank": 0.0,
            "rank": 0,
            "num_nodes": N,
            "feature_dim": D,
        }

    mat = mat / trace

    # Eigenvalues (eigenvalue = variance along one direction)
    eigvals = torch.linalg.eigvalsh(mat)
    eigvals = torch.clamp(eigvals, min=0.0)

    total = eigvals.sum()

    # if total <= eps:
    #     return {
    #         "entropy": 0.0,
    #         "normalized_entropy": 0.0,
    #         "effective_rank": 0.0,
    #         "rank": 0,
    #         "num_nodes": N,
    #         "feature_dim": D,
    #     }

    probs = eigvals / total
    probs = probs[probs > eps]

    # Shannon entropy
    if alpha == 1.0:
        entropy = -(probs * torch.log(probs)).sum()
    #
    else:
        entropy = torch.log(torch.sum(probs ** alpha)) / (1.0 - alpha)

    rank = int((eigvals > eps).sum().item())

    # Normalize entropy
    if rank > 1:
        max_entropy = torch.log(torch.tensor(float(rank), device=z.device))
        normalized_entropy = entropy / max_entropy
    else:
        normalized_entropy = torch.tensor(0.0, device=z.device)

    effective_rank = torch.exp(entropy)

    return {
        "entropy": float(entropy.item()),
        "normalized_entropy": float(normalized_entropy.item()),
        "effective_rank": float(effective_rank.item()),
        "rank": rank,
        "num_nodes": N,
        "feature_dim": D,
    }


def format_embedding_table(
    embeddings: torch.Tensor,
    max_rows: int = 20,
    max_cols: int = 12,
) -> str:
    arr = embeddings.detach().cpu().numpy()
    n, d = arr.shape
    row_end = min(n, max_rows)
    col_end = min(d, max_cols)
    header = [f"f{j}" for j in range(col_end)]
    lines = ["      " + "  ".join(f"{h:>8}" for h in header)]

    for i in range(row_end):
        row_values = arr[i, :col_end]
        formatted = "  ".join(f"{v:8.4f}" for v in row_values)
        lines.append(f"{i:>4}  {formatted}")

    if n > row_end:
        lines.append(f"... ({n - row_end} more nodes)")
    if d > col_end:
        lines.append(f"... ({d - col_end} more features)")

    return "\n".join(lines)