import torch
import numpy as np
from typing import Any, Dict, List
# import repitl.matrix_itl as itl


# def compute_entropy(hidden_states, alpha=1.0, normalizations=["maxEntropy"], eps=1e-12):
#     """
#     Compute matrix-based entropy across ViG layers.

#     Args:
#         hidden_states: torch.Tensor with shape [L, N, D]
#             L = layers
#             N = nodes / patches
#             D = feature channels
#     """
#     L, N, D = hidden_states.shape

#     entropies = []

#     for layer_idx in range(L):
#         z = hidden_states[layer_idx].double()  # [N, D]

#         # Optional but recommended
#         z = z - z.mean(dim=0, keepdim=True)

#         # Use smaller matrix for efficiency.
#         # Non-zero eigenvalues of Z Z^T and Z^T Z are the same.
#         if N > D:
#             mat = z.T @ z   # [D, D]
#         else:
#             mat = z @ z.T   # [N, N]

#         mat = 0.5 * (mat + mat.T)

#         trace = torch.trace(mat)
#         if trace <= eps:
#             entropies.append(np.nan)
#             continue

#         mat = mat / trace

#         try:
#             ent = itl.matrixAlphaEntropy(mat, alpha=alpha).item()
#         except Exception:
#             ent = np.nan

#         entropies.append(ent)

#     return { norm: [entropy_normalization(x, norm, N, D) for x in entropies] for norm in normalizations }


def compute_entropy_single_layer(
    z: torch.Tensor,
    alpha: float = 1.0,
    eps: float = 1e-12,
) -> Dict[str, Any]:
    """
    Matrix-based entropy for one ViG layer.

    Args:
        z: Tensor [N, D] (nodes x features)
        alpha: entropy order (1.0 = Shannon)
        eps: numerical stability
        center: whether to remove feature mean

    """

    z = z.double() # better precision

    # Center features (Gram matrix computes: Similarity = variation between nodes)
    z = z - z.mean(dim=0, keepdim=True)

    N, D = z.shape

    # Build Gram /covariance matrix
    # use smaller matrix for efficiency, cause they both have same non-zero eigenvalues (?)
    if N > D:
        mat = z.T @ z      # [D, D]
    else:
        mat = z @ z.T      # [N, N]

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
    # Renyi entropy
    else:
        entropy = torch.log(torch.sum(probs ** alpha)) / (1.0 - alpha)

    rank = int((eigvals > eps).sum().item())

    # Maximum entropy when all eigenvalues are equal
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

