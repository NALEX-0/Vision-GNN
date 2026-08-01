import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_matrix_entropy(layer_metrics_json: str, output_dir: str):
    output_dir = Path(output_dir)
    if output_dir.name in {"plot", "plots"}:
        plot_dir = output_dir
    else:
        plot_dir = output_dir / "plot"
    plot_dir.mkdir(parents=True, exist_ok=True)

    with open(layer_metrics_json, "r") as f:
        rows = json.load(f)

    if not rows:
        print(f"[Visualization] No layer metrics found in {layer_metrics_json}; skipping plots.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    for col in ["layer_index", "matrix_entropy", "matrix_normalized_entropy", "matrix_effective_rank"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["layer_index"])
    if df.empty:
        print("[Visualization] No valid layer metrics after cleaning; skipping plots.")
        return pd.DataFrame()

    # Average over images per layer
    layer_df = (
        df.groupby("layer_index", sort=True)
        .agg(
            matrix_entropy_mean=("matrix_entropy", "mean"),
            matrix_entropy_std=("matrix_entropy", "std"),
            normalized_entropy_mean=("matrix_normalized_entropy", "mean"),
            normalized_entropy_std=("matrix_normalized_entropy", "std"),
            effective_rank_mean=("matrix_effective_rank", "mean"),
            effective_rank_std=("matrix_effective_rank", "std"),
        )
        .reset_index()
    )

    # Plot raw entropy
    plt.figure()
    plt.errorbar(
        layer_df["layer_index"],
        layer_df["matrix_entropy_mean"],
        yerr=layer_df["matrix_entropy_std"],
        marker="o",
        capsize=3,
    )
    plt.xlabel("Layer")
    plt.ylabel("Matrix Entropy")
    plt.title("Matrix-Based Entropy Across ViG Layers")
    plt.grid(True)
    plt.savefig(plot_dir / "matrix_entropy_by_layer.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Plot normalized entropy
    plt.figure()
    plt.errorbar(
        layer_df["layer_index"],
        layer_df["normalized_entropy_mean"],
        yerr=layer_df["normalized_entropy_std"],
        marker="o",
        capsize=3,
    )
    plt.xlabel("Layer")
    plt.ylabel("Normalized Matrix Entropy")
    plt.title("Normalized Matrix-Based Entropy Across ViG Layers")
    plt.grid(True)
    plt.savefig(plot_dir / "normalized_matrix_entropy_by_layer.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Plot effective rank
    plt.figure()
    plt.errorbar(
        layer_df["layer_index"],
        layer_df["effective_rank_mean"],
        yerr=layer_df["effective_rank_std"],
        marker="o",
        capsize=3,
    )
    plt.xlabel("Layer")
    plt.ylabel("Effective Rank")
    plt.title("Effective Rank Across ViG Layers")
    plt.grid(True)
    plt.savefig(plot_dir / "effective_rank_by_layer.png", dpi=300, bbox_inches="tight")
    plt.close()

    layer_df.to_csv(plot_dir / "layer_entropy_summary.csv", index=False)
    print(f"[Visualization] Saved plots to {plot_dir}")

    return layer_df