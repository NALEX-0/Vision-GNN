from pathlib import Path

from visualization import plot_matrix_entropy


def main():
    layer_metrics_json = "outputs_imagenet_a/n01531178/0.000013_wheelbarrow _ wheelbarrow_0.95062554/layer_metrics.json"
    output_dir = "outputs_imagenet_a/n01531178/0.000013_wheelbarrow _ wheelbarrow_0.95062554/plots"
    # layer_metrics_json = "outputs_imagenet_ILSVRC2012_val_00000001/layer_metrics.json"
    # output_dir = "outputs_imagenet_ILSVRC2012_val_00000001/plots"

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    summary_df = plot_matrix_entropy(
        layer_metrics_json=layer_metrics_json,
        output_dir=output_dir,
    )

    print("\n[Visualization Complete]")
    print(f"Saved plots to: {output_dir}")


if __name__ == "__main__":
    main()