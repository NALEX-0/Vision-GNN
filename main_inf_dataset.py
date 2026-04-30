import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from utils import image_to_tensor, run_model_inference
from utils2 import (
    load_model,
    summarize_adapted_feature,
    summarize_adapted_edges,
    adapt_feature_tensor,
    adapt_edge_tensor,
)

from metrics import compute_entropy_single_layer

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_image_list(
    im_path: Optional[str],
    im_dir: Optional[str],
    im_list: Optional[str],
    max_images: Optional[int] = None,
) -> List[str]:
    image_list: List[str] = []

    provided = [x is not None for x in [im_path, im_dir, im_list]]
    if sum(provided) != 1:
        raise ValueError("Provide exactly one of --im_path, --im_dir, or --im_list")

    if im_path is not None:
        p = Path(im_path)
        if not p.exists():
            raise FileNotFoundError(f"Image path not found: {im_path}")
        image_list = [str(p)]

    elif im_dir is not None:
        p = Path(im_dir)
        if not p.exists():
            raise FileNotFoundError(f"Image directory not found: {im_dir}")
        image_list = sorted(
            str(x) for x in p.rglob("*") if x.suffix.lower() in IMAGE_EXTENSIONS
        )

    elif im_list is not None:
        p = Path(im_list)
        if not p.exists():
            raise FileNotFoundError(f"Image list file not found: {im_list}")
        with open(p, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    image_list.append(line)

    if max_images is not None:
        image_list = image_list[:max_images]

    return image_list


def safe_topk_from_logits(logits: torch.Tensor, k: int = 5) -> Dict[str, Any]:
    """
    logits expected shape: [1, C]
    """
    if logits.ndim != 2 or logits.shape[0] != 1:
        raise ValueError(f"Expected logits shape [1, C], got {tuple(logits.shape)}")

    probs = torch.softmax(logits, dim=1)
    topk_prob, topk_idx = torch.topk(probs, k=min(k, probs.shape[1]), dim=1)

    pred_idx = int(topk_idx[0, 0].item())
    pred_prob = float(topk_prob[0, 0].item())

    entropy = float((-(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)[0]).item())

    if probs.shape[1] >= 2:
        margin = float((topk_prob[0, 0] - topk_prob[0, 1]).item())
    else:
        margin = 0.0

    return {
        "pred_top1_idx": pred_idx,
        "pred_top1_prob": pred_prob,
        "pred_top5_idx": topk_idx[0].detach().cpu().tolist(),
        "pred_top5_prob": topk_prob[0].detach().cpu().tolist(),
        "entropy": entropy,
        "margin_top1_top2": margin,
        "logits": logits[0].detach().cpu().to(torch.float16),
    }


def infer_single_image(
    image_path: str,
    model: torch.nn.Module,
    device: torch.device,
) -> Dict[str, Any]:
    image_tensor = image_to_tensor(image_path, device)
    logits, edge_indexes_per_layer, block_features_per_layer = run_model_inference(model, image_tensor)

    pred_info = safe_topk_from_logits(logits, k=5)

    layer_rows: List[Dict[str, Any]] = []

    for layer_idx, (feat, edge) in enumerate(zip(block_features_per_layer, edge_indexes_per_layer)):
        feat_std = adapt_feature_tensor(feat)
        edge_std = adapt_edge_tensor(edge)

        feat_summary = summarize_adapted_feature(feat_std)
        edge_summary = summarize_adapted_edges(edge_std, num_nodes=feat_summary["num_nodes"])
        entropy_summary = compute_entropy_single_layer(feat_std)

        layer_row = {
            "image_path": image_path,
            "layer_index": layer_idx,
            "num_nodes": feat_summary["num_nodes"],
            "feat_dim": feat_summary["feat_dim"],
            "feat_mean_norm": feat_summary["feat_mean_norm"],
            "feat_std_norm": feat_summary["feat_std_norm"],
            "feat_global_mean": feat_summary["feat_global_mean"],
            "feat_global_std": feat_summary["feat_global_std"],
            "edge_shape": edge_summary["edge_shape"],
            "num_edges": edge_summary["num_edges"],
            "avg_degree": edge_summary["avg_degree"],
            "matrix_entropy": entropy_summary["entropy"],
            "matrix_normalized_entropy": entropy_summary["normalized_entropy"],
            "matrix_effective_rank": entropy_summary["effective_rank"],
            "matrix_rank": entropy_summary["rank"],
            # "pooled_mean": feat_summary["pooled_mean"].tolist(),
            # "pooled_max": feat_summary["pooled_max"].tolist(),
        }

        layer_rows.append(layer_row)

    prediction_row = {
        "image_path": image_path,
        "pred_top1_idx": pred_info["pred_top1_idx"],
        "pred_top1_prob": pred_info["pred_top1_prob"],
        "pred_top5_idx": pred_info["pred_top5_idx"],
        "pred_top5_prob": pred_info["pred_top5_prob"],
        "entropy": pred_info["entropy"],
        "margin_top1_top2": pred_info["margin_top1_top2"],
        "logits": pred_info["logits"].tolist(),
    }

    return {
        "prediction_row": prediction_row,
        "layer_rows": layer_rows,
    }


def main(args):
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[Loading model]")
    model = load_model(args.model_variant, args.model_weights, device)

    image_list = load_image_list(
        im_path=args.im_path,
        im_dir=args.im_dir,
        im_list=args.im_list,
        max_images=args.max_images,
    )

    print(f"Total images to process: {len(image_list)}")

    all_predictions: List[Dict[str, Any]] = []
    all_layer_rows: List[Dict[str, Any]] = []

    for idx, image_path in enumerate(image_list, start=1):
        print(f"[{idx}/{len(image_list)}] Processing: {image_path}")
        try:
            result = infer_single_image(
                image_path=image_path,
                model=model,
                device=device,
            )
            all_predictions.append(result["prediction_row"])
            all_layer_rows.extend(result["layer_rows"])
        except Exception as e:
            print(f"[ERROR] Failed on {image_path}: {e}")

    predictions_path = output_dir / "predictions.json"
    layer_metrics_path = output_dir / "layer_metrics.json"

    with open(predictions_path, "w") as f:
        json.dump(all_predictions, f, indent=2)

    with open(layer_metrics_path, "w") as f:
        json.dump(all_layer_rows, f, indent=2)

    print("\n[Done]")
    print(f"Saved predictions to: {predictions_path}")
    print(f"Saved layer metrics to: {layer_metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run ViG inference on one image or a dataset and save compact outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--im_path", type=str, default=None, help="Path to one input image")

    parser.add_argument("--im_dir", type=str, default=None, help="Directory with input images")

    parser.add_argument("--im_list", type=str, default=None, help="Text file with one image path per line")

    parser.add_argument("--model_weights", type=str, required=True, help="Path to model checkpoint")

    parser.add_argument("--model_variant", type=str, default="vig_b_224_gelu", help="Model constructor name from vig_2.py")

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on")

    parser.add_argument("--output_dir", type=str, default="outputs_dataset", help="Directory to save results")

    parser.add_argument("--max_images", type=int, default=None, help="Optional limit on number of images")

    args = parser.parse_args()
    main(args)