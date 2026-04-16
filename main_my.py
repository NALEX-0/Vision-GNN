import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch

from utils import image_to_tensor, run_model_inference
from utils_my import *
import vig_2



def main(args):
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[Loading model]")
    model = load_model(args.model_variant, args.model_weights, device)

    def process_single(image_path: str):
        print(f"\n[Processing image] {image_path}")
        info = run_and_save_image_embeddings(
            image_path=image_path,
            model=model,
            device=device,
            output_dir=str(output_dir),
            save_raw=args.save_raw,
        )
        print(f"Saved summary to {info['json_path']}")
        print(f"Saved embeddings to {info['tensor_path']}")

    image_list = []
    if args.im_list:
        im_list_path = Path(args.im_list)
        if not im_list_path.exists():
            raise FileNotFoundError(f"Image list file not found: {args.im_list}")
        with open(im_list_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    image_list.append(line)
    elif args.im_dir:
        im_dir_path = Path(args.im_dir)
        if not im_dir_path.exists():
            raise FileNotFoundError(f"Image directory not found: {args.im_dir}")
        image_list = sorted([str(p) for p in im_dir_path.glob("**/*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    elif args.im_path:
        image_list = [args.im_path]
    else:
        raise ValueError("Please provide one of --im_path, --im_dir, or --im_list")

    if args.max_images is not None:
        image_list = image_list[: args.max_images]

    print(f"Total images to process: {len(image_list)}")
    for image_path in image_list:
        process_single(image_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract per-layer graph embeddings and edges from ViG model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--im_path", type=str, required=True, help="Path to input image")

    parser.add_argument("--model_weights", type=str, required=True, help="Path to model checkpoint")

    parser.add_argument("--model_variant", type=str, default="vig_b_224_gelu", help="Model constructor name from vig_2.py")

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on: CUDA or CPU")

    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save extracted tensors")

    parser.add_argument("--save_raw", action="store_true", help="Also save raw outputs exactly as returned by the model")

    args = parser.parse_args()
    main(args)