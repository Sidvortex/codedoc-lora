"""Merge the LoRA adapter into the base weights (fp16) for standalone use / GGUF export.

    python merge.py --out outputs/codedoc-merged
    python merge.py --out outputs/codedoc-merged --push Sidvortex/codedoc-qwen1.5b   # optional
"""
from __future__ import annotations

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import base_parser, load_config


def main() -> None:
    parser = base_parser(__doc__, "configs/qwen_coder_1p5b.yaml")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--push", default=None, help="Hugging Face repo id to push to")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    adapter = args.adapter or f"{cfg['train']['output_dir']}/adapter"

    # Merge on top of the full-precision base, never the 4-bit one.
    base = AutoModelForCausalLM.from_pretrained(cfg["model"]["name"], dtype=torch.float16)
    merged = PeftModel.from_pretrained(base, adapter).merge_and_unload()
    tok = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    merged.save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)
    print(f"Merged model saved to {args.out}")
    if args.push:
        merged.push_to_hub(args.push)
        tok.push_to_hub(args.push)
        print(f"Pushed to https://huggingface.co/{args.push}")


if __name__ == "__main__":
    main()
