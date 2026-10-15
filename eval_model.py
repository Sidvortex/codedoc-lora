"""Compare base model vs. fine-tuned adapter on held-out functions.

    python eval_model.py                                 # base + adapter from config
    python eval_model.py --only base
    python eval_model.py --adapter outputs/codedoc-qwen1.5b/adapter

Writes results.json (ROUGE-L, BLEU, chrF, avg length) and samples.md (side by side).
"""
from __future__ import annotations

import json
from pathlib import Path

import sacrebleu
import torch
from datasets import Dataset
from rouge_score import rouge_scorer
from tqdm import tqdm

from src.config import base_parser, load_config
from src.data import SYSTEM_PROMPT, build_user_prompt, clean_docstring, load_pairs
from src.modeling import load_model, load_tokenizer


@torch.no_grad()
def generate(model, tok, codes: list[str], max_new_tokens: int, batch_size: int) -> list[str]:
    outs = []
    for i in tqdm(range(0, len(codes), batch_size), desc="generating"):
        batch = codes[i : i + batch_size]
        prompts = [
            tok.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(c)},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for c in batch
        ]
        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
        new = gen[:, enc["input_ids"].shape[1] :]
        outs += [clean_docstring(t) for t in tok.batch_decode(new, skip_special_tokens=True)]
    return outs


def score(preds: list[str], refs: list[str]) -> dict:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge = sum(scorer.score(r, p)["rougeL"].fmeasure for p, r in zip(preds, refs)) / len(refs)
    return {
        "rougeL": round(100 * rouge, 2),
        "bleu": round(sacrebleu.corpus_bleu(preds, [refs]).score, 2),
        "chrf": round(sacrebleu.corpus_chrf(preds, [refs]).score, 2),
        "avg_words": round(sum(len(p.split()) for p in preds) / len(preds), 1),
    }


def main() -> None:
    parser = base_parser(__doc__, "configs/qwen_coder_1p5b.yaml")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--only", choices=["base", "tuned"], default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    out = Path(cfg["train"]["output_dir"])
    adapter = args.adapter or str(out / "adapter")

    pairs_file = out / "eval_pairs.jsonl"
    if pairs_file.exists():
        eval_ds = Dataset.from_json(str(pairs_file))
    else:
        _, eval_ds = load_pairs(cfg["data"])
    if args.limit:
        eval_ds = eval_ds.select(range(min(args.limit, len(eval_ds))))
    codes, refs = eval_ds["code"], eval_ds["docstring"]

    tok = load_tokenizer(cfg["model"].get("tokenizer", cfg["model"]["name"]))
    results, preds_by = {}, {}
    runs = [("base", None), ("tuned", adapter)]
    if args.only:
        runs = [r for r in runs if r[0] == args.only]

    for label, adp in runs:
        model = load_model(cfg["model"]["name"], cfg["model"]["load_in_4bit"], adapter=adp)
        preds = generate(model, tok, codes, cfg["eval"]["max_new_tokens"], cfg["eval"]["batch_size"])
        results[label] = score(preds, refs)
        preds_by[label] = preds
        print(label, results[label])
        del model
        torch.cuda.empty_cache()

    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2))

    lines = ["# Sample outputs\n"]
    for i in range(min(15, len(codes))):
        lines.append(f"## Example {i + 1}\n\n```python\n{codes[i]}\n```\n")
        lines.append(f"**Reference**\n\n```\n{refs[i]}\n```\n")
        for label, preds in preds_by.items():
            lines.append(f"**{label}**\n\n```\n{preds[i]}\n```\n")
    (out / "samples.md").write_text("\n".join(lines))
    print(f"Wrote {out / 'results.json'} and {out / 'samples.md'}")


if __name__ == "__main__":
    main()
