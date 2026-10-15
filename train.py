"""QLoRA / LoRA fine-tune for docstring generation.

    python train.py --config configs/qwen_coder_1p5b.yaml
    python train.py --set data.max_train=500 train.epochs=1     # quick smoke run

Loss is computed on the docstring (completion) tokens only; the prompt is masked.
Plain transformers Trainer + PEFT, so it runs on any GPU (T4/P100 included).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import Trainer, TrainingArguments

from src.config import base_parser, load_config
from src.data import SYSTEM_PROMPT, build_user_prompt, load_pairs
from src.modeling import bf16_ok, load_model, load_tokenizer


def make_tokenize_fn(tok, max_length: int):
    def tokenize(ex: dict) -> dict:
        prompt = tok.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(ex["code"])},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        c_ids = tok(ex["docstring"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        # keep the whole completion; trim the prompt from the left if too long
        if len(p_ids) + len(c_ids) > max_length:
            p_ids = p_ids[-max(1, max_length - len(c_ids)) :]
        ids = (p_ids + c_ids)[:max_length]
        labels = ([-100] * len(p_ids) + c_ids)[:max_length]
        return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}

    return tokenize


class PadCollator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: list[dict]) -> dict:
        n = max(len(b["input_ids"]) for b in batch)

        def pad(key, value):
            return torch.tensor([b[key] + [value] * (n - len(b[key])) for b in batch])

        return {
            "input_ids": pad("input_ids", self.pad_id),
            "attention_mask": pad("attention_mask", 0),
            "labels": pad("labels", -100),
        }


def main() -> None:
    args = base_parser(__doc__, "configs/qwen_coder_1p5b.yaml").parse_args()
    cfg = load_config(args.config, args.set)
    mcfg, lcfg, dcfg, tcfg = cfg["model"], cfg["lora"], cfg["data"], cfg["train"]
    out = Path(tcfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    print("Loading data ...")
    train_ds, eval_ds = load_pairs(dcfg)
    print(f"train={len(train_ds)} eval={len(eval_ds)}")
    eval_ds.to_json(str(out / "eval_pairs.jsonl"))  # eval_model.py reuses the exact same split

    tok = load_tokenizer(mcfg["name"])
    tok_fn = make_tokenize_fn(tok, tcfg["max_length"])
    train_tok = train_ds.map(tok_fn, remove_columns=train_ds.column_names)
    # small held-out slice for eval loss during training
    eval_tok = eval_ds.select(range(min(100, len(eval_ds)))).map(tok_fn, remove_columns=eval_ds.column_names)

    model = load_model(mcfg["name"], load_in_4bit=mcfg["load_in_4bit"])
    model.train()
    model.config.use_cache = False
    if mcfg["load_in_4bit"]:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    model = get_peft_model(model, LoraConfig(
        r=lcfg["r"],
        lora_alpha=lcfg["alpha"],
        lora_dropout=lcfg["dropout"],
        target_modules=lcfg["target_modules"],
        task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=tcfg["epochs"],
        learning_rate=tcfg["lr"],
        per_device_train_batch_size=tcfg["batch_size"],
        per_device_eval_batch_size=tcfg["batch_size"],
        gradient_accumulation_steps=tcfg["grad_accum"],
        warmup_steps=tcfg["warmup_ratio"],  # float < 1 = ratio of total steps
        lr_scheduler_type="cosine",
        fp16=torch.cuda.is_available() and not bf16_ok(),
        bf16=bf16_ok(),
        optim="paged_adamw_8bit" if mcfg["load_in_4bit"] else "adamw_torch",
        logging_steps=tcfg["logging_steps"],
        eval_strategy="steps",
        eval_steps=tcfg["eval_steps"],
        save_strategy="steps",
        save_steps=tcfg["save_steps"],
        save_total_limit=2,
        report_to="none",
        seed=dcfg["seed"],
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        data_collator=PadCollator(tok.pad_token_id),
    )

    t0 = time.time()
    trainer.train()
    minutes = (time.time() - t0) / 60

    adapter_dir = out / "adapter"
    model.save_pretrained(str(adapter_dir))
    tok.save_pretrained(str(adapter_dir))

    stats = {
        "train_minutes": round(minutes, 1),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)
        if torch.cuda.is_available()
        else None,
        "final_eval": trainer.evaluate(),
        "config": cfg,
    }
    (out / "train_stats.json").write_text(json.dumps(stats, indent=2, default=str))
    print(json.dumps({k: v for k, v in stats.items() if k != "config"}, indent=2, default=str))
    print(f"Adapter saved to {adapter_dir}")


if __name__ == "__main__":
    main()
