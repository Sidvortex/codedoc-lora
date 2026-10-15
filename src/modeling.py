"""Model / tokenizer loading shared by train, evaluate and infer."""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def bf16_ok() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def compute_dtype() -> torch.dtype:
    # Kaggle T4 / P100 do not support bf16 -> fp16.
    if bf16_ok():
        return torch.bfloat16
    return torch.float16 if torch.cuda.is_available() else torch.float32


def load_tokenizer(name: str):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # needed for batched generation
    return tok


def load_model(name: str, load_in_4bit: bool = True, adapter: str | None = None):
    kwargs = {"dtype": compute_dtype()}
    if torch.cuda.is_available():
        kwargs["device_map"] = {"": 0}
        if load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype(),
            )
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model
