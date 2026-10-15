# codedoc-lora

QLoRA fine-tune of **Qwen2.5-Coder-1.5B-Instruct** that writes Google-style docstrings for Python functions, trained on CodeSearchNet. Includes a base-vs-tuned evaluation and a CLI that documents a whole `.py` file.

It's built to be the small, local "explain this function" model behind a code-reading tool, so you don't pay API costs per function.

## What's inside

| File | Purpose |
|---|---|
| `train.py` | QLoRA training (4-bit NF4 base + LoRA r=16 on all linear layers), loss on docstring tokens only |
| `eval_model.py` | Base vs fine-tuned on held-out functions: ROUGE-L, BLEU, chrF + side-by-side `samples.md` |
| `infer.py` | Adds docstrings to every undocumented function in a file |
| `merge.py` | Merges the adapter into fp16 weights (for GGUF/llama.cpp export or the Hub) |
| `src/data.py` | CodeSearchNet → (function *without* its docstring, docstring) pairs, using `ast` |
| `configs/qwen_coder_1p5b.yaml` | All hyper-parameters; override anything with `--set key=value` |
| `kaggle_notebook.ipynb` | End-to-end run on a free Kaggle T4 |

## Run it on Kaggle (recommended, free GPU)

1. Push this repo to `github.com/Sidvortex/codedoc-lora`.
2. Kaggle → New Notebook → File → Import notebook → `kaggle_notebook.ipynb`.
3. Settings: **GPU T4 x2**, **Internet on**. Run all.

Runtime on one T4: ≈1.5–2 h for 20k examples × 1 epoch, ~6 GB VRAM.

## Run locally (EndeavourOS, no GPU)

Training needs a GPU, but inference on CPU works with the merged model:

```bash
cd ~/Documents/github/codedoc-lora
pip install --break-system-packages -r requirements.txt
# bitsandbytes needs CUDA: on CPU set model.load_in_4bit=false
python infer.py some_file.py --set model.load_in_4bit=false --adapter outputs/codedoc-qwen1.5b/adapter
```

Copy `outputs/codedoc-qwen1.5b/adapter` from the Kaggle output zip into the same path first.

## Useful overrides

```bash
python train.py --set data.max_train=40000 train.epochs=2          # more data
python train.py --set lora.r=64 lora.alpha=128                      # bigger adapter
python train.py --set model.name=Qwen/Qwen2.5-Coder-3B-Instruct train.batch_size=2 train.grad_accum=8
```

## What to report

- `results.json`: ROUGE-L / BLEU / chrF for base vs tuned. Expect the tuned model to follow the docstring style far more consistently, and to stop wrapping answers in prose or code fences.
- `samples.md`: qualitative examples. Pick 3–4 for a write-up.
- `train_stats.json`: training time and peak VRAM.

N-gram metrics undervalue correct-but-differently-worded docstrings, so also read the samples. An LLM-as-judge pass on 50 samples is a good extension.

## Data

[CodeSearchNet](https://huggingface.co/datasets/code-search-net/code_search_net) (Python subset). Functions longer than 2,500 characters and docstrings shorter than 5 or longer than 120 words are filtered out.
