"""CodeSearchNet (Python) -> (function without docstring, docstring) pairs."""
from __future__ import annotations

import ast
import re
import textwrap

from datasets import Dataset, load_dataset

SYSTEM_PROMPT = (
    "You are a senior Python engineer. Write a clear, accurate Google-style docstring "
    "for the given function. Describe what it does, its arguments, return value and "
    "raised exceptions when relevant. Output only the docstring text, without quotes."
)


def build_user_prompt(code: str) -> str:
    return f"Write the docstring for this function:\n\n```python\n{code.strip()}\n```"


def strip_docstring(source: str) -> str | None:
    """Remove the first docstring from a function's source. Returns None if unparsable."""
    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError:
        return None
    funcs = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if not funcs:
        return None
    fn = funcs[0]
    if not fn.body:
        return None
    first = fn.body[0]
    is_doc = (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    )
    if is_doc:
        fn.body = fn.body[1:] or [ast.Pass()]
    try:
        return ast.unparse(fn)
    except Exception:  # noqa: BLE001 - ast.unparse can fail on odd nodes
        return None


_WS = re.compile(r"\s+")


def clean_docstring(doc: str) -> str:
    doc = textwrap.dedent(doc).strip().strip('"').strip("'").strip()
    # normalise trailing whitespace per line, keep structure
    return "\n".join(line.rstrip() for line in doc.splitlines()).strip()


def to_example(row: dict, cfg: dict) -> dict | None:
    code = row.get("func_code_string") or row.get("whole_func_string") or ""
    doc = row.get("func_documentation_string") or ""
    if not code or not doc:
        return None
    if len(code) > cfg["max_code_chars"]:
        return None
    doc = clean_docstring(doc)
    n_words = len(_WS.split(doc))
    if n_words < cfg["min_doc_words"] or n_words > cfg["max_doc_words"]:
        return None
    body = strip_docstring(code)
    if body is None or len(body.splitlines()) < 2:
        return None
    return {"code": body, "docstring": doc}


def load_pairs(cfg: dict) -> tuple[Dataset, Dataset]:
    """Return (train, eval) datasets with 'code' and 'docstring' columns."""
    raw = load_dataset(cfg["dataset"], cfg.get("subset"))
    out = {}
    for split, limit in (("train", cfg["max_train"]), ("test", cfg["max_eval"])):
        ds = raw[split].shuffle(seed=cfg["seed"])
        rows = []
        # Over-sample then filter, so we usually hit the limit without scanning everything.
        for row in ds.select(range(min(len(ds), limit * 3))):
            ex = to_example(row, cfg)
            if ex:
                rows.append(ex)
            if len(rows) >= limit:
                break
        out[split] = Dataset.from_list(rows)
    return out["train"], out["test"]

