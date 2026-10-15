"""Add docstrings to every undocumented function in a Python file.

    python infer.py path/to/file.py                 # prints the result
    python infer.py path/to/file.py --write         # writes path/to/file.documented.py
    python infer.py path/to/file.py --base          # use the base model (no adapter)
"""
from __future__ import annotations

import ast
import textwrap
from pathlib import Path

from eval_model import generate
from src.config import base_parser, load_config
from src.modeling import load_model, load_tokenizer


def undocumented_functions(source: str) -> list[ast.FunctionDef]:
    tree = ast.parse(source)
    found = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            one_liner = node.body[0].lineno == node.lineno  # def f(): return 1
            if ast.get_docstring(node) is None and not one_liner:
                found.append(node)
    return found


def format_docstring(doc: str, indent: str) -> list[str]:
    doc = doc.replace('"""', "'''").strip()
    lines = doc.splitlines() or [""]
    q = 'r"""' if "\\" in doc else '"""'  # raw string when the docstring has backslashes
    if len(lines) == 1:
        return [f'{indent}{q}{lines[0]}"""\n']
    out = [f'{indent}{q}{lines[0]}\n']
    out += [f"{indent}{ln}\n" if ln.strip() else "\n" for ln in lines[1:]]
    out.append(f'{indent}"""\n')
    return out


def main() -> None:
    parser = base_parser(__doc__, "configs/qwen_coder_1p5b.yaml")
    parser.add_argument("file")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--base", action="store_true")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)

    path = Path(args.file)
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines(keepends=True)
    funcs = undocumented_functions(source)
    if not funcs:
        print("Every function already has a docstring.")
        return

    codes = [textwrap.dedent(ast.get_source_segment(source, f) or "") for f in funcs]
    adapter = None if args.base else (args.adapter or f"{cfg['train']['output_dir']}/adapter")
    tok = load_tokenizer(cfg["model"]["name"])
    model = load_model(cfg["model"]["name"], cfg["model"]["load_in_4bit"], adapter=adapter)
    docs = generate(model, tok, codes, cfg["eval"]["max_new_tokens"], cfg["eval"]["batch_size"])

    # Insert bottom-up so earlier line numbers stay valid.
    for fn, doc in sorted(zip(funcs, docs), key=lambda x: x[0].body[0].lineno, reverse=True):
        first = fn.body[0]
        indent = " " * first.col_offset
        insert_at = first.lineno - 1
        lines[insert_at:insert_at] = format_docstring(doc, indent)

    result = "".join(lines)
    if args.write:
        target = path.with_suffix(".documented.py")
        target.write_text(result, encoding="utf-8")
        print(f"Wrote {target} ({len(funcs)} docstrings added)")
    else:
        print(result)


if __name__ == "__main__":
    main()
