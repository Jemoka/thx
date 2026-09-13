"""
Generate Reference/ pages for every Python module in the theseus package.

Run before `zensical serve` / `zensical build` (see the README).
The script DFS-walks theseus/, creates one .md stub per module with a
`:::` autodoc directive, and writes a SUMMARY.md for Zensical’s built-in literate navigation.

Doc paths strip the top-level `theseus` prefix so that e.g.
`theseus/execute/solve.py` appears at `Reference/execute/solve` rather than
`Reference/theseus/execute/solve`.  The mkdocstrings identifier still uses the
full dotted module name so imports resolve correctly.

Skipped:
  - __pycache__ directories
  - files/directories whose name starts with _ (except __init__.py)
"""

from pathlib import Path
from collections.abc import Iterator
from typing import Any
import shutil

ROOT = Path(__file__).resolve().parents[1] / "theseus"
OUTPUT = Path(__file__).resolve().parent / "Reference"
# This directory contains only generated files. Remove stale module pages.
if OUTPUT.exists():
    shutil.rmtree(OUTPUT)
OUTPUT.mkdir(parents=True)
nav: dict[str | None, Any] = {}


def build_nav(tree: dict[str | None, Any], depth: int = 0) -> Iterator[str]:
    """Render package indexes and their children as literate navigation."""
    for name, node in tree.items():
        target = node.get(None)
        label = f"[{name}]({target})" if target else name
        yield f"{'    ' * depth}- {label}\n"
        yield from build_nav(
            {k: v for k, v in node.items() if k is not None}, depth + 1
        )


for path in sorted(ROOT.rglob("*.py")):
    if "__pycache__" in path.parts:
        continue

    # Full module identifier: theseus.execute.solve
    module_parts = path.relative_to(ROOT.parent).with_suffix("").parts

    # Doc path parts: strip the leading 'theseus' component
    doc_parts = path.relative_to(ROOT).with_suffix("").parts

    # Skip private modules but keep __init__
    if any(p.startswith("_") and p != "__init__" for p in doc_parts):
        continue

    if doc_parts[-1] == "__init__":
        # theseus/execute/__init__.py → Reference/execute/index.md
        module_parts = module_parts[:-1]
        doc_parts = doc_parts[:-1]
        if not doc_parts:
            # Root theseus/__init__.py — skip; nothing useful and it would
            # turn the "Reference" nav section into a clickable page.
            continue
        doc_path = Path(*doc_parts, "index.md")
    else:
        doc_path = path.relative_to(ROOT).with_suffix(".md")

    full_doc_path = OUTPUT / doc_path
    identifier = ".".join(module_parts)

    node = nav
    for part in doc_parts:
        node = node.setdefault(part, {})
    node[None] = doc_path.as_posix()

    full_doc_path.parent.mkdir(parents=True, exist_ok=True)
    full_doc_path.write_text(f"::: {identifier}\n", encoding="utf-8")

(OUTPUT / "SUMMARY.md").write_text("".join(build_nav(nav)), encoding="utf-8")
