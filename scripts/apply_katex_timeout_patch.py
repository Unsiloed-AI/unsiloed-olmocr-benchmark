"""Idempotent patch for olmocr.bench.katex.render.

Upstream `wait_for_load_state` and `wait_for_selector` calls use `timeout=0`
(no timeout). A malformed `$…$` block in a candidate's markdown can cause
Playwright's headless Chromium to hang forever, freezing the scorer with no
error and no progress. We replace both with a 30 s timeout that returns an
error result on timeout — the math test then cleanly falls through ("this
candidate equation didn't render" → next hypothesis or test fails).

This is strictly conservative for candidate scores: a hang would have
produced no result at all, so allowing the test to fail is at most equal,
and usually better, for the candidate.

Re-run this after every `uv sync` (which restores upstream).

Usage:
    .venv/bin/python scripts/apply_katex_timeout_patch.py
"""
from __future__ import annotations

import sys
from pathlib import Path


ORIG_LOAD = '    page.wait_for_load_state("networkidle", timeout=0)'
PATCHED_LOAD = '''    try:
        page.wait_for_load_state("networkidle", timeout=30000)
    except PlaywrightError as ex:
        page.close()
        return RenderedEquation(mathml=str(ex), spans=[], error=f"networkidle timeout: {ex}")'''

ORIG_SELECTOR = '    page.wait_for_selector(".katex", state="attached", timeout=0)'
PATCHED_SELECTOR = '''    try:
        page.wait_for_selector(".katex", state="attached", timeout=30000)
    except PlaywrightError as ex:
        page.close()
        return RenderedEquation(mathml=str(ex), spans=[], error=f"katex selector timeout: {ex}")'''


def find_render_py() -> Path:
    import olmocr.bench.katex.render as r  # type: ignore
    return Path(r.__file__)


def main() -> int:
    try:
        path = find_render_py()
    except ImportError:
        print("olmocr is not importable — activate the venv first.", file=sys.stderr)
        return 1

    src = path.read_text()
    changed = False
    if PATCHED_LOAD not in src:
        if ORIG_LOAD not in src:
            print(f"could not find original wait_for_load_state line in {path}", file=sys.stderr)
            return 2
        src = src.replace(ORIG_LOAD, PATCHED_LOAD, 1)
        changed = True
    if PATCHED_SELECTOR not in src:
        if ORIG_SELECTOR not in src:
            print(f"could not find original wait_for_selector line in {path}", file=sys.stderr)
            return 3
        src = src.replace(ORIG_SELECTOR, PATCHED_SELECTOR, 1)
        changed = True

    if not changed:
        print(f"{path}: already patched (no changes)")
        return 0

    path.write_text(src)
    print(f"{path}: patched OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
