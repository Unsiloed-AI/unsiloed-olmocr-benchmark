# olmOCR-Bench — Cross-Service Comparison (Unsiloed)

_Generated: 2026-05-19_
_Scorer: olmocr==0.4.27 (unmodified)_
_Dataset: [allenai/olmOCR-bench](https://huggingface.co/datasets/allenai/olmOCR-bench), 1,403 PDFs / 8,413 unit tests across 8 categories_

Scores are limited to the 10 services run end-to-end in this repository.

## Leaderboard

| Method | ArXiv | Base | Hdr/Ftr | TinyTxt | MultCol | OldScan | OldMath | Tables | **Overall** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `unsiloed` (Unsiloed hosted parser) | 90.5 | 100.0 | 94.6 | 93.7 | 87.9 | 52.9 | 88.2 | 93.2 | **88.0** ± 1.0 |
| `gpt5` (GPT-5.5) | 83.8 | 100.0 | 93.0 | 92.1 | 83.9 | 50.4 | 84.9 | 88.7 | **84.6** ± 0.9 |
| `claude_api` (Opus 4.7) | 90.5 | 97.5 | 91.7 | 71.7 | 78.6 | 49.6 | 87.8 | 88.1 | **81.9** ± 1.0 |
| `llamaparse_agent` (parse_page_with_agent) | 85.4 | 99.9 | 18.3 | 80.1 | 81.2 | 47.7 | 89.7 | 85.3 | **73.5** ± 1.0 |
| `landing_ai` | 76.9 | 99.8 | 29.6 | 84.6 | 71.4 | 39.4 | 67.9 | 86.1 | **69.5** ± 1.0 |
| `reducto_agent` (agentic + bench instruction) | 68.9 | 99.9 | 17.4 | 82.6 | 68.2 | 41.3 | 68.1 | 81.4 | **66.0** ± 1.0 |
| `extend_agent` (parse_performance + agentic) | — | — | — | — | — | — | — | — | **64.0** ± 1.0 |
| `azure_di` | 0.0 | 99.1 | 21.7 | 85.5 | 67.4 | 28.3 | 0.0 | 87.6 | **48.7** ± 0.9 |
| `textract` | 0.0 | 100.0 | 26.8 | 65.8 | 19.7 | 23.6 | 0.0 | 85.3 | **40.2** ± 0.9 |
| `unstructured` | 0.0 | 99.9 | 32.2 | 61.1 | 58.3 | 18.3 | 0.0 | 49.8 | **39.9** ± 1.0 |

All numbers are pass-rate percentages on the official olmOCR-Bench scorer (`python -m olmocr.bench.benchmark`, unmodified, version 0.4.27). Overall is the official leaderboard macro average with 95% bootstrap confidence interval.

### Per-test-type breakdown

| Method | baseline | math | table | order | present | absent |
| --- | --- | --- | --- | --- | --- | --- |
| `unsiloed` (Unsiloed hosted parser) | 100.0 | 90.2 | 93.2 | — | — | — |
| `gpt5` (GPT-5.5) | 100.0 | 83.9 | 88.7 | 75.6 | 77.0 | 92.0 |
| `claude_api` (Opus 4.7) | 97.5 | 90.2 | 88.0 | 71.9 | 63.5 | 89.9 |
| `llamaparse_agent` (parse_page_with_agent) | 99.9 | 86.0 | 85.3 | 74.9 | 69.2 | 19.6 |
| `landing_ai` | 99.8 | 75.7 | 86.1 | 64.7 | 66.7 | 32.0 |
| `reducto_agent` (agentic + bench instruction) | 99.9 | 68.8 | 81.4 | 62.5 | 68.1 | 19.0 |
| `extend_agent` (parse_performance + agentic) | — | — | — | — | — | — |
| `azure_di` | 99.1 | 0.0 | 87.5 | 59.5 | 61.7 | 24.9 |
| `textract` | 100.0 | 0.0 | 85.3 | 19.1 | 49.0 | 27.9 |
| `unstructured` | 99.9 | 0.0 | 49.7 | 49.7 | 40.9 | 36.1 |

- **baseline** (1,394 tests) — page contains some recognizable text
- **math** (3,385 tests) — KaTeX-renderable LaTeX equation appears in output
- **table** (1,022 tests) — cell content and neighbour relationships preserved
- **order** (1,061 tests) — relative reading order between text spans
- **present** (721 tests) — required text substring appears in output
- **absent** (823 tests) — header/footer text correctly suppressed

## Methodology

1. **Same dataset, same scorer.** Every service is evaluated against the exact `allenai/olmOCR-bench` dataset (1,403 single-page PDFs from arXiv, the Internet Archive, and curated internal collections), scored with `python -m olmocr.bench.benchmark` (olmocr 0.4.27, unmodified).
2. **Same orchestration.** All conversion uses upstream's `olmocr.bench.convert.process_pdfs`.
3. **One config per row, applied uniformly** across all 1,403 PDFs without using bench-internal labels at inference time.

### Pipeline

```
1. CONVERT  python bench.py convert <method>:k=v --parallel N
2. SCORE    python bench.py score <method>
3. COMPARE  python bench.py compare
```

### Reused from upstream olmOCR

| Concern | Source |
| --- | --- |
| Conversion driver / file naming / parallelism | `olmocr.bench.convert.process_pdfs` |
| Scoring (KaTeX rendering, bootstrap CIs, per-category aggregation) | `olmocr.bench.benchmark` |
| PDF → PNG rendering for vision LLMs | `olmocr.data.renderpdf.render_pdf_to_base64png` (2048px) |
| Anchor text extraction | `olmocr.prompts.anchor.get_anchor_text` |
| Prompt builders for vision LLMs | `olmocr.prompts.prompts.*` and `olmocr.bench.prompts.*` |
| Single-page PDF extraction for non-LLM SDKs | `pypdf.PdfReader/PdfWriter` |
| Test data + expected answers | `allenai/olmOCR-bench` from HuggingFace |

### Local additions

- 8 provider runners at `services/<svc>/runner.py`, each exposing `run_<svc>(pdf_path, page_num=1, **kwargs)`.
- 30 s timeout in olmocr's KaTeX renderer (`olmocr/bench/katex/render.py`). Without this patch, malformed `$…$` blocks can hang Playwright indefinitely; on timeout the renderer returns an error result and the math test cleanly falls through. Strictly conservative for candidate scores. Re-apply after `uv sync` with `.venv/bin/python scripts/apply_katex_timeout_patch.py` (idempotent).

## Per-service configuration

| Method | Service / Model | Endpoint / Mode | Prompt / Instruction |
| --- | --- | --- | --- |
| `unsiloed` (Unsiloed hosted parser) | Unsiloed | hosted parsing API | not user-configurable |
| `gpt5` (GPT-5.5) | OpenAI gpt-5.5 | chat.completions.create (vision) | upstream `full_no_document_anchoring` prompt + JSON response schema |
| `claude_api` (Opus 4.7) | Anthropic Claude Opus 4.7 | Anthropic messages API (vision) | same upstream prompt; plain text response (no tool_use) |
| `llamaparse_agent` | LlamaParse agentic mode | `parse_mode=parse_page_with_agent` | same upstream prompt as `system_prompt_append` |
| `landing_ai` | Landing AI Agentic Doc Extraction | `agentic_doc.parse(pdf)` (default) | none — service has no prompt knob |
| `reducto_agent` | Reducto agentic pipeline | `parse.run` with `enhance.agentic=[text,table,figure]` | upstream `full_no_document_anchoring` passed as the prompt for each agentic scope |
| `extend_agent` | Extend AI Parse API | `parse_performance` engine with agentic text + tables, formulas and advanced charts enabled | none — service has no prompt knob |
| `azure_di` | Azure Document Intelligence | `prebuilt-layout` | none |
| `textract` | AWS Textract | `AnalyzeDocument` (synchronous) | none — input is PNG (sync API doesn't accept PDF) |
| `unstructured` | Unstructured.io legacy partition | `/general/v0/general` (legacy endpoint, hi_res strategy) | none |

## Reproducibility

```bash
# 1. Install
cd parse/olmOCR && uv sync
uv run playwright install chromium
uv run python scripts/apply_katex_timeout_patch.py

# 2. Credentials (.env — never committed)
cp .env.example .env && $EDITOR .env

# 3. Dataset
uv run python bench.py download

# 4. Run a method (exact kwargs are stored in runs/<method>/manifest.json)
uv run python bench.py convert "gpt5:prompt_template=full_no_document_anchoring:response_template=plain" --parallel 8

# 5. Score
uv run python bench.py score gpt5

# 6. Compare across services
uv run python bench.py compare
```

Pinned: olmocr==0.4.27. Dataset: `allenai/olmOCR-bench` from HuggingFace at the time of fetch. Per-service runner source code, raw scorer logs, and per-method JSON metrics are committed under `services/`, `runs/scores/`, and `runs/<method>/`.
