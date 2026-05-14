# unsiloed-olmocr-benchmark

Open source harness for running [olmOCR-Bench](https://huggingface.co/datasets/allenai/olmOCR-bench) (Allen AI) against third-party OCR / document-parsing services, plus a thin client to evaluate **Unsiloed's hosted parser** via its public API.

**Principle:** we don't reimplement scoring or change the evaluation methodology. We pin the official `olmocr` package and reuse its CLIs unchanged. Per-vendor code is limited to **provider runners** for services not shipped upstream — written to mirror upstream's runner conventions exactly so the comparison is fair.

> Upstream: [github.com/allenai/olmocr](https://github.com/allenai/olmocr) · Dataset: [allenai/olmOCR-bench](https://huggingface.co/datasets/allenai/olmOCR-bench)

This repo supports two distinct workflows:

| You want to… | Use |
|---|---|
| Reproduce our **per-vendor** comparison numbers (GPT, Claude, LlamaParse, Reducto, etc.) | The `bench.py` harness in this repo |
| Evaluate **Unsiloed's** hosted parser end-to-end | The public `/v3/parse` API — see "Evaluating Unsiloed" below |

Both produce markdown outputs that go through the **same unmodified Allen AI scorer**, so the resulting numbers are directly comparable.

---

## Latest comparison

From `reports/preliminary_2026-05-19.md` (pinned scorer: `olmocr[bench]==0.4.27`):

| Method | Overall (95% CI) |
|---|---|
| `unsiloed` (Unsiloed hosted parser) | **88.0 ± 1.0** |
| `gpt5` (GPT-5.5) | 84.6 ± 0.9 |
| `claude_api` (Opus 4.7) | 81.9 ± 1.0 |
| `llamaparse_agent` (parse_page_with_agent) | 73.5 ± 1.0 |
| `landing_ai` | 69.5 ± 1.0 |
| `reducto_agent` (agentic + bench instruction) | 66.0 ± 1.0 |
| `extend_agent` (parse_performance + agentic) | 64.0 ± 1.0 |
| `azure_di` | 48.7 ± 0.9 |
| `textract` | 40.2 ± 0.9 |
| `unstructured` | 39.9 ± 1.0 |

Per-vendor rows (everything except `unsiloed`) reproducible from this repo (Path A below). Unsiloed's score is reproducible via the public API (Path B below) against the same bench data.

---

## Reproduce these results

Everything is open: the harness, per-vendor runners, and a thin client for Unsiloed Parser's public v3 API. Two paths depending on what you want to verify.

### Path A — Re-run the per-vendor leaderboard

Reproduce every commercial-service row against `allenai/olmOCR-bench` with the unmodified `olmocr==0.4.27` scorer.

```bash
git clone https://github.com/Unsiloed-AI/unsiloed-olmocr-benchmark
cd unsiloed-olmocr-benchmark
uv venv && uv pip install -e .
uv run playwright install chromium                # for KaTeX math rendering
cp .env.example .env && $EDITOR .env              # fill keys for methods you want
uv run python bench.py download                   # ~1.4k PDFs
uv run python bench.py convert gpt5 --parallel 8  # any of: gpt5, claude_api, llamaparse,
                                                  # reducto, extend, landing_ai, azure_di,
                                                  # textract, unstructured
uv run python bench.py score gpt5
```

Each runner mirrors upstream conventions (single-page extraction via `pypdf`, PNG rendering via `olmocr.data.renderpdf`, prompts from `olmocr.prompts`). No re-implemented scoring.

### Path B — Evaluate Unsiloed Parser via the public v3 API

Hosted at `https://prod.visionapi.unsiloed.ai/v3/parse`. Full reference: [docs.unsiloed.ai/api-reference/parser/parse-document-v3](https://docs.unsiloed.ai/api-reference/parser/parse-document-v3).

```bash
curl -X POST -H "X-API-Key: <key>" -F file=@input.pdf https://prod.visionapi.unsiloed.ai/v3/parse
# → 201 {"job_id":"...","status":"queued"}
curl -H "X-API-Key: <key>" "https://prod.visionapi.unsiloed.ai/v3/parse/<job_id>?format=markdown" > result.md
```

Keys are personal and isolated per user; artifacts auto-delete after 24h. Email **opensource@unsiloed.ai** to request one.

To reproduce the Unsiloed Parser score on the same dataset:

```bash
uv run python bench.py download
KEY=<your-api-key> python clients/bench_via_api.py \
    data/olmOCR-bench/bench_data outputs/unsiloed
uv run python -m olmocr.bench.benchmark \
    --dir data/olmOCR-bench/bench_data --candidate outputs/unsiloed
```

Same scorer, same dataset — different parser. The number you compute should match what we publish.

---

## Methods

The dispatcher (`bench.py`) registers two groups of methods. They share the same calling convention and go through the same upstream pipeline.

### Upstream methods (shipped by `olmocr[bench]==0.4.27`)

| Method ID         | Auth env var                              |
| ----------------- | ----------------------------------------- |
| `chatgpt`         | `OPENAI_API_KEY`                          |
| `claude`          | `ANTHROPIC_API_KEY`                       |
| `gemini`          | `GOOGLE_API_KEY`                          |
| `mistral`         | `MISTRAL_API_KEY`                         |
| `marker`, `mineru`, `docling`, `gotocr`, `paddlepaddle`, `paddlevl`, `rolmocr`, `transformers`, `nanonetsocr`, `nanonetsocr_2`, `server`, `olmocr_pipeline` | varies (see olmocr docs) |

### Local methods (this repo)

| Method ID       | Service                            | Auth                                                            |
| --------------- | ---------------------------------- | --------------------------------------------------------------- |
| `gpt5`          | OpenAI GPT-5 family (gpt-5.5, gpt-5-mini, gpt-5-nano, ...) | `OPENAI_API_KEY` |
| `claude_api`    | Claude (direct Anthropic API key)  | `ANTHROPIC_API_KEY`                                             |
| `llamaparse`    | LlamaParse (LlamaIndex)            | `LLAMA_CLOUD_API_KEY`                                           |
| `unstructured`  | Unstructured.io API                | `UNSTRUCTURED_API_KEY` (+ `UNSTRUCTURED_API_URL`)               |
| `landing_ai`    | Landing AI Agentic Doc Extraction  | `LANDINGAI_API_KEY`                                             |
| `azure_di`      | Azure Document Intelligence        | `AZURE_DI_ENDPOINT` + `AZURE_DI_KEY`                            |
| `textract`      | AWS Textract                       | standard AWS credential chain                                   |
| `reducto`       | Reducto (default + `use_agentic`/`use_bench_instruction` knobs) | `REDUCTO_API_KEY`                            |
| `extend`        | Extend AI (default + `use_agentic`/`formulas`/`advanced_charts` knobs) | `EXTEND_API_KEY`                     |

> **Why a separate `gpt5` method?** OpenAI's GPT-5 family rejects custom `temperature` / `top_p` server-side (HTTP 400), so upstream `chatgpt` (which sends `temperature=0.1` by default) won't run on those models. `gpt5` mirrors `run_chatgpt` exactly *except* it omits sampling parameters. This matches the configuration upstream olmOCR-Bench expects for GPT-5.x models. For GPT-4o and earlier, keep using `chatgpt`.

> **Why a separate `claude_api` method?** Upstream `olmocr.bench.runners.run_claude` uses a relative-style `from prompts import ...` that fails to import as a module (`ModuleNotFoundError: No module named 'prompts'`) in `olmocr==0.4.27`. `claude_api` is byte-identical apart from importing those helpers via their full paths (`olmocr.bench.prompts`, `olmocr.prompts.prompts`).

`uv run python bench.py methods` prints the live list at any time.

---

## Usage

### Convert (run a method against the dataset)

The CLI mirrors `olmocr.bench.convert`'s `method[:k=v ...]` syntax. Outputs land at `data/olmOCR-bench/bench_data/<method>/<category>/<stem>_pg{N}_repeat{R}.md` — exactly the layout the scorer expects.

```bash
# Upstream methods (good for gpt-4o, gemini, etc.):
uv run python bench.py convert "chatgpt:model=gpt-4o-2024-08-06"   --parallel 8

# GPT-5 family — use the local `gpt5` runner (no temperature/top_p):
uv run python bench.py convert "gpt5:model=gpt-5.5"                --parallel 8
uv run python bench.py convert "gpt5:model=gpt-5-mini"             --parallel 8

# Claude — local runner (upstream's `claude` has a broken import in 0.4.27):
uv run python bench.py convert "claude_api:model=claude-opus-4-7"  --parallel 8

# Local methods:
uv run python bench.py convert llamaparse                          --parallel 4
uv run python bench.py convert unstructured                        --parallel 5
uv run python bench.py convert landing_ai                          --parallel 4
uv run python bench.py convert azure_di                            --parallel 8
uv run python bench.py convert textract                            --parallel 4

# Multiple methods at once (one PDF pass per method):
uv run python bench.py convert chatgpt llamaparse azure_di         --parallel 4
```

Useful flags (forwarded to `process_pdfs`):

- `--repeats N` — run each page N times (default: 1)
- `--parallel N` — concurrent workers
- `--force` — re-run even if output exists
- `--remove_text` — render the page as an image first to strip embedded text (matches the *No Anchor* leaderboard rows)
- `--limit N` — process only first N PDFs (smoke testing; macro scores will be 0/N for missing pages)

### Score

```bash
uv run python bench.py score                       # all candidates present in bench_data/
uv run python bench.py score chatgpt               # one candidate
uv run python bench.py score chatgpt --test_report runs/chatgpt.html --skip_baseline
```

This invokes `python -m olmocr.bench.benchmark` directly. Output is the official leaderboard format: per-category scores + macro average with 95 % bootstrap CI.

---

## Adding a new provider

1. `mkdir services/<name>` and add an empty `__init__.py`.
2. Write `services/<name>/runner.py` following the upstream pattern (mirror `run_mistral.py` / `run_marker.py` for non-LLM SDKs, `run_chatgpt.py` / `run_claude.py` for vision LLMs):

   ```python
   import os, tempfile
   from pypdf import PdfReader, PdfWriter
   # from olmocr.data.renderpdf import render_pdf_to_base64png   # if your provider needs PNG bytes

   def run_<name>(pdf_path: str, page_num: int = 1, **kwargs) -> str:
       if not os.getenv("YOUR_API_KEY"):
           raise SystemExit("You must specify a YOUR_API_KEY")

       # Single-page PDF extraction (canonical upstream pattern):
       pdf_to_process, temp_file = pdf_path, None
       if page_num > 0:
           reader = PdfReader(pdf_path)
           if page_num > len(reader.pages):
               raise ValueError(f"Page {page_num} does not exist in the PDF.")
           writer = PdfWriter()
           writer.add_page(reader.pages[page_num - 1])
           temp_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
           temp_file.close()
           with open(temp_file.name, "wb") as f:
               writer.write(f)
           pdf_to_process = temp_file.name
       try:
           # call your provider with pdf_to_process
           return markdown
       finally:
           if temp_file and os.path.exists(temp_file.name):
               os.unlink(temp_file.name)
   ```
3. Register it in `bench.py`'s `LOCAL_METHODS` dict.
4. Add the env var(s) to `.env.example`.
5. `uv run python bench.py convert <name>` and `uv run python bench.py score <name>`.

---

## Path B — Evaluate Unsiloed via the public API

Unsiloed runs its parser as a hosted service. Public endpoint:

**`https://prod.visionapi.unsiloed.ai/v3/parse`**

Full API reference: **[docs.unsiloed.ai/api-reference/parser/parse-document-v3](https://docs.unsiloed.ai/api-reference/parser/parse-document-v3)**

All routes authenticate with a personal `X-API-Key` header — see
[**Request an API key**](#request-an-api-key) below.

### Request an API key

Personal API keys are issued on request. Pick the channel that's easiest:

- **Email**: `opensource@unsiloed.ai`
- **Issue**: open one on this repo — [new issue](https://github.com/Unsiloed-AI/unsiloed-olmocr-benchmark/issues/new)
- **Web**: contact channels at [unsiloed.ai](https://unsiloed.ai)

Include a short note about what you're evaluating (no formal application —
a sentence is plenty). Typical turnaround is same-day.

Once you have a key, every request uses the `X-API-Key` header:

```bash
export API=https://prod.visionapi.unsiloed.ai/v3/parse
export KEY=<your-api-key>
```

### Guarantees we make

| Property | What it means for you |
|---|---|
| **Per-key isolation** | You can only see your own jobs. Polling another user's `job_id` returns 404, as does trying to re-parse another user's `file_id`. |
| **24h retention** | Every job artifact (your uploaded PDF, status, result, container logs) is auto-deleted after 24 hours. Pull your results promptly. |
| **No scoring on our end** | The API returns markdown only. To verify our published olmOCR-Bench score, you run the upstream Allen AI scorer locally against the markdown we return — same scorer used for every other vendor in this repo. |

### All four input options

The same `POST /v3/parse` endpoint accepts three different body shapes,
plus `POST /v3/parse/batch` for many-PDF archives. Pick by use case:

| Option | When to use | Body | HTTP calls |
|---|---|---|---|
| **(1) Multipart inline** | Single PDF ≤ 3 MB | `multipart/form-data` with `file=<pdf>` | 1 (then poll) |
| **(2) JSON with URL** | Single PDF up to 50 MB, publicly hosted by you | `application/json` `{"url": "..."}` | 1 (then poll) |
| **(3) Upload + file_id** | Single PDF up to 50 MB, you don't want to host it | empty POST → `PUT` → POST with `{"file_id": "..."}` | 3 (then poll) |
| **(4) Batch via URL** | Many PDFs in a tar/tar.gz/zip archive | `application/json` `{"url": "<archive URL>"}` at `/batch` | 1 (then poll) |

All four submit asynchronously and return a `job_id` you poll via
`GET /v3/parse/{job_id}` until `status: done`.

---

### Option 1: Small PDF inline (multipart, 1 call)

```bash
# Submit
curl -X POST -H "X-API-Key: $KEY" -F file=@input.pdf $API
# → 201 {"job_id":"<32-char-uuid>","status":"queued","created_at":"..."}

# Poll
JOB=<job_id>
curl -H "X-API-Key: $KEY" $API/$JOB
# → {"status":"running","progress":{"page":1,"of":3},...}
# → eventually {"status":"done","page_count":3,"pages":[{"page":1,"markdown":"..."},...]}

# Pull just the markdown:
curl -H "X-API-Key: $KEY" "$API/$JOB?format=markdown" > result.md
```

**Optional query params:**
- `?pages=1-5` (or `1,3,5`) — only OCR the listed pages

### Option 2: Larger PDF via URL (1 call)

You host the PDF anywhere reachable (S3 with public-read, S3 presigned,
GitHub release asset, your own web server). We fetch it.

```bash
curl -X POST -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
     -d '{"url":"https://example.com/your-paper.pdf"}' $API
# → 201 {"job_id":"...","status":"queued","created_at":"..."}
```

**Constraints:**
- Scheme must be `https://` or `s3://`
- Server rejects URLs pointing at private/metadata IP ranges (SSRF guard)
- PDF max size: 50 MB

### Option 3: Upload + file_id (3 calls — for bigger files when you don't have a public URL)

```bash
# 1. Get a presigned PUT URL (S3, bypasses our API Gateway's 10 MB cap)
UP=$(curl -s -X POST -H "X-API-Key: $KEY" $API/upload)
FILE_ID=$(jq -r .file_id <<< "$UP")
UPLOAD_URL=$(jq -r .upload_url <<< "$UP")

# 2. PUT the PDF directly to S3 (single PUT supports up to 50 MB)
curl -X PUT -H "Content-Type: application/pdf" --upload-file big.pdf "$UPLOAD_URL"

# 3. Start the parse
curl -X POST -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
     -d "{\"file_id\":\"$FILE_ID\"}" $API
# → 201 {"job_id":"...","status":"queued"}; job_id == file_id.
```

Then poll the same way as options 1 and 2.

### Option 4: Batch — many PDFs in an archive

Use this when you want to process a whole dataset (e.g., the olmOCR-Bench
itself). You host an archive (tar, tar.gz, or zip — auto-detected by
content sniffing) of PDFs anywhere reachable; we fetch it.

```bash
curl -X POST -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
     -d '{"url":"https://example.com/my-bench.tar.gz"}' \
     $API/batch
# → 201 {"job_id":"...","status":"queued"}
```

Same SSRF guard applies. Non-PDF files in the archive are ignored.

The poll response on `done` uses a `documents[]` array (one per PDF)
instead of `pages[]`:

```jsonc
{
  "status": "done",
  "pdf_count": 1403,
  "documents": [
    { "pdf": "arxiv_math/2503.04048_pg46.pdf",
      "page_count": 1,
      "pages": [{ "page": 1, "markdown": "..." }] }
  ]
}
```

---

### Reproduce Unsiloed's olmOCR-Bench score

The `clients/bench_via_api.py` helper wraps options 1 and 3 to submit every
PDF in a bench directory, then writes the markdown into the candidate
layout the upstream scorer expects:

```bash
# 1. Download the bench (same as Path A)
uv run python bench.py download

# 2. Submit each bench PDF through the API and collect markdown.
#    Outputs land at data/olmOCR-bench/bench_data/unsiloed/<category>/...
KEY=<your-api-key> python clients/bench_via_api.py \
    data/olmOCR-bench/bench_data unsiloed

# 3. Score with the unmodified upstream scorer (auto-discovers candidates)
uv run python -m olmocr.bench.benchmark --dir data/olmOCR-bench/bench_data
```

The number you compute should match what Unsiloed publishes — same scorer,
same dataset revision, same inputs.

---

## Response shape

GET `/v3/parse/{job_id}` returns JSON inline (or a presigned `result_url`
if the payload would exceed API Gateway's 10 MB cap).

**Single-PDF result:**
```jsonc
{
  "job_id":      "5eb3493042f84af5860531df5b18c56b",
  "status":      "done",
  "file_name":   "input.pdf",
  "started_at":  "...",
  "finished_at": "...",
  "page_count":  3,
  "pages": [
    { "page": 1, "markdown": "..." },
    { "page": 2, "markdown": "..." },
    { "page": 3, "markdown": "..." }
  ]
}
```

**Batch result:**
```jsonc
{
  "job_id":     "...",
  "status":     "done",
  "source_url": "https://your-host/bench.tar.gz",
  "started_at": "...", "finished_at": "...",
  "pdf_count":  1403,
  "documents": [
    { "pdf": "arxiv_math/2503.04048_pg46.pdf", "page_count": 1,
      "pages": [{ "page": 1, "markdown": "..." }] }
  ]
}
```

**While queued / running:**
```jsonc
{ "job_id": "...", "status": "running", "progress": {"page": 2, "of": 5}, ... }
```

---

## Reproducibility

- **Pinned scorer:** `olmocr[bench]==0.4.27` — all scoring + conversion driver code comes from this exact release.
- **Pinned dataset:** `allenai/olmOCR-bench` from HuggingFace (immutable HF revision).
- **Output naming:** delegated entirely to upstream `process_pdfs` (`{stem}_pg{N}_repeat{R}.md`).
- **PDF rendering for vision LLMs:** `olmocr.data.renderpdf.render_pdf_to_base64png` (the renderer the leaderboard uses).
- **Prompts for vision LLMs:** `olmocr.prompts.prompts.build_openai_silver_data_prompt` and `olmocr.prompts.anchor.get_anchor_text` — never re-implemented locally.
- **Single-page extraction for non-LLM SDKs:** `pypdf.PdfReader/PdfWriter` — same idiom upstream uses in `run_mistral.py`, `run_marker.py`, `run_docling.py`.
- **Scoring CLI:** `python -m olmocr.bench.benchmark` invoked unmodified.

If you swap the pinned olmocr version, expect score differences — they're tied to that release's prompt templates and scoring logic.

---

## Project layout

```
unsiloed-olmocr-benchmark/
├── bench.py                ← dispatcher: extends upstream's available_methods
├── pyproject.toml          ← pins olmocr[bench]==0.4.27 + provider SDKs
├── README.md
├── LICENSE                 ← Apache-2.0
├── .env.example
├── clients/
│   └── bench_via_api.py    ← Path B helper: submit a bench dir via /v3/parse
├── data/olmOCR-bench/      ← from `python bench.py download` (gitignored)
├── runs/scores/            ← <method>.json + <method>.<ts>.log (gitignored)
├── reports/                ← comparison_<ts>.md (committed)
├── scripts/
│   └── apply_katex_timeout_patch.py
└── services/
    ├── gpt5/runner.py
    ├── claude_api/runner.py
    ├── llamaparse/runner.py
    ├── unstructured/runner.py
    ├── landing_ai/runner.py
    ├── azure_di/runner.py
    ├── reducto/runner.py
    ├── extend/runner.py
    └── textract/runner.py
```

---

## License & citation

The dataset is licensed ODC-BY-1.0 (Allen AI). When publishing results, cite the olmOCR paper:

> _olmOCR: Unlocking Trillions of Tokens in PDFs with Vision Language Models_, Allen AI, 2025. [arXiv:2502.18443](https://huggingface.co/papers/2502.18443)
