#!/usr/bin/env python3
"""Submit every PDF in an olmOCR-Bench directory through Unsiloed's hosted
parser (v3/parse) and lay the returned markdown out in the candidate layout
expected by the upstream scorer.

Usage:
    KEY=<your-api-key> python clients/bench_via_api.py \\
        data/olmOCR-bench/bench_data unsiloed

That writes results to:
    data/olmOCR-bench/bench_data/unsiloed/<category>/<stem>_pg1_repeat1.md

Then score with the unmodified upstream scorer (auto-discovers all
candidate subdirs of `--dir`):
    python -m olmocr.bench.benchmark --dir data/olmOCR-bench/bench_data

Requires:
    - KEY env: your API key (request access — see README)
    - The bench_data/ dir produced by `python bench.py download`
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

API = os.environ.get("API", "https://prod.visionapi.unsiloed.ai/v3/parse")
KEY = os.environ.get("KEY")
if not KEY:
    sys.exit("Set KEY env var to your API key (open an issue to request one).")

HEADERS = {"X-API-Key": KEY}
MULTIPART_THRESHOLD = 3 * 1024 * 1024  # 3 MB; above this use /upload flow
POLL_INTERVAL_SEC = 15
RATE_LIMIT_BACKOFF_SEC = 5  # initial backoff on 429; exponential up to 60 s
RESULT_DOWNLOAD_TIMEOUT_SEC = 60


def _request_with_429_retry(method: str, url: str, **kwargs) -> requests.Response:
    """Wrap a single HTTP call with retries on 429 (the API key's 2 RPS / 100/day
    quota). Each retry doubles the wait up to a 60 s cap, then keeps retrying at
    60 s spacing. Other errors are returned as-is."""
    import time as _t
    wait = RATE_LIMIT_BACKOFF_SEC
    while True:
        r = requests.request(method, url, **kwargs)
        if r.status_code != 429:
            return r
        _t.sleep(wait)
        wait = min(wait * 2, 60)


def _submit_pdf(pdf: Path) -> str:
    """Submit a single PDF, return the job_id. Picks the right path by size."""
    size = pdf.stat().st_size
    if size <= MULTIPART_THRESHOLD:
        with pdf.open("rb") as f:
            r = _request_with_429_retry(
                "POST", API, headers=HEADERS,
                files={"file": (pdf.name, f, "application/pdf")},
            )
        r.raise_for_status()
        return r.json()["job_id"]

    # Larger files: 3-step upload via presigned PUT
    r = _request_with_429_retry("POST", f"{API}/upload", headers=HEADERS)
    r.raise_for_status()
    up = r.json()
    file_id = up["file_id"]
    upload_url = up["upload_url"]
    with pdf.open("rb") as f:
        # PUT goes to S3 directly — no API rate limit applies
        put = requests.put(upload_url, headers={"Content-Type": "application/pdf"}, data=f)
    put.raise_for_status()
    r = _request_with_429_retry(
        "POST", API,
        headers={**HEADERS, "Content-Type": "application/json"},
        json={"file_id": file_id},
    )
    r.raise_for_status()
    return r.json()["job_id"]


def _wait(job_id: str, label: str = "") -> dict:
    """Poll the job until status is done or failed; return the final JSON body."""
    while True:
        r = _request_with_429_retry("GET", f"{API}/{job_id}", headers=HEADERS)
        r.raise_for_status()
        body = r.json()
        status = body.get("status")
        if status == "done" and body.get("result_url"):
            # Large results are stored separately. The signed URL authorizes
            # the download without the parser API key.
            try:
                result = requests.get(body["result_url"], timeout=RESULT_DOWNLOAD_TIMEOUT_SEC)
            except requests.RequestException as exc:
                raise RuntimeError(f"Could not download parser result ({type(exc).__name__})") from None
            result.raise_for_status()
            body = result.json()
            if not isinstance(body, dict) or body.get("status") != "done" or not isinstance(body.get("pages"), list):
                raise ValueError("Downloaded parser result is missing completed pages")
            return body
        if status in ("done", "failed"):
            return body
        if status == "running":
            p = body.get("progress") or {}
            print(f"  [{label or job_id}] running  page={p.get('page','?')}/{p.get('of','?')}", flush=True)
        time.sleep(POLL_INTERVAL_SEC)


def _write_candidate(out_dir: Path, rel: Path, body: dict) -> None:
    """Lay out result markdown in the `<category>/<stem>_pg{N}_repeat1.md` layout
    that the upstream `olmocr.bench.benchmark` scorer expects."""
    if body.get("status") == "failed":
        print(f"  FAILED {rel}: {body.get('error','')[:120]}", file=sys.stderr)
        # Write empty stubs so the scorer doesn't crash; tests will simply fail.
        for n in range(1, 2):
            out = out_dir / rel.parent / f"{rel.stem}_pg{n}_repeat1.md"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("")
        return

    for page in body.get("pages", []):
        n = page.get("page", 1)
        md = page.get("markdown", "") or ""
        out = out_dir / rel.parent / f"{rel.stem}_pg{n}_repeat1.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md)


def _process(pdf: Path, bench_root: Path, out_dir: Path) -> tuple[str, str]:
    rel = pdf.relative_to(bench_root / "pdfs")
    label = str(rel)
    try:
        job_id = _submit_pdf(pdf)
        body = _wait(job_id, label=label)
        _write_candidate(out_dir, rel, body)
        return label, body.get("status", "?")
    except requests.HTTPError as e:
        return label, f"http-error:{e.response.status_code}"
    except Exception as e:
        return label, f"err:{type(e).__name__}:{e}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("bench_data", type=Path, help="Path to olmOCR-bench's bench_data/ dir")
    ap.add_argument("candidate_name", type=str,
                    help="Name under bench_data/ where outputs go (e.g. 'unsiloed'). "
                         "Matches the layout the upstream scorer auto-discovers.")
    ap.add_argument("--parallel", type=int, default=2,
                    help="How many PDFs to process concurrently (default 2 — match the 2 RPS quota)")
    ap.add_argument("--limit", type=int, default=0,
                    help="Process only the first N PDFs (smoke testing)")
    args = ap.parse_args()

    bench_root = args.bench_data.resolve()
    out_dir = bench_root / args.candidate_name
    out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(p for p in (bench_root / "pdfs").rglob("*.pdf") if not p.name.startswith("._"))
    if args.limit:
        pdfs = pdfs[: args.limit]
    if not pdfs:
        sys.exit(f"no PDFs found under {bench_root / 'pdfs'}")

    print(f"submitting {len(pdfs)} PDFs to {API} (parallel={args.parallel})", flush=True)
    print(f"  outputs → {out_dir}", flush=True)

    failures = 0
    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futs = {ex.submit(_process, pdf, bench_root, out_dir): pdf for pdf in pdfs}
        for i, f in enumerate(as_completed(futs), 1):
            label, status = f.result()
            if status != "done":
                failures += 1
            print(f"[{i}/{len(pdfs)}] {label}: {status}", flush=True)

    print(f"\ndone. {len(pdfs) - failures}/{len(pdfs)} succeeded.", flush=True)
    print(f"\nNext step — score with the unmodified upstream scorer:")
    print(f"  python -m olmocr.bench.benchmark --dir {bench_root}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
