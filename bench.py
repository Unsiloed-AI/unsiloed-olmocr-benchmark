"""Unsiloed olmOCR-Bench dispatcher.

This is a *thin shim* on top of the official `olmocr.bench` machinery:

  * Uses upstream's `process_pdfs` for the convert step (PDF -> markdown,
    correct filename naming, parallelism, --force, --remove_text).
  * Uses upstream's `python -m olmocr.bench.benchmark` for scoring.
  * Adds our project-local runners (services/<svc>/runner.py) to upstream's
    `available_methods` dispatch dict so they participate identically to the
    upstream methods (chatgpt, claude, gemini, mistral, ...).

Usage examples (all idempotent / resumable via upstream's `--force` flag):

    # Upstream methods — exactly as documented at github.com/allenai/olmocr:
    python bench.py convert chatgpt:model=gpt-5.5
    python bench.py convert claude:model=claude-opus-4-7

    # Our custom services (same machinery, registered below):
    python bench.py convert llamaparse
    python bench.py convert unstructured
    python bench.py convert landing_ai
    python bench.py convert azure_di
    python bench.py convert textract

    # Score (any method):
    python bench.py score chatgpt
    python bench.py score llamaparse
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import os
import subprocess
import sys
from pathlib import Path

# Load .env early so SDKs see credentials.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

# Re-use upstream's process_pdfs and arg parsing 1:1.
from olmocr.bench.convert import parse_method_arg, process_pdfs

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data" / "olmOCR-bench" / "bench_data"
RUNS_DIR = Path(__file__).resolve().parent / "runs"


# Upstream-shipped methods (mirrors olmocr.bench.convert's available_methods).
UPSTREAM_METHODS: dict[str, tuple[str, str]] = {
    "olmocr_pipeline": ("olmocr.bench.runners.run_olmocr_pipeline", "run_olmocr_pipeline"),
    "gotocr": ("olmocr.bench.runners.run_gotocr", "run_gotocr"),
    "nanonetsocr": ("olmocr.bench.runners.run_nanonetsocr", "run_nanonetsocr"),
    "nanonetsocr_2": ("olmocr.bench.runners.run_nanonetsocr_2", "run_server"),
    "marker": ("olmocr.bench.runners.run_marker", "run_marker"),
    "mineru": ("olmocr.bench.runners.run_mineru", "run_mineru"),
    "chatgpt": ("olmocr.bench.runners.run_chatgpt", "run_chatgpt"),
    "claude": ("olmocr.bench.runners.run_claude", "run_claude"),
    "gemini": ("olmocr.bench.runners.run_gemini", "run_gemini"),
    "mistral": ("olmocr.bench.runners.run_mistral", "run_mistral"),
    "docling": ("olmocr.bench.runners.run_docling", "run_docling"),
    "rolmocr": ("olmocr.bench.runners.run_rolmocr", "run_rolmocr"),
    "paddlepaddle": ("olmocr.bench.runners.run_paddlepaddle", "run_paddlepaddle"),
    "paddlevl": ("olmocr.bench.runners.run_paddlevl", "run_paddlevl"),
    "transformers": ("olmocr.bench.runners.run_transformers", "run_transformers"),
    "server": ("olmocr.bench.runners.run_server", "run_server"),
}

# Methods we add (services not yet shipped upstream). Each module exposes a
# `run_<name>` callable with the same signature contract as upstream runners:
#
#     def run_<name>(pdf_path: str, page_num: int = 1, **kwargs) -> str: ...
LOCAL_METHODS: dict[str, tuple[str, str]] = {
    "gpt5": ("services.gpt5.runner", "run_gpt5"),
    "claude_api": ("services.claude_api.runner", "run_claude_api"),
    "llamaparse": ("services.llamaparse.runner", "run_llamaparse"),
    "unstructured": ("services.unstructured.runner", "run_unstructured"),
    "landing_ai": ("services.landing_ai.runner", "run_landing_ai"),
    "azure_di": ("services.azure_di.runner", "run_azure_di"),
    "textract": ("services.textract.runner", "run_textract"),
    "reducto": ("services.reducto.runner", "run_reducto"),
    "extend": ("services.extend.runner", "run_extend"),
}

ALL_METHODS = {**UPSTREAM_METHODS, **LOCAL_METHODS}


_RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}


def _is_retryable(exc: BaseException) -> bool:
    """Heuristic: retry on transient network/server/rate-limit errors only.

    Avoid retrying on input-validation errors (HTTP 400, ValueError, etc.) —
    those will fail identically on retry and waste API calls / wall time.
    """
    import socket
    import urllib.error

    msg = str(exc)
    # 400-class permanent errors should NEVER be retried — they'll just fail again.
    if "400" in msg and ("Unsupported" in msg or "invalid" in msg.lower()):
        return False
    # Specific status code mention.
    for code in _RETRYABLE_STATUS:
        if f" {code}" in msg or f"({code})" in msg or f"{code}:" in msg:
            return True
    # Common transient categories by exception type.
    if isinstance(exc, (TimeoutError, ConnectionError, socket.timeout, urllib.error.URLError)):
        return True
    # Phrase-based fallbacks for SDKs that wrap HTTP details in messages.
    lowered = msg.lower()
    return any(s in lowered for s in (
        "rate limit", "rate_limit", "429", "timeout", "timed out",
        "connection reset", "temporarily unavailable", "service unavailable",
        "overloaded", "throttl", "internal server error",
    ))


# Module-level executor used to enforce per-call timeouts on sync runners.
# We can't kill a thread mid-call (Python doesn't allow it), but we can
# abandon the future and let our worker move on. The leaked thread will
# eventually be reaped when the underlying SDK call returns or times out
# at the OS layer.
_TIMEOUT_EXEC = None


# Module-level submission pacer — serializes the moment of API call across
# all worker threads with jittered spacing. Enables a "human-paced" cadence
# that doesn't look like burst traffic. With submit_interval_mean=60s and
# jitter=30s, consecutive submissions are spaced 30–90s apart uniformly.
import threading as _threading
_PACE_LOCK = _threading.Lock()
_PACE_STATE = {"last_submit_t": 0.0, "calls_since_pause": 0}


def _pace_wait(
    mean_s: float,
    jitter_s: float,
    long_pause_every: int = 0,
    long_pause_mean: float = 0.0,
    long_pause_jitter: float = 0.0,
) -> None:
    """Block until at least mean_s ± jitter_s has elapsed since the last
    paced submission. Optionally insert a longer pause every N calls to
    simulate a coffee break / step-away. Thread-safe."""
    import random
    import time
    if mean_s is None or mean_s <= 0:
        return
    with _PACE_LOCK:
        target_gap = max(0.0, mean_s + random.uniform(-jitter_s, jitter_s))
        # Trigger a long pause every N calls
        if (long_pause_every > 0 and long_pause_mean > 0
            and _PACE_STATE["calls_since_pause"] >= long_pause_every):
            extra = max(0.0, long_pause_mean + random.uniform(-long_pause_jitter, long_pause_jitter))
            target_gap += extra
            _PACE_STATE["calls_since_pause"] = 0
            print(f"[pace] long pause: +{extra:.0f}s (~{extra/60:.1f} min)")
        now = time.monotonic()
        last = _PACE_STATE["last_submit_t"]
        wait = (last + target_gap) - now if last > 0 else 0.0
        if wait > 0:
            time.sleep(wait)
        _PACE_STATE["last_submit_t"] = time.monotonic()
        _PACE_STATE["calls_since_pause"] += 1


def _get_timeout_executor():
    global _TIMEOUT_EXEC
    if _TIMEOUT_EXEC is None:
        from concurrent.futures import ThreadPoolExecutor
        _TIMEOUT_EXEC = ThreadPoolExecutor(max_workers=64, thread_name_prefix="bench-timeout")
    return _TIMEOUT_EXEC


def _wrap_with_retry_and_logging(
    method,
    *,
    method_name: str,
    failures_log: Path,
    max_retries: int,
    base_delay_s: float,
    per_call_timeout_s: float | None = None,
    submit_interval_mean: float = 0.0,
    submit_interval_jitter: float = 0.0,
    long_pause_every: int = 0,
    long_pause_mean: float = 0.0,
    long_pause_jitter: float = 0.0,
):
    """Return a wrapped callable that:
       - retries `max_retries` times on transient errors with jittered backoff;
       - if `per_call_timeout_s` is set, enforces a per-attempt wall clock cap
         (raised as TimeoutError, treated as retryable);
       - on final failure, appends one JSONL line to `failures_log` and writes
         a sidecar `<output>.err.json` next to where the empty .md tombstone
         lands; then re-raises so upstream `process_pdf` records the empty .md.

       Wrapping happens at the run_<method> level — `method` keeps its
       original (sync or async) calling convention; we preserve that.
    """
    import asyncio as _asyncio
    import json as _json
    import random
    import time
    import traceback
    from concurrent.futures import TimeoutError as _FutTimeout

    def _now() -> str:
        from datetime import datetime
        return datetime.now().isoformat(timespec="seconds")

    def _log_final_failure(pdf_path: str, page_num: int, exc: BaseException, attempts: int):
        failures_log.parent.mkdir(parents=True, exist_ok=True)
        rec = {
            "timestamp": _now(),
            "method": method_name,
            "pdf_path": pdf_path,
            "page_num": page_num,
            "attempts": attempts,
            "exception_class": type(exc).__name__,
            "exception_msg": str(exc)[:1000],
        }
        with failures_log.open("a", encoding="utf-8") as f:
            f.write(_json.dumps(rec) + "\n")

        # Sidecar next to where the .md tombstone will be written.
        try:
            from pathlib import Path as _P
            stem = _P(pdf_path).stem
            # Match upstream's output naming: <stem>_pg{N}_repeat{R}.md, with R unknown
            # at this layer. We write .err.json named by stem+page so it's discoverable.
            err_dir = failures_log.parent / "errors"
            err_dir.mkdir(parents=True, exist_ok=True)
            tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-2000:]
            (err_dir / f"{stem}_pg{page_num}.err.json").write_text(
                _json.dumps({**rec, "traceback": tb}, indent=2)
            )
        except Exception:
            pass

    is_async = _asyncio.iscoroutinefunction(method)

    if is_async:
        async def wrapper(pdf_path, *args, **kwargs):
            page_num = kwargs.get("page_num", 1)
            last_exc: BaseException | None = None
            for attempt in range(1, max_retries + 1):
                try:
                    if submit_interval_mean and submit_interval_mean > 0 and attempt == 1:
                        _pace_wait(submit_interval_mean, submit_interval_jitter,
                                   long_pause_every, long_pause_mean, long_pause_jitter)
                    if per_call_timeout_s and per_call_timeout_s > 0:
                        return await _asyncio.wait_for(
                            method(pdf_path, *args, **kwargs),
                            timeout=per_call_timeout_s,
                        )
                    return await method(pdf_path, *args, **kwargs)
                except _asyncio.TimeoutError as exc:
                    exc = TimeoutError(f"per-call timeout after {per_call_timeout_s}s")
                    last_exc = exc
                    if attempt >= max_retries:
                        break
                    delay = base_delay_s * (2 ** (attempt - 1)) * (0.5 + random.random())
                    print(f"[retry] {method_name} {Path(pdf_path).name} pg{page_num} "
                          f"attempt {attempt}/{max_retries} timed out; sleeping {delay:.1f}s")
                    await _asyncio.sleep(delay)
                    continue
                except BaseException as exc:
                    last_exc = exc
                    if attempt >= max_retries or not _is_retryable(exc):
                        break
                    delay = base_delay_s * (2 ** (attempt - 1)) * (0.5 + random.random())
                    print(f"[retry] {method_name} {Path(pdf_path).name} pg{page_num} "
                          f"attempt {attempt}/{max_retries} failed ({type(exc).__name__}: {str(exc)[:120]}); "
                          f"sleeping {delay:.1f}s")
                    await _asyncio.sleep(delay)
            assert last_exc is not None
            _log_final_failure(str(pdf_path), page_num, last_exc, attempt)
            raise last_exc
        wrapper.__name__ = method.__name__
        wrapper.__doc__ = method.__doc__
        return wrapper

    def wrapper(pdf_path, *args, **kwargs):
        page_num = kwargs.get("page_num", 1)
        last_exc: BaseException | None = None
        for attempt in range(1, max_retries + 1):
            try:
                if submit_interval_mean and submit_interval_mean > 0 and attempt == 1:
                    _pace_wait(submit_interval_mean, submit_interval_jitter)
                if per_call_timeout_s and per_call_timeout_s > 0:
                    fut = _get_timeout_executor().submit(method, pdf_path, *args, **kwargs)
                    try:
                        return fut.result(timeout=per_call_timeout_s)
                    except _FutTimeout:
                        # We can't actually kill the thread; abandon the future.
                        # The leaked thread will eventually return or die at
                        # the OS layer. Our worker moves on.
                        fut.cancel()
                        raise TimeoutError(
                            f"per-call timeout after {per_call_timeout_s}s"
                        )
                return method(pdf_path, *args, **kwargs)
            except BaseException as exc:
                last_exc = exc
                if attempt >= max_retries or not _is_retryable(exc):
                    break
                delay = base_delay_s * (2 ** (attempt - 1)) * (0.5 + random.random())
                print(f"[retry] {method_name} {Path(pdf_path).name} pg{page_num} "
                      f"attempt {attempt}/{max_retries} failed ({type(exc).__name__}: {str(exc)[:120]}); "
                      f"sleeping {delay:.1f}s")
                time.sleep(delay)
        assert last_exc is not None
        _log_final_failure(str(pdf_path), page_num, last_exc, attempt)
        raise last_exc
    wrapper.__name__ = method.__name__
    wrapper.__doc__ = method.__doc__
    return wrapper


def _build_config(
    method_args: list[str],
    *,
    max_retries: int,
    retry_base_delay: float,
    per_call_timeout: float | None = None,
    submit_interval_mean: float = 0.0,
    submit_interval_jitter: float = 0.0,
    long_pause_every: int = 0,
    long_pause_mean: float = 0.0,
    long_pause_jitter: float = 0.0,
) -> dict:
    """Mirrors olmocr.bench.convert's main() config-build step, but wraps each
    method with retry + structured failure logging + optional per-call timeout
    + optional submission pacing."""
    config: dict = {}
    for method_arg in method_args:
        method_name, extra_kwargs, folder_name = parse_method_arg(method_arg)
        if method_name not in ALL_METHODS:
            raise SystemExit(
                f"Unknown method: {method_name}. "
                f"Available: {', '.join(sorted(ALL_METHODS))}"
            )
        module_path, function_name = ALL_METHODS[method_name]
        module = importlib.import_module(module_path)
        function = getattr(module, function_name)
        failures_log = RUNS_DIR / folder_name / "failures.jsonl"
        wrapped = _wrap_with_retry_and_logging(
            function,
            method_name=folder_name,
            failures_log=failures_log,
            max_retries=max_retries,
            base_delay_s=retry_base_delay,
            per_call_timeout_s=per_call_timeout,
            submit_interval_mean=submit_interval_mean,
            submit_interval_jitter=submit_interval_jitter,
            long_pause_every=long_pause_every,
            long_pause_mean=long_pause_mean,
            long_pause_jitter=long_pause_jitter,
        )
        config[method_name] = {
            "method": wrapped,
            "kwargs": extra_kwargs,
            "folder_name": folder_name,
        }
    return config


def cmd_convert(args: argparse.Namespace) -> int:
    import json as _json
    import time as _time
    from datetime import datetime

    config = _build_config(
        args.methods,
        max_retries=args.max_retries,
        retry_base_delay=args.retry_base_delay,
        per_call_timeout=args.per_call_timeout,
        submit_interval_mean=getattr(args, "submit_interval_mean", 0.0) or 0.0,
        submit_interval_jitter=getattr(args, "submit_interval_jitter", 0.0) or 0.0,
        long_pause_every=getattr(args, "long_pause_every", 0) or 0,
        long_pause_mean=getattr(args, "long_pause_mean", 0.0) or 0.0,
        long_pause_jitter=getattr(args, "long_pause_jitter", 0.0) or 0.0,
    )
    data_dir = Path(args.dir).resolve()
    pdf_dir = data_dir / "pdfs"
    if not pdf_dir.exists():
        raise SystemExit(
            f"{pdf_dir} not found. Run: python bench.py download"
        )

    # Optional --limit N / --category <name> / --shuffle: monkey-patch
    # glob.glob in olmocr.bench.convert to filter and reorder the PDF set.
    if ((args.limit is not None and args.limit > 0)
        or args.category
        or args.shuffle):
        from olmocr.bench import convert as _conv

        real_glob = _conv.glob.glob
        cat = args.category
        shuffle = args.shuffle
        shuffle_seed = args.shuffle_seed

        def _filtered_glob(pattern: str, *a, **kw):
            res = real_glob(pattern, *a, **kw)
            if pattern.endswith("**/*.pdf") or pattern.endswith("/*.pdf"):
                if cat:
                    res = [p for p in res if f"/pdfs/{cat}/" in p or p.endswith(f"/{cat}/" + p.rsplit("/", 1)[-1])]
                    print(f"[bench] --category {cat!r} -> {len(res)} PDFs")
                # First sort deterministically, then optionally shuffle so
                # the seed produces a reproducible random order.
                res = sorted(res)
                if shuffle:
                    import random as _r
                    rng = _r.Random(shuffle_seed) if shuffle_seed is not None else _r.Random()
                    rng.shuffle(res)
                    print(f"[bench] --shuffle (seed={shuffle_seed}) -> categories interleaved")
                if args.limit is not None and args.limit > 0:
                    res = res[: args.limit]
                    print(f"[bench] --limit {args.limit} -> {len(res)} PDFs")
            return res

        _conv.glob.glob = _filtered_glob

    # Per-method run manifest scaffolding.
    started = datetime.now().isoformat(timespec="seconds")
    started_mono = _time.monotonic()
    for method_name, entry in config.items():
        run_dir = RUNS_DIR / entry["folder_name"]
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = run_dir / "manifest.json"
        manifest = {
            "method": method_name,
            "folder_name": entry["folder_name"],
            "kwargs": entry["kwargs"],
            "data_dir": str(data_dir),
            "parallel": args.parallel,
            "repeats": args.repeats,
            "force": args.force,
            "remove_text": args.remove_text,
            "limit": args.limit,
            "max_retries": args.max_retries,
            "retry_base_delay": args.retry_base_delay,
            "started_at": started,
        }
        manifest_path.write_text(_json.dumps(manifest, indent=2))

    asyncio.run(
        process_pdfs(
            config,
            str(pdf_dir),
            str(data_dir),
            args.repeats,
            args.remove_text,
            args.force,
            args.parallel,
        )
    )

    # Finalize manifest with end timestamp + failure count.
    elapsed = _time.monotonic() - started_mono
    finished = datetime.now().isoformat(timespec="seconds")
    for method_name, entry in config.items():
        run_dir = RUNS_DIR / entry["folder_name"]
        manifest_path = run_dir / "manifest.json"
        try:
            manifest = _json.loads(manifest_path.read_text())
        except Exception:
            manifest = {}
        failures_log = run_dir / "failures.jsonl"
        n_failures = 0
        if failures_log.exists():
            with failures_log.open() as f:
                n_failures = sum(1 for _ in f)
        manifest["finished_at"] = finished
        manifest["elapsed_seconds"] = round(elapsed, 1)
        manifest["failures_logged"] = n_failures
        manifest_path.write_text(_json.dumps(manifest, indent=2))
        print(f"[bench] {entry['folder_name']}: {n_failures} failures logged -> {failures_log}")
    return 0


_SCORES_DIR = Path(__file__).resolve().parent / "runs" / "scores"
_REPORTS_DIR = Path(__file__).resolve().parent / "reports"


def _parse_score_stdout(stdout: str, candidate: str) -> dict | None:
    """Parse `python -m olmocr.bench.benchmark` stdout into structured metrics.

    Looks for the candidate's section after "Final Summary with 95% Confidence Intervals:":
      <candidate>          : Average Score: 76.3% ± 1.1% (average of per-JSONL scores)
          math    : 75.5% average pass rate over 3385 tests
          ...
          Results by JSONL file:
              arxiv_math.jsonl              : 76.3% (2235/2927 tests)
              ...
    """
    import re

    sum_pat = re.compile(
        rf"^{re.escape(candidate)}\s*:\s*Average Score:\s*(?P<status>FAILED \(errors\)|[\d.]+%)\s*(?:±\s*(?P<ci>[\d.]+)%)?",
        re.MULTILINE,
    )
    m = sum_pat.search(stdout)
    if not m:
        return None
    status = m.group("status")
    if status.startswith("FAILED"):
        overall = {"status": "failed"}
    else:
        overall = {
            "score_pct": float(status.rstrip("%")),
            "ci_half_width_pct": float(m.group("ci")) if m.group("ci") else None,
        }

    tail = stdout[m.end():]
    next_cand = re.search(r"\n[A-Za-z][\w-]*\s*:\s*Average Score:", tail)
    if next_cand:
        tail = tail[: next_cand.start()]

    by_test_type: dict = {}
    for tm in re.finditer(r"^\s{4,}(\w+)\s*:\s*([\d.]+)%\s*average pass rate over\s*(\d+)", tail, re.MULTILINE):
        by_test_type[tm.group(1)] = {"score_pct": float(tm.group(2)), "tests": int(tm.group(3))}

    by_category: dict = {}
    for cm in re.finditer(r"^\s{8,}([\w./-]+)\s*:\s*([\d.]+)%\s*\((\d+)/(\d+) tests\)", tail, re.MULTILINE):
        by_category[cm.group(1)] = {
            "score_pct": float(cm.group(2)),
            "passed": int(cm.group(3)),
            "total": int(cm.group(4)),
        }

    return {
        "candidate": candidate,
        "overall": overall,
        "by_category": by_category,
        "by_test_type": by_test_type,
    }


def cmd_score(args: argparse.Namespace) -> int:
    import json
    from datetime import datetime

    cmd = [sys.executable, "-m", "olmocr.bench.benchmark", "--dir", str(args.dir)]
    if args.candidate:
        cmd += ["--candidate", args.candidate]
    if args.force:
        cmd += ["--force"]
    if args.test_report:
        cmd += ["--test_report", args.test_report]
    if args.skip_baseline:
        cmd += ["--skip_baseline"]
    print("$", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)

    if not args.candidate or args.no_save:
        return proc.returncode

    _SCORES_DIR.mkdir(parents=True, exist_ok=True)
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    log_path = _SCORES_DIR / f"{args.candidate}.{ts}.log"
    log_path.write_text(proc.stdout + "\n---STDERR---\n" + proc.stderr)

    metrics = _parse_score_stdout(proc.stdout, args.candidate)
    if metrics is not None:
        try:
            from importlib.metadata import version as _v
            metrics["olmocr_version"] = _v("olmocr")
        except Exception:
            metrics["olmocr_version"] = None
        metrics["timestamp"] = ts
        metrics_path = _SCORES_DIR / f"{args.candidate}.json"
        metrics_path.write_text(json.dumps(metrics, indent=2))
        print(f"\n[bench] metrics  -> {metrics_path}")
        print(f"[bench] full log -> {log_path}")
    else:
        print(f"\n[bench] WARNING: could not parse metrics for '{args.candidate}'. Raw log: {log_path}")

    return proc.returncode


def cmd_compare(args: argparse.Namespace) -> int:
    """Aggregate runs/scores/*.json into a leaderboard-style comparison markdown."""
    import json
    from datetime import datetime

    if not _SCORES_DIR.exists():
        raise SystemExit(f"No metrics found at {_SCORES_DIR}. Run `bench.py score <candidate>` first.")

    all_metrics: list[dict] = []
    for p in sorted(_SCORES_DIR.glob("*.json")):
        try:
            all_metrics.append(json.loads(p.read_text()))
        except Exception as e:
            print(f"[bench] skip {p}: {e}", file=sys.stderr)
    if not all_metrics:
        raise SystemExit(f"No JSON metric files in {_SCORES_DIR}.")

    cat_order = [
        ("arxiv_math.jsonl", "ArXiv"),
        ("baseline", "Base"),
        ("headers_footers.jsonl", "Hdr/Ftr"),
        ("long_tiny_text.jsonl", "TinyTxt"),
        ("multi_column.jsonl", "MultCol"),
        ("old_scans.jsonl", "OldScan"),
        ("old_scans_math.jsonl", "OldMath"),
        ("table_tests.jsonl", "Tables"),
    ]

    def _row(m: dict) -> str:
        cells = [f"`{m['candidate']}`"]
        for key, _ in cat_order:
            v = m.get("by_category", {}).get(key)
            cells.append(f"{v['score_pct']:.1f}" if v else "—")
        ov = m.get("overall", {})
        if "score_pct" in ov:
            ci = ov.get("ci_half_width_pct")
            cells.append(f"{ov['score_pct']:.1f}" + (f" ± {ci:.1f}" if ci is not None else ""))
        else:
            cells.append("FAIL")
        return "| " + " | ".join(cells) + " |"

    header = "| Method | " + " | ".join(name for _, name in cat_order) + " | Overall |"
    sep    = "| --- " + "| --- " * len(cat_order) + "| --- |"
    rows = [_row(m) for m in sorted(all_metrics, key=lambda m: -(m.get("overall", {}).get("score_pct") or -1))]

    versions = sorted({m.get("olmocr_version") for m in all_metrics if m.get("olmocr_version")})
    body = [
        "# olmOCR-Bench — Cross-Service Comparison",
        "",
        f"_Generated: {datetime.now().isoformat(timespec='seconds')}_  ",
        f"_olmocr version(s): {', '.join(versions) or 'unknown'}_",
        "",
        "Per-category pass rates (%) plus the official leaderboard macro average.",
        "",
        header,
        sep,
        *rows,
        "",
        "## Per-method JSONs",
        "",
        *[f"- `runs/scores/{m['candidate']}.json` (run at {m.get('timestamp', '?')})" for m in all_metrics],
    ]
    out = _REPORTS_DIR / f"comparison_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.md"
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(body) + "\n")
    print(out.read_text())
    print(f"\n[bench] comparison -> {out}")
    return 0


def cmd_retry_failed(args: argparse.Namespace) -> int:
    """Re-run only the pages logged as failures in runs/<method>/failures.jsonl.

    Strategy:
      1. Read the failures log for the candidate.
      2. Delete the corresponding empty .md tombstones so process_pdfs picks them up.
      3. Truncate failures.jsonl (failures from this retry will be re-logged fresh).
      4. Dispatch a normal convert run — process_pdfs will only regenerate the
         missing files because all other outputs still exist on disk.
    """
    import json as _json
    import time as _time
    from datetime import datetime

    # We need the dispatched method spec — accept method[:k=v] like convert does.
    method_args: list[str] = args.methods
    config = _build_config(
        method_args,
        max_retries=args.max_retries,
        retry_base_delay=args.retry_base_delay,
        per_call_timeout=args.per_call_timeout,
    )
    data_dir = Path(args.dir).resolve()
    pdf_dir = data_dir / "pdfs"
    if not pdf_dir.exists():
        raise SystemExit(f"{pdf_dir} not found. Run: python bench.py download")

    total_to_retry = 0
    for method_name, entry in config.items():
        folder = entry["folder_name"]
        run_dir = RUNS_DIR / folder
        failures_log = run_dir / "failures.jsonl"
        if not failures_log.exists():
            print(f"[bench] {folder}: no failures.jsonl — nothing to retry.")
            continue
        records: list[dict] = []
        with failures_log.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(_json.loads(line))
                    except _json.JSONDecodeError:
                        continue
        if not records:
            print(f"[bench] {folder}: failures.jsonl is empty.")
            continue

        # Delete the corresponding .md tombstones so process_pdfs regenerates them.
        candidate_dir = data_dir / folder
        deleted = 0
        for rec in records:
            stem = Path(rec["pdf_path"]).stem
            page = rec.get("page_num", 1)
            for md in candidate_dir.rglob(f"{stem}_pg{page}_repeat*.md"):
                if md.stat().st_size == 0:
                    md.unlink()
                    deleted += 1
        # Rotate the failures log so this retry's failures are re-logged fresh.
        rotated = run_dir / f"failures.{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.jsonl"
        failures_log.rename(rotated)
        print(f"[bench] {folder}: deleted {deleted} empty tombstones, rotated old log -> {rotated.name}")
        total_to_retry += len(records)

    if total_to_retry == 0:
        print("[bench] nothing to retry across all methods.")
        return 0

    print(f"[bench] retrying {total_to_retry} page(s); dispatching convert...")
    started = datetime.now().isoformat(timespec="seconds")
    started_mono = _time.monotonic()

    # CRITICAL: process_pdfs glob's `pdfs/**/*.pdf`. If we let it run on the full
    # tree, every PDF without a tombstone (i.e. PDFs we haven't even attempted)
    # will be (re)processed. Constrain the glob to ONLY the failed PDF paths.
    failed_paths: set[str] = set()
    for entry in config.values():
        run_dir = RUNS_DIR / entry["folder_name"]
        # We rotated the live log above; read the most recent rotated copy.
        rotated = sorted(run_dir.glob("failures.*.jsonl"))[-1] if list(run_dir.glob("failures.*.jsonl")) else None
        if not rotated:
            continue
        with rotated.open() as f:
            for line in f:
                try:
                    failed_paths.add(_json.loads(line)["pdf_path"])
                except Exception:
                    pass

    if failed_paths:
        from olmocr.bench import convert as _conv
        real_glob = _conv.glob.glob

        def _filtered_glob(pattern: str, *a, **kw):
            res = real_glob(pattern, *a, **kw)
            if pattern.endswith("**/*.pdf") or pattern.endswith("/*.pdf"):
                res = [p for p in res if p in failed_paths]
                print(f"[bench] retry filter: {len(res)} PDFs (from {len(failed_paths)} failures)")
            return res
        _conv.glob.glob = _filtered_glob

    asyncio.run(
        process_pdfs(
            config,
            str(pdf_dir),
            str(data_dir),
            args.repeats,
            args.remove_text,
            False,                # force=False: only fill in deleted tombstones
            args.parallel,
        )
    )

    finished = datetime.now().isoformat(timespec="seconds")
    elapsed = round(_time.monotonic() - started_mono, 1)
    for method_name, entry in config.items():
        run_dir = RUNS_DIR / entry["folder_name"]
        failures_log = run_dir / "failures.jsonl"
        n_failures = 0
        if failures_log.exists():
            with failures_log.open() as f:
                n_failures = sum(1 for _ in f)
        retry_summary = run_dir / "retry_summary.json"
        retry_summary.write_text(_json.dumps({
            "started_at": started,
            "finished_at": finished,
            "elapsed_seconds": elapsed,
            "failures_remaining": n_failures,
        }, indent=2))
        print(f"[bench] {entry['folder_name']}: {n_failures} failures remain after retry "
              f"-> {failures_log if n_failures else '(none)'}")
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    target = Path(args.target).resolve()
    target.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="allenai/olmOCR-bench",
        repo_type="dataset",
        local_dir=str(target),
    )
    print(f"olmOCR-bench downloaded to {target}")
    return 0


def cmd_methods(args: argparse.Namespace) -> int:
    print("Upstream (from olmocr 0.4.27):")
    for k in sorted(UPSTREAM_METHODS):
        print(f"  {k}")
    print("\nLocal (this repo):")
    for k in sorted(LOCAL_METHODS):
        print(f"  {k}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_conv = sub.add_parser("convert", help="Run a method against the bench dataset.")
    p_conv.add_argument("methods", nargs="+", help="method[:k=v ...] (one or more)")
    p_conv.add_argument("--dir", type=str, default=str(DEFAULT_DATA_DIR))
    p_conv.add_argument("--repeats", type=int, default=1)
    p_conv.add_argument("--parallel", type=int, default=1)
    p_conv.add_argument("--force", action="store_true")
    p_conv.add_argument("--remove_text", action="store_true")
    p_conv.add_argument("--limit", type=int, default=None, help="Process only the first N PDFs (smoke tests).")
    p_conv.add_argument("--category", type=str, default=None,
                        help="Restrict to one bench category subdir (arxiv_math, headers_footers, long_tiny_text, multi_column, old_scans, old_scans_math, tables).")
    p_conv.add_argument("--max-retries", dest="max_retries", type=int, default=4,
                        help="Per-page retry attempts on transient errors (default: 4 — i.e. up to 3 retries).")
    p_conv.add_argument("--retry-base-delay", dest="retry_base_delay", type=float, default=2.0,
                        help="Base seconds for exponential backoff with jitter (default: 2.0).")
    p_conv.add_argument("--per-call-timeout", dest="per_call_timeout", type=float, default=None,
                        help="Wall-clock cap (seconds) per provider call. Times out are treated as retryable.")
    p_conv.add_argument("--submit-interval-mean", dest="submit_interval_mean", type=float, default=0.0,
                        help="Mean seconds between successive submissions (humane pacing). 0 disables.")
    p_conv.add_argument("--submit-interval-jitter", dest="submit_interval_jitter", type=float, default=0.0,
                        help="±jitter (seconds) added to --submit-interval-mean (uniform).")
    p_conv.add_argument("--long-pause-every", dest="long_pause_every", type=int, default=0,
                        help="Every N calls, insert an extra long pause (simulates a coffee break). 0 disables.")
    p_conv.add_argument("--long-pause-mean", dest="long_pause_mean", type=float, default=180.0,
                        help="Mean seconds for the long pause when triggered (default 180=3 min).")
    p_conv.add_argument("--long-pause-jitter", dest="long_pause_jitter", type=float, default=120.0,
                        help="±jitter on the long pause (default 120 → 1-5 min total range).")
    p_conv.add_argument("--shuffle", action="store_true",
                        help="Randomize PDF order so categories interleave naturally (vs. alphabetical clustering).")
    p_conv.add_argument("--shuffle-seed", type=int, default=None,
                        help="Optional seed for --shuffle (reproducible random order).")
    p_conv.set_defaults(func=cmd_convert)

    p_score = sub.add_parser("score", help="Run olmocr.bench.benchmark on collected outputs and persist metrics.")
    p_score.add_argument("candidate", nargs="?", default=None, help="Restrict to one candidate folder.")
    p_score.add_argument("--dir", type=str, default=str(DEFAULT_DATA_DIR))
    p_score.add_argument("--force", action="store_true")
    p_score.add_argument("--test_report", type=str, default=None)
    p_score.add_argument("--skip_baseline", action="store_true")
    p_score.add_argument("--no-save", action="store_true", help="Don't write structured JSON/log artifacts to runs/scores/.")
    p_score.set_defaults(func=cmd_score)

    p_cmp = sub.add_parser("compare", help="Aggregate runs/scores/*.json into a comparison markdown table.")
    p_cmp.set_defaults(func=cmd_compare)

    p_retry = sub.add_parser("retry-failed", help="Re-run only the pages logged in runs/<method>/failures.jsonl.")
    p_retry.add_argument("methods", nargs="+", help="method[:k=v ...] — same syntax as `convert`")
    p_retry.add_argument("--dir", type=str, default=str(DEFAULT_DATA_DIR))
    p_retry.add_argument("--repeats", type=int, default=1)
    p_retry.add_argument("--parallel", type=int, default=1)
    p_retry.add_argument("--remove_text", action="store_true")
    p_retry.add_argument("--max-retries", dest="max_retries", type=int, default=4)
    p_retry.add_argument("--retry-base-delay", dest="retry_base_delay", type=float, default=2.0)
    p_retry.add_argument("--per-call-timeout", dest="per_call_timeout", type=float, default=None)
    p_retry.set_defaults(func=cmd_retry_failed)

    p_dl = sub.add_parser("download", help="Download allenai/olmOCR-bench from HuggingFace.")
    p_dl.add_argument("--target", default=str(DEFAULT_DATA_DIR.parent), help="dataset root (default: ./data/olmOCR-bench)")
    p_dl.set_defaults(func=cmd_download)

    p_m = sub.add_parser("methods", help="List available methods (upstream + local).")
    p_m.set_defaults(func=cmd_methods)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
