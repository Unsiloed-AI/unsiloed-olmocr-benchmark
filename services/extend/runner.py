"""Extend AI runner conforming to olmocr.bench convention.

Required env: EXTEND_API_KEY

Two-step API flow (mirrors `services/reducto/runner.py`):
  1) client.files.upload(file=<binary>) -> file_id
  2) client.parse(file={"fileId": file_id}, config={...}) -> chunks
"""
from __future__ import annotations

import os
import tempfile

from extend_ai import Extend
from pypdf import PdfReader, PdfWriter


def _to_bool(v):
    """Normalize :k=v string values from parse_method_arg to bool."""
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in ("true", "1", "yes", "on")
    return bool(v)


def _extract_file_id(upload_response) -> str:
    """The Extend Python SDK (>=1.9.0) returns a `File` object directly with
    `.id` (unwrapping the raw HTTP `{success, file: {...}}` envelope).
    We still tolerate the unwrapped dict / nested shapes defensively."""
    # Preferred: SDK File object — `.id` directly
    file_id = getattr(upload_response, "id", None)
    if file_id:
        return file_id
    # Fallbacks: raw HTTP shape or older SDKs
    file_obj = getattr(upload_response, "file", None)
    if file_obj is None and isinstance(upload_response, dict):
        file_obj = upload_response.get("file") or upload_response
    if file_obj is None:
        raise RuntimeError(f"Extend upload returned no id/file: {upload_response!r}")
    file_id = getattr(file_obj, "id", None)
    if file_id is None and isinstance(file_obj, dict):
        file_id = file_obj.get("id")
    if not file_id:
        raise RuntimeError(f"Extend upload returned no file id: {file_obj!r}")
    return file_id


def _extract_markdown(parse_response) -> str:
    """Extend's parse response shape:
       chunks: [ { content: "<md>", blocks: [...], metadata: {...} } ]
       Possibly nested under `output`. We try both."""
    chunks = None
    out = getattr(parse_response, "output", None)
    if out is not None:
        chunks = getattr(out, "chunks", None)
        if chunks is None and isinstance(out, dict):
            chunks = out.get("chunks")
    if chunks is None:
        chunks = getattr(parse_response, "chunks", None)
    if chunks is None and isinstance(parse_response, dict):
        chunks = parse_response.get("chunks") or (parse_response.get("output") or {}).get("chunks")
    if not chunks:
        return ""

    parts: list[str] = []
    for chunk in chunks:
        content = getattr(chunk, "content", None)
        if content is None and isinstance(chunk, dict):
            content = chunk.get("content")
        if content:
            parts.append(content)
    return "\n\n".join(parts).strip()


def run_extend(
    pdf_path: str,
    page_num: int = 1,
    use_agentic: bool = True,
    engine: str = "parse_performance",
    formulas: bool = True,
    advanced_charts: bool = True,
    figures: bool = True,
    signature_detection: bool = True,
    page_rotation: bool = True,
    table_format: str = "markdown",
) -> str:
    """
    Convert page of a PDF file to markdown using Extend AI's Parse API.
    https://docs.extend.ai/product/parsing/parse

    Mirrors the upstream non-LLM runner pattern: extract the requested page
    via pypdf, upload to Extend's /files/upload, then call /parse referencing
    the returned fileId.

    Args:
        pdf_path (str): The local path to the PDF file.
        page_num (int): The page number to process (starting from 1).
        use_agentic (bool): Enable Extend's VLM-based correction for both
            text (low-confidence OCR) and tables. ~Higher cost vs default.
            Default True — this is the "best-quality" configuration.
        engine (str): "parse_performance" (default, higher quality) or
            "parse_light" (lower latency, lower quality).
        formulas (bool): Detect mathematical formulas and emit their LaTeX
            representation. Default True — required for the math test
            class of olmOCR-Bench.
        advanced_charts (bool): Run a specialized vision model on figures
            classified as charts. Default True.
        figures (bool): Detect figures and include their captions/contents
            in the markdown. Default True.
        signature_detection (bool): Default True. Extend's signature detector.
        page_rotation (bool): Auto-correct rotated pages. Default True.
        table_format (str): "markdown" (default) or "html".

    Returns:
        str: The parsed page in markdown.
    """
    if not os.getenv("EXTEND_API_KEY"):
        raise SystemExit("You must specify an EXTEND_API_KEY")

    use_agentic = _to_bool(use_agentic)
    formulas = _to_bool(formulas)
    advanced_charts = _to_bool(advanced_charts)
    figures = _to_bool(figures)
    signature_detection = _to_bool(signature_detection)
    page_rotation = _to_bool(page_rotation)

    pdf_to_process = pdf_path
    temp_file = None

    if page_num > 0:
        reader = PdfReader(pdf_path)
        if page_num > len(reader.pages):
            raise ValueError(f"Page {page_num} does not exist in the PDF. PDF has {len(reader.pages)} pages.")
        writer = PdfWriter()
        writer.add_page(reader.pages[page_num - 1])
        temp_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        temp_file.close()
        with open(temp_file.name, "wb") as output_pdf:
            writer.write(output_pdf)
        pdf_to_process = temp_file.name

    try:
        client = Extend(token=os.environ["EXTEND_API_KEY"])

        with open(pdf_to_process, "rb") as fh:
            upload = client.files.upload(file=fh)
        file_id = _extract_file_id(upload)

        config = {
            "target": "markdown",
            "engine": engine,
            "chunking_strategy": {"type": "page"},
            "block_options": {
                "text": {
                    "agentic": {"enabled": use_agentic},
                    "signature_detection_enabled": signature_detection,
                },
                "tables": {
                    "target_format": table_format,
                    "agentic": {"enabled": use_agentic},
                },
                "figures": {
                    "enabled": figures,
                    "advanced_chart_extraction_enabled": advanced_charts,
                },
                "formulas": {"enabled": formulas},
            },
            "advanced_options": {"page_rotation_enabled": page_rotation},
        }

        parse_response = client.parse(file={"id": file_id}, config=config)
        markdown = _extract_markdown(parse_response)
        if not markdown:
            raise RuntimeError(
                "Extend returned no extractable markdown — check API key, "
                "credit balance, or response shape."
            )
        return markdown
    finally:
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
