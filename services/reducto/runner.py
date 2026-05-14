"""Reducto runner conforming to olmocr.bench convention.

Required env: REDUCTO_API_KEY
"""
from __future__ import annotations

import os
import tempfile

from pypdf import PdfReader, PdfWriter
from reducto import Reducto


def _to_bool(v):
    """Normalize :k=v string values from parse_method_arg to bool."""
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in ("true", "1", "yes", "on")
    return bool(v)


def _extract_markdown(parse_response) -> str:
    """Reducto's parse response shape:
       result: { type: "full"/"chunked", chunks: [ {blocks: [{content, ...}]} ] }
       Each block.content is markdown for that block."""
    out: list[str] = []
    result = getattr(parse_response, "result", None)
    if result is None:
        return ""
    chunks = getattr(result, "chunks", None) or []
    for chunk in chunks:
        blocks = getattr(chunk, "blocks", None) or []
        for block in blocks:
            content = getattr(block, "content", None)
            if content:
                out.append(content)
    return "\n\n".join(out).strip()


def run_reducto(
    pdf_path: str,
    page_num: int = 1,
    use_agentic: bool = False,
    use_bench_instruction: bool = False,
    intelligent_ordering: bool = True,
    summarize_figures: bool = True,
    table_output_format: str = "md",
) -> str:
    """
    Convert page of a PDF file to markdown using Reducto.
    https://docs.reducto.ai/

    Mirrors the upstream non-LLM runner pattern: extract the requested page
    via pypdf, write it to a temp PDF, upload to Reducto, parse, return markdown.

    Args:
        pdf_path (str): The local path to the PDF file.
        page_num (int): The page number to process (starting from 1).
        use_agentic (bool): Enable Reducto's premium agentic pipeline (text +
            table + figure agents). ~3-10x cost vs default. Recommended for
            documents with math equations or complex tables.
        use_bench_instruction (bool): Pass upstream olmOCR's
            `build_openai_silver_data_prompt_no_document_anchoring` text as the
            prompt for each agentic scope — same prompt that drives gpt5_v2
            and llamaparse_agent to emit KaTeX-renderable LaTeX. Implies
            use_agentic=True.
        intelligent_ordering (bool): Reducto's reading-order enhancer (default True).
        summarize_figures (bool): Generate natural-language figure descriptions (default True).
        table_output_format (str): "md" (default), "html", "json", "csv", etc.

    Returns:
        str: The OCR result in markdown format.
    """
    if not os.getenv("REDUCTO_API_KEY"):
        raise SystemExit("You must specify a REDUCTO_API_KEY")

    use_agentic = _to_bool(use_agentic)
    use_bench_instruction = _to_bool(use_bench_instruction)
    intelligent_ordering = _to_bool(intelligent_ordering)
    summarize_figures = _to_bool(summarize_figures)
    if use_bench_instruction:
        use_agentic = True

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
        client = Reducto(api_key=os.environ["REDUCTO_API_KEY"])
        with open(pdf_to_process, "rb") as fh:
            upload = client.upload(file=fh, extension="pdf")
        file_url = getattr(upload, "file_id", None) or getattr(upload, "url", None) or str(upload)

        parse_kwargs = {
            "input": file_url,
            "formatting": {"table_output_format": table_output_format},
        }

        enhance: dict = {}
        if intelligent_ordering:
            enhance["intelligent_ordering"] = True
        if summarize_figures:
            enhance["summarize_figures"] = True

        if use_agentic:
            instruction = None
            if use_bench_instruction:
                from olmocr.bench.prompts import build_openai_silver_data_prompt_no_document_anchoring
                instruction = build_openai_silver_data_prompt_no_document_anchoring("")
            agentic_list = [
                {"scope": "text",   **({"prompt": instruction} if instruction else {})},
                {"scope": "table",  **({"prompt": instruction} if instruction else {})},
                {"scope": "figure", "advanced_chart_agent": True,
                 **({"prompt": instruction} if instruction else {})},
            ]
            enhance["agentic"] = agentic_list

        if enhance:
            parse_kwargs["enhance"] = enhance

        parse_response = client.parse.run(**parse_kwargs)
        markdown = _extract_markdown(parse_response)
        if not markdown:
            raise RuntimeError(
                "Reducto returned no extractable markdown — check API key, "
                "credit balance, or response shape."
            )
        return markdown
    finally:
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
