"""GPT-5 family runner.

Mirrors `olmocr.bench.runners.run_chatgpt.run_chatgpt` exactly, except it
omits `temperature` (and other sampling parameters) from the API call —
OpenAI rejects any non-default `temperature`/`top_p` for the GPT-5 family
of reasoning models with HTTP 400. Keeping these kwargs out matches the
configuration upstream olmOCR-Bench expects for GPT-5.x models.

Use this runner for: gpt-5, gpt-5-mini, gpt-5-nano, gpt-5.2, gpt-5.4, gpt-5.5.
For GPT-4o and earlier, use the upstream `chatgpt` method with default
temperature=0.1.
"""
import json
import os
from typing import Literal

from openai import OpenAI

from olmocr.bench.prompts import (
    build_basic_prompt,
    build_openai_silver_data_prompt_no_document_anchoring,
)
from olmocr.data.renderpdf import (
    get_png_dimensions_from_base64,
    render_pdf_to_base64png,
)
from olmocr.prompts.anchor import get_anchor_text
from olmocr.prompts.prompts import (
    PageResponse,
    build_finetuning_prompt,
    build_openai_silver_data_prompt,
    build_openai_silver_data_prompt_v2,
    build_openai_silver_data_prompt_v2_simple,
    build_openai_silver_data_prompt_v3_simple,
    openai_response_format_schema,
)


def run_gpt5(
    pdf_path: str,
    page_num: int = 1,
    model: str = "gpt-5.5",
    target_longest_image_dim: int = 2048,
    prompt_template: Literal[
        "full", "full_no_document_anchoring", "basic", "finetune",
        "fullv2", "fullv2simple", "fullv3simple",
    ] = "finetune",
    response_template: Literal["plain", "json"] = "json",
    max_completion_tokens: int = 20000,
) -> str:
    """
    Convert a page of a PDF file to markdown using OpenAI's GPT-5 family.

    Args:
        pdf_path (str): The local path to the PDF file.
        page_num (int): The page number to process (starting from 1).
        model (str): GPT-5 family model id (default: gpt-5.5).
        target_longest_image_dim (int): Render dimension for the page PNG.
        prompt_template: Which upstream prompt builder to use.
        response_template: "json" (structured) or "plain".

    Returns:
        str: The OCR result in markdown format.
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("You must specify an OPENAI_API_KEY")

    image_base64 = render_pdf_to_base64png(
        pdf_path, page_num=page_num, target_longest_image_dim=target_longest_image_dim
    )
    # Anchor text is best-effort: corrupt font dicts in some PDFs make pypdf
    # raise KeyError('/LastChar') etc. Fall back to empty anchor — image stays.
    try:
        anchor_text = get_anchor_text(pdf_path, page_num, pdf_engine="pdfreport")
    except Exception as e:
        print(f"[gpt5] anchor-text extraction failed for {pdf_path}: {type(e).__name__}: {e!s} — falling back to empty anchor")
        anchor_text = ""

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    if prompt_template == "full":
        prompt = build_openai_silver_data_prompt(anchor_text)
    elif prompt_template == "full_no_document_anchoring":
        prompt = build_openai_silver_data_prompt_no_document_anchoring(anchor_text)
    elif prompt_template == "finetune":
        prompt = build_finetuning_prompt(anchor_text)
    elif prompt_template == "basic":
        prompt = build_basic_prompt()
    elif prompt_template == "fullv2":
        prompt = build_openai_silver_data_prompt_v2(anchor_text)
    elif prompt_template == "fullv2simple":
        width, height = get_png_dimensions_from_base64(image_base64)
        prompt = build_openai_silver_data_prompt_v2_simple(width, height)
    elif prompt_template == "fullv3simple":
        width, height = get_png_dimensions_from_base64(image_base64)
        prompt = build_openai_silver_data_prompt_v3_simple(width, height)
    else:
        raise ValueError("Unknown prompt template")

    # Note: no temperature, no top_p — GPT-5 family rejects them server-side.
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                ],
            }
        ],
        max_completion_tokens=max_completion_tokens,
        response_format=openai_response_format_schema() if response_template == "json" else None,
        safety_identifier="olmocr-bench-runner",
    )

    raw_response = response.choices[0].message.content

    if not response.choices:
        raise RuntimeError("OpenAI returned no choices")
    if response.choices[0].message.refusal is not None:
        raise RuntimeError(f"OpenAI refused: {response.choices[0].message.refusal}")

    finish_reason = response.choices[0].finish_reason
    if finish_reason != "stop":
        # Most common with reasoning models: "length" means we hit
        # max_completion_tokens before the model finished its reasoning trace.
        # The response may be empty or truncated. We fall back to whatever was
        # produced rather than discarding the page entirely.
        print(f"[gpt5] WARN: finish_reason={finish_reason!r} (not 'stop'); attempting to recover partial output")

    if response_template == "json":
        try:
            data = json.loads(raw_response or "{}")
            data = PageResponse(**data)
            return data.natural_text or ""
        except Exception as e:
            # Truncation or schema mismatch — fall back to raw content.
            print(f"[gpt5] WARN: JSON parse/validation failed ({type(e).__name__}); returning raw content")
            return raw_response or ""
    return raw_response or ""
