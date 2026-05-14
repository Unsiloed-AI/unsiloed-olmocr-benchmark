"""Claude (Anthropic API-key auth) runner.

Mirrors `olmocr.bench.runners.run_claude.run_claude` for backward compatibility,
but adds the same `prompt_template` and `response_template` kwargs our `gpt5`
runner exposes — so the v2 prompt config (`full_no_document_anchoring` + plain)
can be run against Claude for an apples-to-apples comparison with `gpt5_v2`.

Provided locally because upstream `run_claude.py` does

    from prompts import build_openai_silver_data_prompt, claude_response_format_schema

— a relative-style import that breaks when the module is loaded normally
(`ModuleNotFoundError: No module named 'prompts'` in olmocr==0.4.27).

Required env: ANTHROPIC_API_KEY
"""
import json
import os
from typing import Literal

from anthropic import Anthropic

from olmocr.bench.prompts import (
    build_basic_prompt,
    build_openai_silver_data_prompt_no_document_anchoring,
    claude_response_format_schema,
)
from olmocr.data.renderpdf import (
    get_png_dimensions_from_base64,
    render_pdf_to_base64png,
)
from olmocr.prompts.anchor import get_anchor_text
from olmocr.prompts.prompts import (
    build_finetuning_prompt,
    build_openai_silver_data_prompt,
    build_openai_silver_data_prompt_v2,
    build_openai_silver_data_prompt_v2_simple,
    build_openai_silver_data_prompt_v3_simple,
)


def run_claude_api(
    pdf_path: str,
    page_num: int = 1,
    model: str = "claude-3-7-sonnet-20250219",
    temperature: float = 0.1,
    target_longest_image_dim: int = 2048,
    prompt_template: Literal[
        "full", "full_no_document_anchoring", "basic", "finetune",
        "fullv2", "fullv2simple", "fullv3simple",
    ] = "full",
    response_template: Literal["plain", "json"] = "json",
    max_tokens: int = 8192,
) -> str:
    """
    Convert a page of a PDF file to markdown using Claude (direct API key auth).

    Args:
        pdf_path (str): The local path to the PDF file.
        page_num (int): The page number to process (starting from 1).
        model (str): Anthropic model id (default: claude-3-7-sonnet-20250219).
        temperature (float): Sampling temperature (Claude allows custom values).
        prompt_template: Which upstream prompt builder to use. Default "full"
            mirrors upstream `run_claude`; use "full_no_document_anchoring"
            for the bench's "No Anchor" leaderboard config.
        response_template: "json" wraps the response via the page_response
            tool_use schema (upstream default). "plain" requests free-form
            markdown (better for KaTeX-renderable LaTeX in math pages).
        max_tokens: Output token cap (default: 8192).

    Returns:
        str: The OCR result in markdown format.
    """
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit("You must specify an ANTHROPIC_API_KEY")

    image_base64 = render_pdf_to_base64png(
        pdf_path, page_num=page_num, target_longest_image_dim=target_longest_image_dim
    )
    # Anchor text is best-effort: corrupt font dicts can make pypdf raise.
    try:
        anchor_text = get_anchor_text(pdf_path, page_num, pdf_engine="pdfreport")
    except Exception as e:
        print(f"[claude_api] anchor-text extraction failed for {pdf_path}: "
              f"{type(e).__name__}: {e!s} — falling back to empty anchor")
        anchor_text = ""

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
        raise ValueError(f"Unknown prompt template: {prompt_template}")

    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    if response_template == "json":
        # Upstream's tool_use path: model returns structured JSON via the
        # page_response tool. Mirrors `run_claude` exactly.
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=claude_response_format_schema(),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_base64}},
                        {
                            "type": "text",
                            "text": (
                                f"{prompt}. Use the page_response tool to respond. "
                                "If the propeties are true, then extract the text "
                                "from them and respond in natural_text."
                            ),
                        },
                    ],
                }
            ],
        )
        for block in response.content:
            if getattr(block, "type", None) == "tool_use" and block.name == "page_response":
                return json.dumps(block.input or {}, indent=2)
        # Fall back to any text blocks if the model didn't use the tool.
        parts = [b.text for b in response.content if getattr(b, "type", None) == "text"]
        return "\n".join(parts).strip()

    # Plain mode — no tool_use, free-form markdown response.
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_base64}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    parts = [b.text for b in response.content if getattr(b, "type", None) == "text"]
    return "\n".join(parts).strip()
