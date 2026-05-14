import os
import tempfile

from llama_cloud_services import LlamaParse
from pypdf import PdfReader, PdfWriter


def run_llamaparse(
    pdf_path: str,
    page_num: int = 1,
    result_type: str = "markdown",
    parse_mode: str | None = None,
    use_bench_instruction: bool = False,
    parsing_instruction: str | None = None,
) -> str:
    """
    Convert page of a PDF file to markdown using LlamaParse (LlamaIndex).

    Mirrors the upstream non-LLM runner pattern: extract the requested page
    via pypdf, write it to a temp PDF, and hand that to the provider SDK.

    Args:
        pdf_path (str): The local path to the PDF file.
        page_num (int): The page number to process (starting from 1).
        result_type (str): "markdown" (default) or "text".
        parse_mode (str): Optional LlamaParse parse_mode. Common values:
            "parse_page_with_llm" (default), "parse_page_with_agent" (best
            quality, ~10x credit cost), "parse_page_with_layout_agent",
            "parse_document_with_agent". See LlamaParse docs.
        use_bench_instruction (bool): If True, pass upstream olmOCR's
            `build_openai_silver_data_prompt_no_document_anchoring` text
            as the parsing_instruction — the same prompt that drives gpt5_v2
            to emit KaTeX-renderable `\\(...\\)` LaTeX, strip headers/footers,
            and emit tables as Markdown. Recommended with parse_mode=
            parse_page_with_agent for an apples-to-apples math comparison.
        parsing_instruction (str): Override the parsing instruction directly.
            Mutually exclusive with use_bench_instruction.

    Returns:
        str: The OCR result in markdown format.
    """
    if not os.getenv("LLAMA_CLOUD_API_KEY"):
        raise SystemExit("You must specify an LLAMA_CLOUD_API_KEY")

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
        kwargs = dict(
            api_key=os.environ["LLAMA_CLOUD_API_KEY"],
            result_type=result_type,
            verbose=False,
        )
        # Optional region override: SDK defaults to US. Set
        # LLAMA_CLOUD_BASE_URL to https://api.cloud.eu.llamaindex.ai for EU keys.
        # See: https://developers.llamaindex.ai/python/cloud/general/regions
        base_url = os.environ.get("LLAMA_CLOUD_BASE_URL")
        if base_url:
            kwargs["base_url"] = base_url
        if parse_mode:
            kwargs["parse_mode"] = parse_mode
        if use_bench_instruction:
            from olmocr.bench.prompts import build_openai_silver_data_prompt_no_document_anchoring
            kwargs["system_prompt_append"] = build_openai_silver_data_prompt_no_document_anchoring("")
        elif parsing_instruction:
            kwargs["system_prompt_append"] = parsing_instruction
        parser = LlamaParse(**kwargs)
        docs = parser.load_data(pdf_to_process)
        # Guard: LlamaParse swallows auth/transport errors and returns []
        # rather than raising. Treat that as a failure so our retry wrapper
        # logs it instead of silently writing an empty tombstone.
        if not docs:
            raise RuntimeError(
                "LlamaParse returned 0 documents — likely auth/region/transport "
                "error. Check LLAMA_CLOUD_API_KEY and LLAMA_CLOUD_BASE_URL."
            )
        return "\n\n".join((getattr(d, "text", "") or "") for d in docs).strip()
    finally:
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
