import os
import tempfile

from agentic_doc.parse import parse
from pypdf import PdfReader, PdfWriter


def _to_bool(v):
    """Normalize :k=v string values from parse_method_arg to bool."""
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in ("true", "1", "yes", "on")
    return bool(v)


def run_landing_ai(
    pdf_path: str,
    page_num: int = 1,
    include_marginalia: bool = True,
    include_metadata_in_markdown: bool = True,
) -> str:
    """
    Convert page of a PDF file to markdown using Landing AI's
    Agentic Document Extraction (agentic-doc SDK).
    https://docs.landing.ai/agentic-document-extraction

    Mirrors the upstream non-LLM runner pattern: extract the requested page
    via pypdf and pass the resulting single-page PDF to the SDK.

    The SDK reads `VISION_AGENT_API_KEY`; we accept either that or the
    more conventional `LANDINGAI_API_KEY` and forward it.

    Args:
        pdf_path (str): The local path to the PDF file.
        page_num (int): The page number to process (starting from 1).

    Returns:
        str: The OCR result in markdown format.
    """
    if not os.getenv("VISION_AGENT_API_KEY") and os.getenv("LANDINGAI_API_KEY"):
        os.environ["VISION_AGENT_API_KEY"] = os.environ["LANDINGAI_API_KEY"]
    if not os.getenv("VISION_AGENT_API_KEY"):
        raise SystemExit("You must specify a LANDINGAI_API_KEY")

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
        results = parse(
            pdf_to_process,
            include_marginalia=_to_bool(include_marginalia),
            include_metadata_in_markdown=_to_bool(include_metadata_in_markdown),
        )
        # Guard: agentic-doc auto-retries 429s internally and may exhaust those
        # retries while returning either an empty list or a result with no
        # markdown/text content (no exception raised). Treat that as a failure
        # so our retry wrapper logs it instead of silently writing an empty
        # tombstone. Mirrors the LlamaParse empty-result guard.
        if not results:
            raise RuntimeError(
                "Landing AI returned 0 results — likely retry-exhausted 429 or "
                "transport error. Check rate limits and reduce --parallel."
            )
        out = results[0]
        markdown = (getattr(out, "markdown", None) or getattr(out, "text", "") or "").strip()
        if not markdown:
            raise RuntimeError(
                "Landing AI returned a result with empty markdown/text — likely "
                "retry-exhausted 429 or unparseable page."
            )
        return markdown
    finally:
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
