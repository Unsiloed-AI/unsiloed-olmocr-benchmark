import os
import tempfile

from pypdf import PdfReader, PdfWriter
from unstructured_client import UnstructuredClient
from unstructured_client.models import operations, shared


def _elements_to_markdown(elements: list[dict]) -> str:
    """Render Unstructured's element list as markdown — preserve element order
    (Unstructured returns elements in reading order)."""
    lines: list[str] = []
    for el in elements:
        text = (el.get("text") or "").strip()
        if not text:
            continue
        cat = el.get("type", "")
        if cat == "Title":
            lines.append(f"# {text}")
        elif cat == "Header":
            lines.append(f"## {text}")
        elif cat == "ListItem":
            lines.append(f"- {text}")
        elif cat == "Table":
            md = (el.get("metadata") or {}).get("text_as_html") or text
            lines.append(md)
        else:
            lines.append(text)
        lines.append("")
    return "\n".join(lines).strip()


def run_unstructured(
    pdf_path: str,
    page_num: int = 1,
    strategy: str = "hi_res",
) -> str:
    """
    Convert page of a PDF file to markdown using the Unstructured.io API.
    https://docs.unstructured.io/api-reference/api-services/overview

    Mirrors the upstream non-LLM runner pattern: extract the requested page
    via pypdf, write it to a temp PDF, partition that file via the SDK.

    Args:
        pdf_path (str): The local path to the PDF file.
        page_num (int): The page number to process (starting from 1).
        strategy (str): Unstructured partition strategy ("hi_res" / "fast" / "auto" / "ocr_only").

    Returns:
        str: The OCR result in markdown format.
    """
    if not os.getenv("UNSTRUCTURED_API_KEY"):
        raise SystemExit("You must specify an UNSTRUCTURED_API_KEY")

    api_url = os.environ.get(
        "UNSTRUCTURED_API_URL", "https://api.unstructuredapp.io/general/v0/general"
    )

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
        client = UnstructuredClient(
            api_key_auth=os.environ["UNSTRUCTURED_API_KEY"],
            server_url=api_url,
        )
        with open(pdf_to_process, "rb") as fh:
            req = operations.PartitionRequest(
                partition_parameters=shared.PartitionParameters(
                    files=shared.Files(
                        content=fh.read(),
                        file_name=os.path.basename(pdf_path),
                    ),
                    strategy=shared.Strategy(strategy),
                    languages=["eng"],
                )
            )
        resp = client.general.partition(request=req)
        return _elements_to_markdown(resp.elements or [])
    finally:
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
