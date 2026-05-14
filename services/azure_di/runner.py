import os
import tempfile

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest, DocumentContentFormat
from azure.core.credentials import AzureKeyCredential
from pypdf import PdfReader, PdfWriter


def run_azure_di(
    pdf_path: str,
    page_num: int = 1,
    model: str = "prebuilt-layout",
) -> str:
    """
    Convert page of a PDF file to markdown using Azure Document Intelligence.
    https://learn.microsoft.com/azure/ai-services/document-intelligence/

    Mirrors the upstream non-LLM runner pattern: extract the requested page
    via pypdf, write it to a temp PDF, and analyze that file.

    Args:
        pdf_path (str): The local path to the PDF file.
        page_num (int): The page number to process (starting from 1).
        model (str): Azure DI model id (default: prebuilt-layout, which emits markdown).

    Returns:
        str: The OCR result in markdown format.
    """
    endpoint = os.environ.get("AZURE_DI_ENDPOINT")
    key = os.environ.get("AZURE_DI_KEY")
    if not endpoint or not key:
        raise SystemExit("You must specify AZURE_DI_ENDPOINT and AZURE_DI_KEY")

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
        client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))
        with open(pdf_to_process, "rb") as fh:
            poller = client.begin_analyze_document(
                model_id=model,
                body=AnalyzeDocumentRequest(bytes_source=fh.read()),
                output_content_format=DocumentContentFormat.MARKDOWN,
            )
        result = poller.result()
        return (result.content or "").strip()
    finally:
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
