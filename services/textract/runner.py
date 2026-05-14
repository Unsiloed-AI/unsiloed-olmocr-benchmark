"""AWS Textract runner.

NOTE: AWS Textract's *synchronous* `AnalyzeDocument` API does not accept
multi-page PDFs — it requires PNG/JPEG bytes (the asynchronous
`StartDocumentAnalysis` API does support PDFs but only via S3, which we'd
rather not require for a self-contained benchmark). For each page we
therefore render a PNG via `olmocr.data.renderpdf.render_pdf_to_base64png`
(the same renderer upstream's vision-LLM runners use) and feed the image
bytes to Textract.

Auth: standard AWS chain — env vars or `~/.aws/credentials`.
"""
import base64
import os
from collections import defaultdict

import boto3

from olmocr.data.renderpdf import render_pdf_to_base64png


def _blocks_to_markdown(blocks: list[dict]) -> str:
    """Render Textract's block graph as markdown: text lines in reading order
    + tables formatted as Markdown tables."""
    by_id = {b["Id"]: b for b in blocks}
    table_blocks = [b for b in blocks if b["BlockType"] == "TABLE"]

    skip_ids: set[str] = set()
    for t in table_blocks:
        for r in t.get("Relationships", []) or []:
            if r["Type"] == "CHILD":
                for cid in r["Ids"]:
                    skip_ids.add(cid)
                    cell = by_id.get(cid, {})
                    for cr in cell.get("Relationships", []) or []:
                        if cr["Type"] == "CHILD":
                            for wid in cr["Ids"]:
                                skip_ids.add(wid)

    body_lines: list[str] = []
    for b in blocks:
        if b["BlockType"] != "LINE" or b["Id"] in skip_ids:
            continue
        text = (b.get("Text") or "").strip()
        if text:
            body_lines.append(text)
    body = "\n".join(body_lines).strip()

    table_mds: list[str] = []
    for t in table_blocks:
        rows: dict[int, dict[int, str]] = defaultdict(dict)
        for r in t.get("Relationships", []) or []:
            if r["Type"] != "CHILD":
                continue
            for cid in r["Ids"]:
                cell = by_id.get(cid, {})
                if cell.get("BlockType") != "CELL":
                    continue
                row, col = cell.get("RowIndex", 1), cell.get("ColumnIndex", 1)
                txt: list[str] = []
                for cr in cell.get("Relationships", []) or []:
                    if cr["Type"] == "CHILD":
                        for wid in cr["Ids"]:
                            w = by_id.get(wid, {})
                            if w.get("BlockType") == "WORD":
                                txt.append(w.get("Text", ""))
                rows[row][col] = " ".join(txt).strip()
        if not rows:
            continue
        max_col = max(max(c.keys()) for c in rows.values() if c)
        ordered = [rows[r] for r in sorted(rows)]
        head = ordered[0]
        md = "| " + " | ".join(head.get(c, "") for c in range(1, max_col + 1)) + " |"
        sep = "| " + " | ".join(["---"] * max_col) + " |"
        rest = ["| " + " | ".join(row.get(c, "") for c in range(1, max_col + 1)) + " |" for row in ordered[1:]]
        table_mds.append("\n".join([md, sep, *rest]))

    if table_mds:
        body = (body + "\n\n" + "\n\n".join(table_mds)).strip()
    return body


def run_textract(
    pdf_path: str,
    page_num: int = 1,
    target_longest_image_dim: int = 2048,
    region: str = "",
) -> str:
    """
    Convert page of a PDF file to markdown using AWS Textract.
    https://docs.aws.amazon.com/textract/

    Args:
        pdf_path (str): The local path to the PDF file.
        page_num (int): The page number to process (starting from 1).
        target_longest_image_dim (int): Render dimension passed to
            `olmocr.data.renderpdf.render_pdf_to_base64png`.
        region (str): AWS region (default: $AWS_DEFAULT_REGION or us-east-1).

    Returns:
        str: The OCR result in markdown format.
    """
    image_base64 = render_pdf_to_base64png(
        pdf_path, page_num=page_num, target_longest_image_dim=target_longest_image_dim
    )
    raw = base64.b64decode(image_base64)
    region = region or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
    client = boto3.client("textract", region_name=region)
    resp = client.analyze_document(Document={"Bytes": raw}, FeatureTypes=["TABLES"])
    return _blocks_to_markdown(resp.get("Blocks", []))
