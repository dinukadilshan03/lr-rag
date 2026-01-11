from pathlib import Path

from llama_index.readers.file import PDFReader


def load_pdf(pdf_path: str | Path) -> str:
    """
    Load a PDF file and return all extracted text as a single string.
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Not a PDF file: {pdf_path}")

    reader = PDFReader()
    documents = reader.load_data(file=pdf_path)

    if not documents:
        raise RuntimeError(f"No content extracted from PDF: {pdf_path}")

    full_text = "\n".join(doc.text for doc in documents if doc.text)

    if not full_text.strip():
        raise RuntimeError(f"Extracted empty text from PDF: {pdf_path}")

    return full_text
