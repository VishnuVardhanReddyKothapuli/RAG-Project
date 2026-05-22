"""
RAG Application — PDF Loader
==============================
Extracts text from PDF files using PyMuPDF (fitz).
Preserves page-level metadata for source citation.
"""

from typing import List, Optional
import fitz  # PyMuPDF
from langchain_core.documents import Document

from utils.logger import get_logger

logger = get_logger(__name__)


def load_pdf(file_path: str) -> List[Document]:
    """
    Load a PDF from a file path and extract text page-by-page.

    Args:
        file_path: Absolute or relative path to the PDF file.

    Returns:
        List of Document objects, one per page, with metadata.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If the PDF cannot be parsed.
    """
    documents: List[Document] = []

    try:
        doc = fitz.open(file_path)
        filename = doc.name.split("/")[-1].split("\\")[-1]

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")

            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": filename,
                            "page": page_num + 1,         # 1-indexed
                            "total_pages": len(doc),
                        },
                    )
                )

        doc.close()
        logger.info(
            f"Loaded '{filename}': {len(documents)} pages with text "
            f"(out of {len(doc)} total pages)"
        )

    except FileNotFoundError:
        logger.error(f"PDF file not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Failed to parse PDF '{file_path}': {e}")
        raise RuntimeError(f"Could not parse PDF: {e}") from e

    return documents


def load_uploaded_pdfs(
    uploaded_files: list,
    existing_hashes: Optional[set] = None,
) -> tuple[List[Document], set, list[str]]:
    """
    Process Streamlit UploadedFile objects into LangChain Documents.

    Args:
        uploaded_files: List of Streamlit UploadedFile objects.
        existing_hashes: Set of file hashes already indexed (for deduplication).

    Returns:
        Tuple of:
        - List of Document objects from all new PDFs.
        - Updated set of file hashes.
        - List of file names that were skipped (duplicates).
    """
    import hashlib

    if existing_hashes is None:
        existing_hashes = set()

    all_documents: List[Document] = []
    skipped_files: list[str] = []

    for uploaded_file in uploaded_files:
        filename = uploaded_file.name

        try:
            # Read file bytes
            file_bytes = uploaded_file.read()
            uploaded_file.seek(0)  # Reset for potential re-read

            # Deduplication check
            file_hash = hashlib.sha256(file_bytes).hexdigest()
            if file_hash in existing_hashes:
                logger.info(f"Skipping duplicate file: {filename}")
                skipped_files.append(filename)
                continue

            # Parse PDF from bytes
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            page_documents: List[Document] = []

            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text")

                if text.strip():
                    page_documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": filename,
                                "page": page_num + 1,
                                "total_pages": len(doc),
                                "file_hash": file_hash,
                            },
                        )
                    )

            doc.close()

            if page_documents:
                all_documents.extend(page_documents)
                existing_hashes.add(file_hash)
                logger.info(
                    f"Processed '{filename}': {len(page_documents)} pages extracted"
                )
            else:
                logger.warning(f"No text found in '{filename}' — may be image-based PDF")
                skipped_files.append(filename)

        except Exception as e:
            logger.error(f"Failed to process '{filename}': {e}")
            skipped_files.append(filename)

    logger.info(
        f"Total: {len(all_documents)} pages from "
        f"{len(uploaded_files) - len(skipped_files)} files processed, "
        f"{len(skipped_files)} skipped"
    )

    return all_documents, existing_hashes, skipped_files
