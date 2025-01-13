from pathlib import Path
from typing import List
import pandas as pd
import pdfplumber
from src.embeddings import VectorStore


class PDFProcessor:
    """Handles PDF text and table extraction."""

    def __init__(self):
        """Initialize the PDFProcessor."""
        pass

    @staticmethod
    def extract_text(pdf_path: Path) -> List[str]:
        """Extract plain text from a PDF file."""
        try:
            from pypdf import PdfReader
            reader = PdfReader(pdf_path)
            return [page.extract_text().strip() for page in reader.pages if page.extract_text()]
        except Exception as e:
            print(f"Error extracting text from {pdf_path.name}: {e}")
            return []

    @staticmethod
    def extract_tables(pdf_path: Path) -> List[str]:
        """Extract tables from a PDF file."""
        try:
            tables_as_strings = []
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    tables = page.extract_tables()
                    for table in tables:
                        df = pd.DataFrame(table)
                        tables_as_strings.append(df.to_csv(index=False, header=False))
            return tables_as_strings
        except Exception as e:
            print(f"Error extracting tables from {pdf_path.name}: {e}")
            return []

    def extract_content(self, pdf_path: Path) -> List[str]:
        """Extract all content (text + tables) from a PDF."""
        text = self.extract_text(pdf_path)
        tables = self.extract_tables(pdf_path)
        return text + tables


def process_pdf_directory(directory: str) -> None:
    """Process all PDFs in a directory and store their embeddings."""
    pdf_dir = Path(directory)
    processor = PDFProcessor()
    store = VectorStore()

    for pdf_path in pdf_dir.glob("*.pdf"):
        print(f"\nProcessing {pdf_path.name}...")

        # Extract content
        content = processor.extract_content(pdf_path)

        # Skip if no content
        if not content:
            print(f"No content found in {pdf_path.name}, skipping.")
            continue

        # Prepare metadata
        metadata = [{"filename": pdf_path.name, "page": i + 1} for i in range(len(content))]
        document_ids = [f"{pdf_path.name}_page_{i + 1}" for i in range(len(content))]

        # Check for existing embeddings
        existing_embeddings = store.collection.get(ids=document_ids)
        if existing_embeddings and existing_embeddings.get("ids"):
            print(f"Embeddings already exist for {pdf_path.name}, skipping.")
            continue

        # Create and store embeddings
        embeddings = store.create_embeddings(content)
        store.store_embeddings(embeddings, metadata, documents=content)
        print(f"Successfully processed and stored embeddings for {pdf_path.name}.")


if __name__ == "__main__":
    process_pdf_directory("datasources")
