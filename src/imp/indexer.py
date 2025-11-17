import os
from typing import List
from pathlib import Path
from src.interface.base_datastore import DataItem
from src.interface.base_indexer import BaseIndexer
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker, DocChunk

class Indexer(BaseIndexer):
    def __init__(self):
        self.converter = DocumentConverter()
        # Use 1024 tokens - larger chunks preserve more context
        self.chunker = HybridChunker(max_tokens=1024)
        # Disable tokenizers parallelism to avoid OOM errors.
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def index(self, document_paths: List[str]) -> List[DataItem]:
        items = []
        for document_path in document_paths:
            try:
                # Handle .txt files by converting to markdown format (which DocumentConverter supports)
                if Path(document_path).suffix.lower() == '.txt':
                    with open(document_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    # Convert text to markdown and use DocumentConverter
                    from docling.datamodel.base_models import InputFormat
                    document = self.converter.convert_string(
                        content=text,
                        format=InputFormat.MD,
                        name=Path(document_path).name
                    ).document
                else:
                    document = self.converter.convert(document_path).document
                # Convert iterator to list for overlap processing
                chunks = list(self.chunker.chunk(document))
                items.extend(self._items_from_chunks(chunks, document_path))
            except Exception as e:
                print(f"⚠️  Error processing {document_path}: {e}")
                continue
        print(f"✅ Indexed {len(items)} chunks from {len(document_paths)} documents")
        return items

    def _items_from_chunks(self, chunks: List[DocChunk], document_path: str) -> List[DataItem]:
        """Convert chunks to DataItems."""
        items = []
        filename = Path(document_path).name
        for i, chunk in enumerate(chunks):
            # Handle headings if they exist, otherwise use empty string
            headings = chunk.meta.headings if hasattr(chunk.meta, 'headings') and chunk.meta.headings else []
            content_headings = "## " + ", ".join(headings) if headings else ""
            content_text = f"{content_headings}\n{chunk.text}" if content_headings else chunk.text
            source = f"{filename}:{i}"
            item = DataItem(content=content_text, source=source)
            items.append(item)
        return items
