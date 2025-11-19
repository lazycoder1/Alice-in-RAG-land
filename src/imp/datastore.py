from typing import List
from src.interface.base_datastore import BaseDatastore, DataItem
import lancedb
from lancedb.table import Table
import pyarrow as pa
import ollama
from concurrent.futures import ThreadPoolExecutor
from rank_bm25 import BM25Okapi
import pickle
import os

class Datastore(BaseDatastore):
    DB_PATH = "data/sample_lancedb"
    DB_TABLE_NAME = "rag_table"
    EMBEDDING_MODEL = "mxbai-embed-large"
    BM25_INDEX_PATH = "data/bm25_index.pkl"

    def __init__(self):
        # mxbai-embed-large produces 1024-dimensional embeddings
        self.vector_dimensions = 1024
        self.vector_store = lancedb.connect(self.DB_PATH)
        self.table = self._get_table() # Opens or creates the table 
        self.bm25 = None
        self.corpus = []
        self._load_bm25_index()

    def reset(self) -> Table:
        """Drops the table and creates a new one"""
        # Drop the table 
        try:
            self.vector_store.drop_table(self.DB_TABLE_NAME)
        except Exception as e:
            print(f"Error dropping table: {e}")

        # Create a new table
        schema = pa.schema(
            [
                pa.field("vector" , pa.list_(pa.float32(), self.vector_dimensions)),
                pa.field("content" , pa.utf8()),
                pa.field("source", pa.utf8())
            ]
        );

        # Create the table
        try:
            self.vector_store.create_table(self.DB_TABLE_NAME, schema=schema)
            self.table = self.vector_store.open_table(self.DB_TABLE_NAME)
            print(f"✅ Table Reset/Created: {self.DB_TABLE_NAME} in {self.DB_PATH}")
            # Reset BM25 index as well
            self.bm25 = None
            self.corpus = []
            if os.path.exists(self.BM25_INDEX_PATH):
                os.remove(self.BM25_INDEX_PATH)
            return self.table
        except Exception as e:
            print(f"Error creating table: {e}")
            return None

    def _get_table(self) -> Table:
        """Opens or creates the table"""
        try:
            return self.vector_store.open_table(self.DB_TABLE_NAME)
        except Exception as e:
            print(f"Error opening or creating table: {e}")
            return self.reset()

    def get_vector(self, content: str) -> List[float]:
        """Gets the vector for the content using Ollama"""
        try:
            response = ollama.embed(
                model=self.EMBEDDING_MODEL,
                input=content
            )
            return response["embeddings"][0] if isinstance(response["embeddings"][0], list) else response["embeddings"]
        except Exception as e:
            print(f"Error getting vector: {e}")
            return None

    def add_items(self, items: List[DataItem]) -> None:
        """Adds the data to the table"""
        try:
            with ThreadPoolExecutor(max_workers=10) as executor:
                entries = list(executor.map(self._convert_item_to_entry, items))
            
            self.table.merge_insert(
                "source"
            ).when_matched_update_all().when_not_matched_insert_all().execute(entries)

            print(f"✅ Added {len(items)} items to the table")
            
            # Update BM25 Index
            self._update_bm25_index(items)
            
        except Exception as e:
            print(f"Error adding data: {e}")
            return None
    
    def search(self, query: str, top_k: int = 20) -> List[str]:
        vector = self.get_vector(query)
        results = (
            self.table.search(vector)
            .select(["content", "source"])
            .limit(top_k)
            .to_list()
        )

        result_content = [result.get("content") for result in results]
        return result_content

    def search_bm25(self, query: str, top_k: int = 20) -> List[str]:
        """Search using BM25"""
        if not self.bm25:
            print("⚠️ BM25 index not initialized.")
            return []
        
        tokenized_query = query.lower().split()
        results = self.bm25.get_top_n(tokenized_query, self.corpus, n=top_k)
        return results

    def _convert_item_to_entry(self, item: DataItem) -> dict:
        """Convert a DataItem to match table schema."""
        vector = self.get_vector(item.content)
        return {
            "vector": vector,
            "content": item.content,
            "source": item.source,
        }

    def _update_bm25_index(self, new_items: List[DataItem]):
        """Rebuilds the BM25 index with new items."""
        # In a production system, we would append, but BM25Okapi requires the full corpus to init.
        # So we add new content to self.corpus and rebuild.
        new_contents = [item.content for item in new_items]
        self.corpus.extend(new_contents)
        
        tokenized_corpus = [doc.lower().split() for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        self._save_bm25_index()
        print(f"✅ BM25 Index updated with {len(self.corpus)} documents.")

    def _save_bm25_index(self):
        """Saves the BM25 index and corpus to disk."""
        with open(self.BM25_INDEX_PATH, 'wb') as f:
            pickle.dump({'bm25': self.bm25, 'corpus': self.corpus}, f)

    def _load_bm25_index(self):
        """Loads the BM25 index and corpus from disk."""
        if os.path.exists(self.BM25_INDEX_PATH):
            try:
                with open(self.BM25_INDEX_PATH, 'rb') as f:
                    data = pickle.load(f)
                    self.bm25 = data['bm25']
                    self.corpus = data['corpus']
                print("✅ BM25 Index loaded.")
            except Exception as e:
                print(f"⚠️ Failed to load BM25 index: {e}")