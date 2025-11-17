from typing import List
from src.interface.base_datastore import BaseDatastore, DataItem
import lancedb
from lancedb.table import Table
import pyarrow as pa
import ollama
from concurrent.futures import ThreadPoolExecutor

class Datastore(BaseDatastore):
    DB_PATH = "data/sample_lancedb"
    DB_TABLE_NAME = "rag_table"
    EMBEDDING_MODEL = "mxbai-embed-large"

    def __init__(self):
        # mxbai-embed-large produces 1024-dimensional embeddings
        self.vector_dimensions = 1024
        self.vector_store = lancedb.connect(self.DB_PATH)
        self.table = self._get_table() # Opens or creates the table 

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

    def _convert_item_to_entry(self, item: DataItem) -> dict:
        """Convert a DataItem to match table schema."""
        vector = self.get_vector(item.content)
        return {
            "vector": vector,
            "content": item.content,
            "source": item.source,
        }