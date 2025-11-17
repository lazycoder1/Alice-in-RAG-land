from src.interface.base_datastore import BaseDatastore
from src.interface.base_retriever import BaseRetriever
import cohere


class Retriever(BaseRetriever):
    def __init__(self, datastore: BaseDatastore):
        self.datastore = datastore

    def search(self, query: str, top_k: int = 20) -> list[str]:
        # Retrieve more chunks for better context coverage
        search_results = self.datastore.search(query, top_k=top_k)
        return search_results