from src.interface.base_datastore import BaseDatastore
from src.interface.base_retriever import BaseRetriever
import cohere


class Retriever(BaseRetriever):
    def __init__(self, datastore: BaseDatastore):
        self.datastore = datastore

    def search(self, query: str, top_k: int = 20) -> list[str]:
        # 1. Retrieve a larger pool of candidates using vector search
        initial_results = self.datastore.search(query, top_k=100)
        
        if not initial_results:
            return []

        # 2. Rerank the results using Cohere
        try:
            co = cohere.Client() # Expects COHERE_API_KEY in env
            rerank_results = co.rerank(
                query=query,
                documents=initial_results,
                top_n=top_k,
                model="rerank-english-v3.0"
            )
            
            # 3. Extract the content from the reranked results
            final_results = [result.document.text for result in rerank_results.results]
            return final_results
            
        except Exception as e:
            print(f"⚠️  Reranking failed: {e}. Falling back to vector search results.")
            return initial_results[:top_k]