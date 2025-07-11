import logging
import json
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from time import time
from core.indexing_service.milvus_manager import MilvusManager
from core.indexing_service.sparse_embedding_generator import SparseEmbeddingGenerator
from dotenv import load_dotenv
import os
from retrying import retry

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
    return logger

logger = get_logger(__name__)

class SparseVectorSearch:
    """
    Performs true sparse vector search using a SPLADE model and Milvus
    for state-of-the-art retrieval.
    """
    
    def __init__(
        self,
        milvus_host: str = None,
        milvus_port: str = None,
        collection_name: str = "multi_agent_rag",
        max_retries: int = 3,
        use_metadata_filters: bool = False
    ):
        """Initialize Milvus connection and sparse embedding generator."""
        load_dotenv()
        self.milvus_host = milvus_host or os.getenv("MILVUS_HOST", "127.0.0.1")
        self.milvus_port = milvus_port or os.getenv("MILVUS_PORT", "19530")
        self.max_retries = max_retries
        self.use_metadata_filters = use_metadata_filters
        
        self._initialize_milvus(collection_name)
        # --- ENHANCEMENT: Initialize the sparse embedder for query conversion ---
        self._initialize_embedder()

    @retry(stop_max_attempt_number=3, wait_fixed=5000, retry_on_exception=lambda x: isinstance(x, Exception))
    def _initialize_milvus(self, collection_name: str):
        """Initialize MilvusManager with retry logic."""
        try:
            self.milvus_manager = MilvusManager(collection_name=collection_name)
            logger.info(f"Initialized MilvusManager for collection '{collection_name}' for sparse search.")
        except Exception as e:
            logger.error(f"Failed to initialize MilvusManager for sparse search: {e}")
            raise
    
    def _initialize_embedder(self):
        """Initialize SparseEmbeddingGenerator."""
        try:
            self.embedder = SparseEmbeddingGenerator()
            logger.info("Initialized SparseEmbeddingGenerator.")
        except Exception as e:
            logger.error(f"Failed to initialize SparseEmbeddingGenerator: {e}")
            raise

    def search(self, 
               query_docs: List[Document], 
               retrieval_params: Dict[str, Any],
               metadata_filters: Optional[Dict[str, str]] = None
        ) -> List[Document]:
        """
        Search Milvus by converting queries to sparse vectors and performing a similarity search.
        """
        unique_results = {}
        start_time = time()
        
        if not query_docs:
            logger.warning("No query documents provided for sparse search.")
            return []

        top_k = retrieval_params.get("sparse_top_k", 30)
        query_type = retrieval_params.get("query_type", "default")
        logger.info(f"Performing sparse vector search for query type '{query_type}' with dynamic top_k={top_k}")

        expr_filter = None
        if self.use_metadata_filters and metadata_filters:
            expr_filter = self._build_milvus_filter(metadata_filters)
            if expr_filter:
                logger.info(f"Applying Milvus filter for sparse search: '{expr_filter}'")
        else:
            logger.info("Metadata filtering is deactivated for sparse search.")

        try:
            # 1. Filter out HyDE queries and get text for sparse embedding
            texts_to_embed = [
                q_doc.page_content for q_doc in query_docs 
                if not q_doc.metadata.get('is_hyde', False) and q_doc.page_content.strip()
            ]

            if not texts_to_embed:
                logger.warning("No valid text queries (non-HyDE) found for sparse embedding.")
                return []
            
            # 2. Convert text queries to sparse vectors using the SPLADE model
            sparse_query_vectors = self.embedder.generate_embeddings(texts_to_embed)
            logger.debug(f"Generated {len(sparse_query_vectors)} sparse query vectors.")

            # 3. Define search parameters for sparse vector search
            # The metric type must match the sparse index metric ('IP' for Inner Product)
            search_params = {"metric_type": "IP", "params": {}}

            all_milvus_hits = []
            for i, sparse_vector in enumerate(sparse_query_vectors):
                if not sparse_vector:
                    logger.warning(f"Skipping empty sparse vector for query: '{texts_to_embed[i][:50]}...'")
                    continue
                try:
                    # 4. Perform search using the correct anns_field and parameters
                    search_results_per_query = self.milvus_manager.collection.search(
                        data=[sparse_vector],
                        anns_field="sparse_vector", # Search the correct field
                        param=search_params,
                        limit=top_k,
                        expr=expr_filter,
                        output_fields=["id", "content", "metadata", "created_at"]
                    )
                    all_milvus_hits.extend(search_results_per_query)
                    logger.debug(f"Received {len(search_results_per_query[0]) if search_results_per_query else 0} hits for sparse query '{texts_to_embed[i][:50]}...'")
                except Exception as e:
                    logger.error(f"Milvus sparse vector search failed for query '{texts_to_embed[i][:50]}...': {e}")
                    continue
            
            for hits_group in all_milvus_hits:
                for hit in hits_group:
                    try:
                        doc_id = hit.id
                        score = hit.score
                        if doc_id not in unique_results or score > unique_results[doc_id][0]:
                            metadata = json.loads(hit.entity.get("metadata"))
                            doc = Document(
                                page_content=hit.entity.get("content"),
                                metadata={
                                    **metadata,
                                    "score": score,
                                    "created_at": hit.entity.get("created_at"),
                                    "doc_id": doc_id,
                                    "retrieval_method": "sparse"
                                }
                            )
                            unique_results[doc_id] = (score, doc)
                    except Exception as e:
                        logger.error(f"Failed to process search hit (ID: {hit.id if hasattr(hit, 'id') else 'N/A'}): {e}")
                        continue
            
            sorted_results = sorted(unique_results.values(), key=lambda x: x[0], reverse=True)
            results = [doc for _, doc in sorted_results[:top_k]]
            
            if len(results) < top_k and len(sorted_results) < top_k:
                logger.warning(f"Only {len(results)} unique documents retrieved from sparse search (target: {top_k})")
                
        except Exception as e:
            logger.error(f"Sparse search failed: {e}", exc_info=True)
            return []
        
        logger.info(f"Retrieved {len(results)} unique sparse documents in {time() - start_time:.2f} seconds.")
        return results

    def _build_milvus_filter(self, metadata_filters: Optional[Dict[str, str]]) -> str:
        if not metadata_filters:
            return ""
        filter_parts = []
        for key, value in metadata_filters.items():
            escaped_value = str(value).replace("'", "''")
            if key == "date":
                filter_parts.append(f"created_at LIKE '{escaped_value}%'")
            else:
                filter_obj = {key: escaped_value}
                json_str = json.dumps(filter_obj)
                substring_to_find = json_str[1:-1]
                filter_parts.append(f"metadata like '%{substring_to_find}%'")
        return " and ".join(filter_parts) if filter_parts else ""

if __name__ == "__main__":
    try:
        simulated_transformer_output = {
            "query_docs": [
                Document(page_content="What is RAG and how does it function?"),
                Document(page_content="Explain the concept of modular RAG systems."),
            ],
            "retrieval_parameters": {"sparse_top_k": 10}
        }
        
        searcher = SparseVectorSearch(use_metadata_filters=False)
        
        print("--- Testing True Sparse Vector Search ---")
        results = searcher.search(
            query_docs=simulated_transformer_output["query_docs"],
            retrieval_params=simulated_transformer_output["retrieval_parameters"]
        )
        
        unique_ids = len(set(doc.metadata['doc_id'] for doc in results))
        print(f"\nRetrieved {len(results)} documents for sparse query with {unique_ids} unique IDs.")
        for i, doc in enumerate(results[:5]):
            print(f"Doc {i+1}: ID: {doc.metadata.get('doc_id', 'N/A')}, Score: {doc.metadata.get('score', 'N/A'):.4f}, Method: {doc.metadata.get('retrieval_method', 'N/A')}")
            print(f"  Content: {doc.page_content[:100]}...")
            print(f"  Source: {doc.metadata.get('source', 'N/A')}, Page: {doc.metadata.get('page_number', 'N/A')}")

    except Exception as e:
        logger.error(f"SparseVectorSearch test failed: {e}", exc_info=True)
        raise