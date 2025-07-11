import logging
import json
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from time import time
from retrying import retry
from core.indexing_service.dense_embedding_generator import DenseEmbeddingGenerator
from core.indexing_service.milvus_manager import MilvusManager
from dotenv import load_dotenv
import os

# Set up logger
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

class DenseVectorSearch:
    """
    Performs dense vector search using Milvus, supporting multi-vector queries
    and dynamic retrieval parameters for a state-of-the-art RAG system.
    """
    
    def __init__(
        self,
        milvus_host: str = None,
        milvus_port: str = None,
        collection_name: str = "multi_agent_rag",
        vector_dim: int = 1536,
        max_retries: int = 3,
        # --- ENHANCEMENT: Added flag to control metadata filtering ---
        use_metadata_filters: bool = False
    ):
        """Initialize Milvus connection and embedding generator with retries."""
        load_dotenv()
        self.milvus_host = milvus_host or os.getenv("MILVUS_HOST", "127.0.0.1")
        self.milvus_port = milvus_port or os.getenv("MILVUS_PORT", "19530")
        self.max_retries = max_retries
        self.use_metadata_filters = use_metadata_filters
        
        self._initialize_milvus(collection_name, vector_dim)
        self._initialize_embedder()
    
    @retry(stop_max_attempt_number=3, wait_fixed=5000, retry_on_exception=lambda x: isinstance(x, Exception))
    def _initialize_milvus(self, collection_name: str, vector_dim: int):
        """Initialize MilvusManager with retry logic."""
        try:
            self.milvus_manager = MilvusManager(
                host=self.milvus_host,
                port=self.milvus_port,
                collection_name=collection_name,
                vector_dim=vector_dim
            )
            logger.info(f"Initialized MilvusManager for collection '{collection_name}' with vector_dim={vector_dim}")
        except Exception as e:
            logger.error(f"Failed to initialize MilvusManager: {e}")
            raise
    
    def _initialize_embedder(self):
        """Initialize DenseEmbeddingGenerator."""
        try:
            self.embedder = DenseEmbeddingGenerator()
            logger.info("Initialized DenseEmbeddingGenerator")
        except Exception as e:
            logger.error(f"Failed to initialize DenseEmbeddingGenerator: {e}")
            raise

    def search(self, 
               query_docs: List[Document], 
               retrieval_params: Dict[str, Any],
               metadata_filters: Optional[Dict[str, str]] = None
        ) -> List[Document]:
        """
        Search Milvus using dynamically provided retrieval parameters.
        """
        unique_results = {}
        start_time = time()
        
        if not query_docs:
            logger.warning("No query documents provided for dense search.")
            return []

        top_k = retrieval_params.get("dense_top_k", 30)
        ef_search = retrieval_params.get("ef_dense_search", 150)
        query_type = retrieval_params.get("query_type", "default")

        search_params = {
            "metric_type": "COSINE",
            "params": {"ef": ef_search}
        }
        logger.info(f"Performing dense search with dynamic params for query type '{query_type}': ef={ef_search}, top_k={top_k}")

        # --- ENHANCEMENT: Conditional metadata filtering ---
        expr_filter = None
        if self.use_metadata_filters and metadata_filters:
            expr_filter = self._build_milvus_filter(metadata_filters)
            if expr_filter:
                logger.info(f"Applying Milvus filter: '{expr_filter}'")
        else:
            logger.info("Metadata filtering is deactivated.")

        try:
            embedded_queries = self.embedder.embed_documents(query_docs)
            logger.debug(f"Embedded {len(query_docs)} query documents for dense search.")
            
            hyde_embedding = None
            sub_query_embeddings = []
            
            for q_doc in embedded_queries:
                q_embedding = q_doc.metadata.get('embedding')
                if not q_embedding:
                    logger.warning(f"Document '{q_doc.page_content[:50]}...' is missing an embedding.")
                    continue

                if q_doc.metadata.get('is_hyde', False):
                    hyde_embedding = q_embedding
                    logger.debug(f"Identified HyDE embedding for query '{q_doc.page_content[:50]}...'")
                else:
                    sub_query_embeddings.append(q_embedding)

            embeddings_to_search = []
            if hyde_embedding:
                embeddings_to_search.append(hyde_embedding)
                logger.info("HyDE embedding will be used for search.")
            
            if sub_query_embeddings:
                embeddings_to_search.extend(sub_query_embeddings)
                logger.info(f"{len(sub_query_embeddings)} sub-query embeddings will be used for search.")

            if not embeddings_to_search:
                logger.warning("No valid embeddings found in query documents for dense search.")
                return []
            
            all_milvus_hits = []
            for i, query_emb in enumerate(embeddings_to_search):
                try:
                    query_label = "HyDE" if hyde_embedding and i == 0 else f"Sub-query-{i}"
                    logger.debug(f"Searching with vector for: {query_label}")

                    milvus_results_per_query = self.milvus_manager.collection.search(
                        data=[query_emb],
                        anns_field="dense_vector",
                        param=search_params,
                        limit=top_k,
                        expr=expr_filter, # This will be None if filtering is deactivated
                        output_fields=["id", "content", "metadata", "created_at"]
                    )
                    all_milvus_hits.extend(milvus_results_per_query)
                    logger.debug(f"Received {len(milvus_results_per_query[0]) if milvus_results_per_query else 0} hits for {query_label}.")
                except Exception as e:
                    logger.error(f"Milvus search failed for one query embedding: {e}")
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
                                    "retrieval_method": "dense"
                                }
                            )
                            unique_results[doc_id] = (score, doc)
                        else:
                            logger.debug(f"Skipped duplicate document ID {doc_id} with lower dense score {score}")
                    except Exception as e:
                        logger.error(f"Failed to process search hit (ID: {hit.id if hasattr(hit, 'id') else 'N/A'}): {e}")
                        continue
            
            sorted_results = sorted(unique_results.values(), key=lambda x: x[0], reverse=True)
            results = [doc for _, doc in sorted_results[:top_k]]
            
            if len(results) < top_k and len(sorted_results) < top_k:
                logger.warning(f"Only {len(results)} unique documents retrieved from dense search (target: {top_k})")
            
        except Exception as e:
            logger.error(f"Dense search failed: {e}")
            return []
        
        logger.info(f"Retrieved {len(results)} unique dense documents in {time() - start_time:.2f} seconds.")
        return results

    def _build_milvus_filter(self, metadata_filters: Optional[Dict[str, str]]) -> str:
        """
        Converts a dictionary of metadata filters into a Milvus expression string.
        """
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
                Document(page_content="latest security threats to cloud infrastructure"),
                Document(page_content="mitigate cloud security risks with zero-trust"),
                Document(page_content="Hypothetical document about cloud security threats and zero-trust mitigation. It covers common attack vectors, best practices for identity management, network segmentation, and data protection in modern cloud environments.", metadata={"is_hyde": True})
            ],
            "metadata_filters": {"source": "cloud_report", "date": "2023"},
            "retrieval_parameters": {
                "dense_top_k": 50,
                "sparse_top_k": 70,
                "hybrid_weights_dense": 0.7,
                "hybrid_weights_sparse": 0.3,
                "ef_dense_search": 250,
                "query_type": "long_tail"
            }
        }

        # --- TESTING: Instantiate the searcher with filtering DEACTIVATED ---
        searcher = DenseVectorSearch(use_metadata_filters=False)

        print("--- Testing Complex Query with DYNAMIC Parameters (Metadata Filtering DEACTIVATED) ---")
        results_complex = searcher.search(
            query_docs=simulated_transformer_output["query_docs"],
            metadata_filters=simulated_transformer_output["metadata_filters"],
            retrieval_params=simulated_transformer_output["retrieval_parameters"]
        )
        
        doc_ids_complex = [doc.metadata['doc_id'] for doc in results_complex] if results_complex else []
        unique_ids_complex = len(set(doc_ids_complex))
        print(f"\nRetrieved {len(results_complex)} documents for complex query with {unique_ids_complex} unique IDs.")
        for i, doc in enumerate(results_complex[:5]):
            print(f"Doc {i+1}: ID: {doc.metadata.get('doc_id', 'N/A')}, Score: {doc.metadata.get('score', 'N/A'):.4f}, Method: {doc.metadata.get('retrieval_method', 'N/A')}")
            print(f"  Content: {doc.page_content[:100]}...")
            print(f"  Source: {doc.metadata.get('source', 'N/A')}, Page: {doc.metadata.get('page_number', 'N/A')}")

    except Exception as e:
        logger.error(f"DenseVectorSearch test failed: {e}")
        raise