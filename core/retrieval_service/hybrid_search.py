import logging
import numpy as np
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from time import time
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics.pairwise import cosine_similarity

from core.retrieval_service.dense_vector_search import DenseVectorSearch
from core.retrieval_service.sparse_vector_search import SparseVectorSearch
from core.retrieval_service.re_ranker import ReRanker
from core.indexing_service.dense_embedding_generator import DenseEmbeddingGenerator 

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

class HybridSearch:
    """
    Performs a state-of-the-art hybrid search by fusing dense and sparse results
    with Reciprocal Rank Fusion (RRF), followed by a multi-stage re-ranking
    process involving a cross-encoder and Maximal Marginal Relevance (MMR).
    """
    
    def __init__(
        self,
        dense_searcher: DenseVectorSearch = None,
        sparse_searcher: SparseVectorSearch = None,
        reranker: ReRanker = None,
        use_metadata_filters: bool = True
    ):
        """Initialize searchers, re-ranker, and dense embedder for MMR."""
        self.dense_searcher = dense_searcher or DenseVectorSearch(use_metadata_filters=use_metadata_filters)
        self.sparse_searcher = sparse_searcher or SparseVectorSearch(use_metadata_filters=use_metadata_filters)
        self.reranker = reranker or ReRanker()
        self.dense_embedder = DenseEmbeddingGenerator()
    
    def _reciprocal_rank_fusion(self, search_results: List[List[Document]], k: int = 60) -> List[Document]:
        """
        Perform Reciprocal Rank Fusion on multiple lists of search results.
        """
        fused_scores = {}
        all_docs = {doc.metadata['doc_id']: doc for res_list in search_results for doc in res_list}

        for res_list in search_results:
            for rank, doc in enumerate(res_list):
                doc_id = doc.metadata['doc_id']
                if doc_id not in fused_scores:
                    fused_scores[doc_id] = 0
                fused_scores[doc_id] += 1 / (k + rank + 1)
        
        reranked_ids = sorted(fused_scores.keys(), key=lambda id: fused_scores[id], reverse=True)
        
        return [all_docs[doc_id] for doc_id in reranked_ids]

    def _maximal_marginal_relevance(
        self,
        docs_to_rerank: List[Document],
        top_k: int = 10,
        lambda_mult: float = 0.5
    ) -> List[Document]:
        """
        Perform Maximal Marginal Relevance (MMR) using the superior cross-encoder scores for relevance.
        """
        if not docs_to_rerank:
            return []
        
        # We only need embeddings for the diversity calculation.
        # Relevance is now based on the pre-computed 'rerank_score'.
        doc_embeddings = self.dense_embedder.generate_embeddings([doc.page_content for doc in docs_to_rerank])
        
        selected_docs_indices = []
        remaining_docs_indices = list(range(len(docs_to_rerank)))

        # Docs are already sorted by rerank_score, so the first one is the most relevant.
        selected_docs_indices.append(remaining_docs_indices.pop(0))
        
        while len(selected_docs_indices) < min(top_k, len(docs_to_rerank)):
            max_mmr_score = -float('inf')
            next_doc_index = -1
            
            for i in remaining_docs_indices:
                # --- BEST OF THE BEST ENHANCEMENT: Use the cross-encoder's score for relevance ---
                # The rerank_score is a more accurate measure of relevance than cosine similarity.
                # We use a sigmoid function to normalize the score to a [0, 1] range for the MMR formula.
                relevance_score = 1.0 / (1.0 + np.exp(-docs_to_rerank[i].metadata['rerank_score']))
                
                # Calculate diversity as the max similarity to already selected docs
                if selected_docs_indices:
                    selected_embeddings = [doc_embeddings[j] for j in selected_docs_indices]
                    diversity_score = max(cosine_similarity(selected_embeddings, [doc_embeddings[i]]).flatten())
                else:
                    diversity_score = 0
                
                # MMR formula
                mmr_score = lambda_mult * relevance_score - (1 - lambda_mult) * diversity_score
                
                if mmr_score > max_mmr_score:
                    max_mmr_score = mmr_score
                    next_doc_index = i
            
            if next_doc_index != -1:
                selected_docs_indices.append(next_doc_index)
                remaining_docs_indices.remove(next_doc_index)
            else:
                break
        
        return [docs_to_rerank[i] for i in selected_docs_indices]

    def search(
        self,
        query_docs: List[Document],
        original_query: str,
        retrieval_params: Dict[str, Any],
        metadata_filters: Optional[Dict[str, str]] = None
    ) -> List[Document]:
        """
        Orchestrate the advanced hybrid search and multi-stage re-ranking pipeline.
        """
        start_time = time()
        
        if not query_docs or not original_query:
            logger.warning("No query documents or original query provided.")
            return []
        
        # --- 1. Parallel Initial Retrieval ---
        logger.info("Step 1: Starting parallel dense and sparse retrieval.")
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_dense = executor.submit(self.dense_searcher.search, query_docs, retrieval_params, metadata_filters)
            future_sparse = executor.submit(self.sparse_searcher.search, query_docs, retrieval_params, metadata_filters)
            
            dense_docs = future_dense.result()
            sparse_docs = future_sparse.result()
        logger.info(f"Retrieved {len(dense_docs)} dense and {len(sparse_docs)} sparse candidate documents.")

        # --- 2. Reciprocal Rank Fusion (RRF) ---
        logger.info("Step 2: Fusing results with Reciprocal Rank Fusion (RRF).")
        fused_docs = self._reciprocal_rank_fusion([dense_docs, sparse_docs])
        logger.info(f"Fused into {len(fused_docs)} unique documents.")

        # --- 3. Relevance Re-ranking (Cross-Encoder) ---
        rerank_candidate_count = retrieval_params.get("rerank_top_k", 50)
        candidates_for_reranking = fused_docs[:rerank_candidate_count]
        
        logger.info(f"Step 3: Re-ranking top {len(candidates_for_reranking)} candidates for RELEVANCE with cross-encoder.")
        relevance_ranked_docs = self.reranker.rerank(original_query, candidates_for_reranking)
        if not relevance_ranked_docs:
            logger.warning("Re-ranking returned no documents. Aborting pipeline.")
            return []

        # --- 4. Diversity Re-ranking (MMR) ---
        final_top_k = retrieval_params.get("final_top_k", 10)
        # --- BEST OF THE BEST ENHANCEMENT: Make lambda_mult dynamic ---
        lambda_mult = retrieval_params.get("mmr_lambda", 0.7) # Default favors relevance slightly

        logger.info(f"Step 4: Selecting final {final_top_k} documents for DIVERSITY and RELEVANCE with MMR (lambda={lambda_mult}).")
        final_results = self._maximal_marginal_relevance(
            docs_to_rerank=relevance_ranked_docs, 
            top_k=final_top_k, 
            lambda_mult=lambda_mult
        )

        # --- 5. Final Logging and Return ---
        final_ids = {doc.metadata.get("doc_id") for doc in final_results}
        dense_final = sum(1 for doc_id in final_ids if any(d.metadata.get("doc_id") == doc_id for d in dense_docs))
        sparse_final = sum(1 for doc_id in final_ids if any(d.metadata.get("doc_id") == doc_id for d in sparse_docs))
        
        logger.info(f"Final results composition: Contributed by {dense_final} unique dense docs and {sparse_final} unique sparse docs.")
        logger.info(f"Completed hybrid search pipeline in {time() - start_time:.2f} seconds, returning {len(final_results)} documents.")
        
        return final_results

if __name__ == "__main__":
    from unittest.mock import MagicMock

    # --- Setup Mocks for Standalone Testing ---
    mock_dense_searcher = MagicMock(spec=DenseVectorSearch)
    mock_sparse_searcher = MagicMock(spec=SparseVectorSearch)
    mock_reranker = MagicMock(spec=ReRanker)

    # Mock dense results with embeddings for MMR testing
    mock_dense_docs = [
        Document(page_content="RAG is Retrieval-Augmented Generation.", metadata={"doc_id": "doc_1", "score": 0.9, "embedding": [0.1, 0.9]}),
        Document(page_content="A firewall protects networks.", metadata={"doc_id": "doc_2", "score": 0.8, "embedding": [0.8, 0.2]}),
        Document(page_content="MMR stands for Maximal Marginal Relevance.", metadata={"doc_id": "doc_3", "score": 0.7, "embedding": [0.5, 0.5]})
    ]
    mock_dense_searcher.search.return_value = mock_dense_docs

    # Mock sparse results (doc_2 appears here again)
    mock_sparse_docs = [
        Document(page_content="A firewall is a security device.", metadata={"doc_id": "doc_2", "score": 55.0, "embedding": [0.8, 0.2]}),
        Document(page_content="The C programming language is old.", metadata={"doc_id": "doc_4", "score": 40.0, "embedding": [0.9, 0.1]}),
    ]
    mock_sparse_searcher.search.return_value = mock_sparse_docs
    
    # Mock reranker to add a 'rerank_score'
    def mock_rerank_func(query, docs):
        # Simulate higher score for more relevant docs
        for doc in docs:
            if "RAG" in doc.page_content or "firewall" in doc.page_content:
                doc.metadata['rerank_score'] = 10.0
            else:
                doc.metadata['rerank_score'] = 1.0
        return sorted(docs, key=lambda x: x.metadata['rerank_score'], reverse=True)

    mock_reranker.rerank.side_effect = mock_rerank_func

    try:
        searcher = HybridSearch(
            dense_searcher=mock_dense_searcher,
            sparse_searcher=mock_sparse_searcher,
            reranker=mock_reranker
        )
        
        sample_query = "What is a firewall and how does it relate to RAG?"
        sample_query_docs = [Document(page_content=sample_query)]
        sample_retrieval_params = {
            "dense_top_k": 3,
            "sparse_top_k": 2,
            "rerank_top_k": 4, 
            "final_top_k": 3, # Request 3 docs after MMR
            "mmr_lambda": 0.6 # Slightly favor relevance
        }

        print("--- Testing 'Best of the Best' Hybrid Search with RRF and Enhanced MMR ---")
        final_docs = searcher.search(
            query_docs=sample_query_docs,
            original_query=sample_query,
            retrieval_params=sample_retrieval_params
        )

        print(f"\nFinal retrieved {len(final_docs)} documents:")
        for doc in final_docs:
            print(f"  - ID: {doc.metadata['doc_id']}, Content: '{doc.page_content}'")
            
        assert len(final_docs) == sample_retrieval_params["final_top_k"]
        print("\nTest completed successfully.")

    except Exception as e:
        logger.error(f"HybridSearch test failed: {e}", exc_info=True)
        raise