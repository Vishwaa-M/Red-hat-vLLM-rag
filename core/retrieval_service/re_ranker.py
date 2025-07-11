import logging
from typing import List, Optional
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from time import time
from retrying import retry
import torch

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

logger = get_logger(__name__)

class ReRanker:
    """
    Re-ranks documents using a state-of-the-art, transformer-based cross-encoder,
    with dynamic GPU/CPU support for optimal performance.
    """
    
    def __init__(
        self,
        # --- ENHANCEMENT: Upgraded to a top-tier reranking model ---
        model_name: str = "BAAI/bge-reranker-large",
        max_retries: int = 3,
        device: Optional[str] = None,
        batch_size: int = 8
    ):
        """Initialize the cross-encoder model with retry logic."""
        self.model_name = model_name
        self.max_retries = max_retries
        self.batch_size = batch_size
        # --- ENHANCEMENT: Dynamic device selection (GPU or CPU) ---
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._initialize_model()
    
    @retry(stop_max_attempt_number=3, wait_fixed=2000, retry_on_exception=lambda x: isinstance(x, Exception))
    def _initialize_model(self):
        """Initialize CrossEncoder with retries on the selected device."""
        try:
            self.model = CrossEncoder(self.model_name, max_length=512, device=self.device)
            logger.info(f"Initialized CrossEncoder model: {self.model_name} on device: {self.device.upper()}")
        except Exception as e:
            logger.error(f"Failed to initialize CrossEncoder after {self.max_retries} attempts: {e}")
            raise RuntimeError(f"CrossEncoder initialization failed: {e}")
    
    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """
        Re-rank documents based on relevance to the original query.
        The `top_k` logic is now handled by the calling function (HybridSearch).
        """
        start_time = time()
        
        if not documents:
            logger.warning("No documents provided for re-ranking.")
            return []
        if not query:
            logger.error("No query provided for re-ranking.")
            raise ValueError("Query cannot be empty for re-ranking")
        
        try:
            # Prepare query-document pairs for the model
            pairs = [(query, doc.page_content) for doc in documents]
            logger.info(f"Preparing {len(pairs)} query-document pairs for re-ranking.")
            
            # Compute relevance scores with a progress bar for better UX
            scores = self.model.predict(
                pairs, 
                batch_size=self.batch_size,
                show_progress_bar=True,
                convert_to_tensor=True
            )
            logger.info(f"Computed relevance scores for {len(scores)} documents.")
            
            # Add the new, more accurate score to each document's metadata
            for doc, score in zip(documents, scores):
                doc.metadata['rerank_score'] = float(score)
            
            # Sort documents by the new score in descending order
            sorted_docs = sorted(documents, key=lambda x: x.metadata['rerank_score'], reverse=True)
            
            # The calling function will handle the final top_k selection
            logger.info(f"Re-ranked {len(sorted_docs)} documents in {time() - start_time:.2f} seconds.")
            return sorted_docs
        
        except Exception as e:
            logger.error(f"Re-ranking failed: {e}", exc_info=True)
            raise RuntimeError(f"Re-ranking failed: {e}")

if __name__ == "__main__":
    try:
        # --- This test will now use the powerful bge-reranker-large model ---
        # It will automatically use your GPU if available.
        reranker = ReRanker()
        
        sample_query = "What is Retrieval-Augmented Generation?"
        
        # Documents are intentionally in a suboptimal order
        sample_docs = [
            Document(page_content="A firewall protects a computer network.", metadata={"doc_id": "doc_3", "score": 0.5}),
            Document(page_content="RAG combines large language models with external knowledge bases.", metadata={"doc_id": "doc_2", "score": 0.7}),
            Document(page_content="The term RAG stands for Retrieval-Augmented Generation.", metadata={"doc_id": "doc_1", "score": 0.8}),
        ]

        print(f"\nOriginal order based on initial scores: {[doc.metadata['doc_id'] for doc in sample_docs]}")
        
        results = reranker.rerank(sample_query, sample_docs)

        print("\nRe-ranked order based on cross-encoder scores:")
        for doc in results:
            print(f"  - ID: {doc.metadata['doc_id']}, Re-rank Score: {doc.metadata['rerank_score']:.4f}, Content: '{doc.page_content[:100]}...'")
            
        # Verify that the most relevant documents are now ranked first
        assert results[0].metadata['doc_id'] == 'doc_1'
        assert results[1].metadata['doc_id'] == 'doc_2'
        
        print("\nTest completed successfully. Documents were correctly re-ranked.")

    except Exception as e:
        logger.error(f"Re-ranking test failed: {e}", exc_info=True)
        raise