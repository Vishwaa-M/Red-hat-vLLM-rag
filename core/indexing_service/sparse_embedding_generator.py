import logging
import os
from typing import List, Dict
from dotenv import load_dotenv
import torch
import numpy as np
from langchain_core.documents import Document
from transformers import AutoModelForMaskedLM, AutoTokenizer
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import HfApi

# Configure logging with structured format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

class SparseEmbeddingGenerator:
    """Enterprise-grade class for generating sparse embeddings using a SPLADE model."""
    
    def __init__(
        self,
        model_name: str | None = None,
        batch_size: int = 8,
        max_retries: int = 3,
        device: str | None = None,
        hf_token: str | None = None
    ):
        """Initialize the sparse embedding generator with configuration from .env or parameters."""
        # Load environment variables
        load_dotenv()
        
        # Use provided values or fallback to environment variables
        self.model_name = model_name or os.getenv(
            "SPARSE_EMBEDDING_MODEL", "naver/splade-cocondenser-ensembledistil"
        )
        self.batch_size = self._validate_batch_size(batch_size)
        self.max_retries = max_retries
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.hf_token = hf_token or os.getenv("HUGGINGFACE_TOKEN")
        logger.info(f"Initializing SparseEmbeddingGenerator with model: {self.model_name}, device: {self.device}")

        # Validate configurations
        if not self.model_name:
            logger.error("Model name is missing. Set SPARSE_EMBEDDING_MODEL in .env or pass as parameter.")
            raise ValueError("Model name is required")

        # Initialize model and tokenizer
        try:
            # Check if model exists on Hugging Face
            api = HfApi()
            model_info = api.model_info(self.model_name, token=self.hf_token)
            logger.info(f"Verified model exists: {self.model_name}")
            
            kwargs = {"token": self.hf_token} if self.hf_token else {}
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, **kwargs)
            self.model = AutoModelForMaskedLM.from_pretrained(self.model_name, **kwargs).to(self.device)
            self.model.eval()  # Set to evaluation mode
            logger.info(f"Loaded SPLADE model: {self.model_name} on {self.device}")
        except Exception as e:
            logger.error(f"Failed to initialize SPLADE model: {e}")
            fallback_model = "naver/splade-v3"
            logger.info(f"Attempting fallback model: {fallback_model}")
            try:
                kwargs = {"token": self.hf_token} if self.hf_token else {}
                self.tokenizer = AutoTokenizer.from_pretrained(fallback_model, **kwargs)
                self.model = AutoModelForMaskedLM.from_pretrained(fallback_model, **kwargs).to(self.device)
                self.model.eval()
                self.model_name = fallback_model
                logger.info(f"Successfully loaded fallback model: {self.model_name}")
            except Exception as fallback_e:
                logger.error(f"Failed to load fallback model {fallback_model}: {fallback_e}")
                raise ValueError(
                    f"Model initialization failed for {self.model_name} and fallback {fallback_model}. "
                    f"Original error: {e}. Fallback error: {fallback_e}"
                )

    def _validate_batch_size(self, batch_size: int) -> int:
        """Validate batch size is positive and reasonable."""
        if not isinstance(batch_size, int) or batch_size <= 0:
            logger.error(f"Invalid batch_size: {batch_size}. Must be a positive integer.")
            raise ValueError("batch_size must be a positive integer")
        if batch_size > 32:
            logger.warning(f"Large batch_size ({batch_size}) may cause memory issues. Consider reducing.")
        return batch_size

    def generate_embeddings(self, texts: List[str]) -> List[Dict[int, float]]:
        """Generate sparse embeddings for a list of texts using the SPLADE model."""
        if not texts:
            logger.warning("No texts provided for embedding")
            return []

        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            attempt = 0
            while attempt < self.max_retries:
                try:
                    # Tokenize batch
                    inputs = self.tokenizer(
                        batch_texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512
                    ).to(self.device)
                    logger.debug(f"Tokenized batch of {len(batch_texts)} texts")

                    # Generate embeddings
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        logits = outputs.logits  # Shape: (batch_size, seq_len, vocab_size)

                    # Compute sparse vectors
                    batch_embeddings = []
                    for b in range(logits.size(0)):
                        # Apply ReLU and max pooling over sequence
                        weights = torch.relu(logits[b]).max(dim=0).values  # Shape: (vocab_size,)
                        # Get non-zero indices and values
                        non_zero_indices = torch.nonzero(weights, as_tuple=False).squeeze()
                        non_zero_weights = weights[non_zero_indices].cpu().numpy()
                        non_zero_indices = non_zero_indices.cpu().numpy()
                        # Create sparse embedding as dictionary
                        sparse_emb = {int(idx): float(weight) for idx, weight in zip(non_zero_indices, non_zero_weights)}
                        batch_embeddings.append(sparse_emb)
                    embeddings.extend(batch_embeddings)
                    logger.debug(f"Generated sparse embeddings for batch of {len(batch_texts)} texts")
                    break
                except Exception as e:
                    attempt += 1
                    logger.warning(f"Error generating embeddings for batch, attempt {attempt}/{self.max_retries}: {e}")
                    if attempt == self.max_retries:
                        logger.error(f"Failed to generate embeddings after {self.max_retries} attempts: {e}")
                        # Return empty embeddings for failed batch
                        embeddings.extend([{} for _ in batch_texts])
                        break
        return embeddings

    def embed_documents(self, documents: List[Document]) -> List[Document]:
        """Embed a list of LangChain documents and store sparse embeddings in metadata."""
        if not documents:
            logger.warning("No documents provided for embedding")
            return []

        # Validate documents
        for doc in documents:
            if not isinstance(doc, Document):
                logger.error(f"Invalid document type: {type(doc)}. Expected langchain_core.documents.Document")
                raise ValueError("All inputs must be langchain_core.documents.Document instances")
            if not hasattr(doc, 'metadata'):
                logger.error("Document missing metadata field")
                raise ValueError("Document must have a metadata field")

        texts = [doc.page_content for doc in documents]
        try:
            # Parallelize embedding generation
            with ThreadPoolExecutor(max_workers=4) as executor:
                future = executor.submit(self.generate_embeddings, texts)
                embeddings = future.result()
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise ValueError(f"Embedding generation failed: {e}")

        if len(embeddings) != len(documents):
            logger.error(f"Mismatch in documents ({len(documents)}) and embeddings ({len(embeddings)})")
            raise ValueError("Embedding generation failed: mismatch in lengths")

        for doc, embedding in zip(documents, embeddings):
            doc.metadata['sparse_embedding'] = embedding
            # Add sparsity metric for debugging
            doc.metadata['sparse_embedding_size'] = len(embedding)
        logger.info(f"Embedded {len(documents)} documents with sparse embeddings")
        return documents

    def __enter__(self):
        """Support context manager for resource management."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources on exit."""
        # Clear model from GPU memory if applicable
        if self.device.startswith("cuda"):
            try:
                del self.model
                torch.cuda.empty_cache()
                logger.debug("Cleared model from GPU memory")
            except Exception as e:
                logger.warning(f"Failed to clear GPU memory: {e}")
        logger.debug("Closed SparseEmbeddingGenerator")

if __name__ == "__main__":
    try:
        sample_docs = [
            Document(
                page_content="Linux Bash commands manage files and processes.",
                metadata={"doc_id": "1", "content_type": "text", "source": "doc1"}
            ),
            Document(
                page_content="| Command | Description |\n| ls | Lists files |",
                metadata={"doc_id": "2", "content_type": "table", "source": "doc1"}
            )
        ]
        with SparseEmbeddingGenerator() as embedder:
            embedded_docs = embedder.embed_documents(sample_docs)
            for doc in embedded_docs:
                logger.info(f"Document ID: {doc.metadata['doc_id']}, Sparse embedding size: {doc.metadata['sparse_embedding_size']}")
                logger.debug(f"Sparse embedding (sample): {dict(list(doc.metadata['sparse_embedding'].items())[:5])}")
                print(f"Embedding for Document ID {doc.metadata['doc_id']}: {doc.metadata['sparse_embedding']}")
    except Exception as e:
        logger.error(f"Failed to generate sparse embeddings: {e}")
        raise