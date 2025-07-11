import logging
import os
from typing import List
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from langchain_core.documents import Document
import urllib.parse

# Configure logging with structured format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

class DenseEmbeddingGenerator:
    """Enterprise-grade class for generating embeddings using Azure OpenAI API."""
    
    def __init__(
        self,
        endpoint: str | None = None,
        api_key: str | None = None,
        deployment: str | None = None,
        api_version: str | None = None,
        batch_size: int = 8,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        timeout: int = 30
    ):
        """Initialize the embedding generator with configuration from .env or parameters."""
        # Load environment variables from .env file
        load_dotenv()
        
        # Use provided values or fallback to environment variables
        self.endpoint = endpoint or os.getenv("AZURE_OPENAI_EMBEDDINGS_ENDPOINT", "https://seyon-openai-embedding.openai.azure.com/")
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.deployment = deployment or os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
        self.api_version = api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")
        self.batch_size = self._validate_batch_size(batch_size)
        self.timeout = timeout
        logger.info(f"Initializing DenseEmbeddingGenerator with Azure endpoint: {self.endpoint}")

        # Validate critical configurations
        if not self.api_key:
            logger.error("API key is missing. Set AZURE_OPENAI_API_KEY in .env or pass as parameter.")
            raise ValueError("API key is required")
        if not self.endpoint:
            logger.error("Endpoint is missing. Set AZURE_OPENAI_EMBEDDINGS_ENDPOINT in .env or pass as parameter.")
            raise ValueError("Endpoint is required")
        if not self.deployment:
            logger.error("Deployment is missing. Set AZURE_OPENAI_EMBEDDING_DEPLOYMENT in .env or pass as parameter.")
            raise ValueError("Deployment is required")
        if not self.api_version:
            logger.error("API version is missing. Set AZURE_OPENAI_API_VERSION in .env or pass as parameter.")
            raise ValueError("API version is required")

        # Decode and clean the endpoint
        self.endpoint = urllib.parse.unquote(self.endpoint).rstrip('/')
        self.azure_endpoint = (
            f"{self.endpoint}/openai/deployments/"
            f"{self.deployment}/embeddings?api-version={self.api_version}"
        )
        logger.info(f"Constructed Azure embedding endpoint: {self.azure_endpoint}")

        # Set up HTTP session with retry logic
        self.session = requests.Session()
        retries = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST", "GET"]
        )
        self.session.mount("http://", HTTPAdapter(max_retries=retries))
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

        # Verify server connectivity
        self._verify_server()

    def _validate_batch_size(self, batch_size: int) -> int:
        """Validate batch size is positive and reasonable."""
        if not isinstance(batch_size, int) or batch_size <= 0:
            logger.error(f"Invalid batch_size: {batch_size}. Must be a positive integer.")
            raise ValueError("batch_size must be a positive integer")
        if batch_size > 32:
            logger.warning(f"Large batch_size ({batch_size}) may cause memory issues. Consider reducing.")
        return batch_size

    def _verify_server(self) -> None:
        """Verify connection to the Azure OpenAI embedding server."""
        try:
            # Azure OpenAI doesn't have a direct /v1/models endpoint, so test with a small embedding request
            response = self.session.post(
                self.azure_endpoint,
                headers={
                    "Content-Type": "application/json",
                    "api-key": self.api_key
                },
                json={"input": "test"},
                timeout=self.timeout
            )
            response.raise_for_status()
            logger.info(f"Connected to Azure OpenAI embedding server at {self.endpoint}")
        except requests.RequestException as e:
            logger.error(f"Failed to connect to embedding server: {e}")
            raise ValueError(f"Embedding server unreachable: {e}")

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts using Azure OpenAI API with retry logic."""
        if not texts:
            logger.warning("No texts provided for embedding")
            return []

        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            try:
                payload = {"input": batch_texts}
                headers = {
                    "Content-Type": "application/json",
                    "api-key": self.api_key
                }
                logger.debug(f"Requesting embeddings for batch of {len(batch_texts)} texts from {self.azure_endpoint}")
                response = self.session.post(
                    self.azure_endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                data = response.json()
                batch_embeddings = [item["embedding"] for item in data["data"]]
                embeddings.extend(batch_embeddings)
                logger.debug(f"Generated embeddings for batch of {len(batch_texts)} texts")
            except requests.RequestException as e:
                logger.error(f"Error generating embeddings for batch: {e}")
                raise ValueError(f"Embedding generation failed: {e}")
        return embeddings

    def embed_documents(self, documents: List[Document]) -> List[Document]:
        """Embed a list of LangChain documents and store embeddings in metadata."""
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
            embeddings = self.generate_embeddings(texts)
        except ValueError as e:
            logger.error(f"Embedding generation failed: {e}")
            raise

        if len(embeddings) != len(documents):
            logger.error(f"Mismatch in documents ({len(documents)}) and embeddings ({len(embeddings)})")
            raise ValueError("Embedding generation failed: mismatch in lengths")

        for doc, embedding in zip(documents, embeddings):
            doc.metadata['embedding'] = embedding
        logger.info(f"Embedded {len(documents)} documents")
        return documents

    def __enter__(self):
        """Support context manager for session management."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close HTTP session on exit."""
        self.session.close()
        logger.debug("Closed HTTP session")

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
        with DenseEmbeddingGenerator() as embedder:
            embedded_docs = embedder.embed_documents(sample_docs)
            for doc in embedded_docs:
                logger.info(f"Document ID: {doc.metadata['doc_id']}, Embedding length: {len(doc.metadata['embedding'])}")
                print(f"Embedding for Document ID {doc.metadata['doc_id']}: {doc.metadata['embedding']}")
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}")
        raise