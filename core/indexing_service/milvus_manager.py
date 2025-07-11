import logging
import json
from typing import List, Dict, Any, Optional
from pymilvus import (
    connections, Collection, CollectionSchema, FieldSchema, DataType, utility
)
from langchain_core.documents import Document
from dotenv import load_dotenv
import os
import uuid
from datetime import datetime, timezone

# Configure logging with structured format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class MilvusManager:
    """Manages Milvus collections for storing summarized content, dense/sparse vectors, and metadata."""

    def __init__(
        self,
        host: str = None,
        port: str = None,
        collection_name: str = None,
        vector_dim: int = 1536,  # Default for text-embedding-ada-002
        max_content_size: int = 65535,  # Milvus VARCHAR max length for content
        max_metadata_size: int = 65535  # Milvus VARCHAR max length for metadata
    ):
        """
        Initialize MilvusManager with connection and schema parameters.

        Args:
            host (str): Milvus server host (default: from MILVUS_HOST env).
            port (str): Milvus server port (default: from MILVUS_PORT env).
            collection_name (str): Name of the Milvus collection (default: from MILVUS_COLLECTION env).
            vector_dim (int): Dimension of dense vectors (default: 1536 for Azure OpenAI ada-002).
            max_content_size (int): Max length for content field.
            max_metadata_size (int): Max length for metadata JSON string.
        """
        self.host = host or os.getenv("MILVUS_HOST", "127.0.0.1")
        self.port = port or os.getenv("MILVUS_PORT", "19530")
        self.collection_name = collection_name or os.getenv("MILVUS_COLLECTION", "multi_agent_rag")
        self.vector_dim = vector_dim
        self.max_content_size = max_content_size
        self.max_metadata_size = max_metadata_size
        self.collection = None

        # Validate configurations
        if not all([self.host, self.port, self.collection_name]):
            logger.error("Missing required configurations: host=%s, port=%s, collection_name=%s",
                         self.host, self.port, self.collection_name)
            raise ValueError("Host, port, and collection name are required")

        # Connect to Milvus
        try:
            connections.connect(host=self.host, port=self.port)
            logger.info("Connected to Milvus server at %s:%s", self.host, self.port)
        except Exception as e:
            logger.error("Failed to connect to Milvus server: %s", str(e))
            raise ValueError(f"Milvus connection failed: {str(e)}")

        # Initialize or load collection
        self._initialize_collection()

    def _initialize_collection(self) -> None:
        """Create or load the Milvus collection with the defined schema."""
        try:
            if utility.has_collection(self.collection_name):
                self.collection = Collection(self.collection_name)
                logger.info("Loaded existing Milvus collection: %s", self.collection_name)
                return

            # Define schema
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=36),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=self.max_content_size),
                FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=self.vector_dim),
                FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
                FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=self.max_metadata_size),
                FieldSchema(name="created_at", dtype=DataType.VARCHAR, max_length=32)
            ]
            schema = CollectionSchema(fields=fields, description="Multi-agent RAG collection for summarized content", enable_dynamic_field=False)

            # Create collection
            self.collection = Collection(name=self.collection_name, schema=schema)
            logger.info("Created Milvus collection: %s with vector dimension %d", self.collection_name, self.vector_dim)

            # Create index for dense vector
            dense_index_params = {
                "metric_type": "COSINE",
                "index_type": "HNSW",
                "params": {"M": 16, "efConstruction": 200}
            }
            self.collection.create_index("dense_vector", dense_index_params)
            logger.info("Created HNSW index for dense_vector field")

            # Create index for sparse vector
            sparse_index_params = {
                "metric_type": "IP",
                "index_type": "SPARSE_INVERTED_INDEX",
                "params": {"drop_ratio_build": 0.2}
            }
            self.collection.create_index("sparse_vector", sparse_index_params)
            logger.info("Created SPARSE_INVERTED_INDEX for sparse_vector field")

            # Load collection
            self.collection.load()
            logger.info("Loaded Milvus collection: %s", self.collection_name)

        except Exception as e:
            logger.error("Failed to initialize collection %s: %s", self.collection_name, str(e))
            raise RuntimeError(f"Collection initialization failed: {str(e)}")

    def _sanitize_content(self, content: str) -> str:
        """Truncate content to fit within max_content_size."""
        if len(content) > self.max_content_size:
            logger.warning("Content size %d exceeds max %d, truncating", len(content), self.max_content_size)
            return content[:self.max_content_size]
        return content

    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> str:
        """Sanitize and serialize metadata to JSON string, ensuring size limit."""
        try:
            # Remove embeddings to avoid redundancy
            metadata = {k: v for k, v in metadata.items() if k not in ["embedding", "sparse_embedding"]}
            # Serialize to JSON
            metadata_str = json.dumps(metadata, ensure_ascii=False)
            if len(metadata_str) > self.max_metadata_size:
                logger.warning("Metadata size %d exceeds max %d, truncating", len(metadata_str), self.max_metadata_size)
                # Truncate by removing non-essential fields
                essential_keys = ["doc_id", "content_type", "source", "original_source", "page_number"]
                truncated_metadata = {k: metadata.get(k) for k in essential_keys if k in metadata}
                metadata_str = json.dumps(truncated_metadata, ensure_ascii=False)
                if len(metadata_str) > self.max_metadata_size:
                    logger.error("Truncated metadata still exceeds max size: %d", len(metadata_str))
                    raise ValueError("Metadata size exceeds maximum limit even after truncation")
            return metadata_str
        except Exception as e:
            logger.error("Failed to serialize metadata: %s", str(e))
            raise ValueError(f"Metadata serialization failed: {str(e)}")

    def _convert_sparse_embedding(self, sparse_dict: Dict[Any, Any]) -> Dict[int, float]:
        """Convert sparse embedding dictionary to Milvus-compatible format."""
        try:
            if not sparse_dict:
                logger.warning("Empty sparse embedding provided, returning empty sparse vector")
                return {}

            # Handle COO format
            if "indices" in sparse_dict and "values" in sparse_dict:
                result = {}
                for idx, val in zip(sparse_dict["indices"], sparse_dict["values"]):
                    idx = int(idx)
                    if idx < 0:
                        logger.warning(f"Skipping negative sparse index: {idx}")
                        continue
                    val = float(val)
                    if val == 0:
                        logger.warning(f"Skipping zero sparse value for index: {idx}")
                        continue
                    result[idx] = val
                return result

            # Handle direct dictionary format
            result = {}
            for k, v in sparse_dict.items():
                try:
                    idx = int(k)
                    if idx < 0:
                        logger.warning(f"Skipping negative sparse index: {idx}")
                        continue
                    val = float(v)
                    if val == 0:
                        logger.warning(f"Skipping zero sparse value for index: {idx}")
                        continue
                    result[idx] = val
                except (ValueError, TypeError):
                    logger.warning(f"Skipping invalid sparse vector key-value pair: {k}:{v}")
                    continue

            if not result:
                logger.warning("No valid sparse indices/values after validation, returning empty sparse vector")
                return {}

            logger.debug("Converted sparse embedding: %s", result)
            return result
        except Exception as e:
            logger.error("Failed to convert sparse embedding: %s", str(e))
            raise ValueError(f"Sparse embedding conversion failed: {str(e)}")

    def insert_documents(self, documents: List[Document]) -> List[str]:
        """
        Insert summarized documents into Milvus collection.

        Args:
            documents (List[Document]): List of LangChain Document objects with summaries and embeddings.
                                       The page_content should be the summary, and metadata must include doc_id.

        Returns:
            List[str]: List of inserted document IDs.
        """
        if not documents:
            logger.warning("No documents provided for insertion")
            return []

        inserted_ids = []
        entities = []

        for doc in documents:
            doc_id = doc.metadata.get("doc_id")
            if not doc_id:
                doc_id = str(uuid.uuid4())
                logger.warning("No doc_id in metadata, generated: %s", doc_id)
                doc.metadata["doc_id"] = doc_id

            content = self._sanitize_content(doc.page_content)
            dense_embedding = doc.metadata.get("embedding")
            sparse_embedding = doc.metadata.get("sparse_embedding")
            metadata = doc.metadata

            if not content.strip():
                logger.warning("Empty content for document %s, skipping", doc_id)
                continue

            if not dense_embedding or not sparse_embedding:
                logger.warning("Document %s missing dense or sparse embedding, skipping", doc_id)
                continue

            if not isinstance(dense_embedding, list) or not isinstance(sparse_embedding, dict):
                logger.warning("Invalid embedding format for document %s, skipping", doc_id)
                continue

            try:
                sparse_vector = self._convert_sparse_embedding(sparse_embedding)
                if not sparse_vector:
                    logger.warning("Empty sparse vector for document %s, skipping", doc_id)
                    continue
                logger.debug("Sparse vector for doc %s: %s", doc_id, sparse_vector)
                entity = {
                    "id": doc_id,
                    "content": content,
                    "dense_vector": dense_embedding,
                    "sparse_vector": sparse_vector,
                    "metadata": self._sanitize_metadata(metadata),
                    "created_at": datetime.now(timezone.utc).isoformat()
                }
                entities.append(entity)
                inserted_ids.append(doc_id)
            except Exception as e:
                logger.error("Failed to prepare document %s for insertion: %s", doc_id, str(e))
                continue

        if not entities:
            logger.warning("No valid documents to insert")
            return []

        try:
            batch_size = 400
            total_batches = (len(entities) - 1) // batch_size + 1
            for i in range(0, len(entities), batch_size):
                batch = entities[i:i + batch_size]
                logger.debug("Inserting batch %d/%d with %d entities", i // batch_size + 1, total_batches, len(batch))
                self.collection.insert(batch)
                logger.info("Inserted batch %d/%d", i // batch_size + 1, total_batches)
            self.collection.flush()
            logger.info("Inserted %d documents into collection %s", len(entities), self.collection_name)
            return inserted_ids
        except Exception as e:
            logger.error("Failed to insert documents into collection %s: %s", self.collection_name, str(e))
            raise RuntimeError(f"Document insertion failed: {str(e)}")

    def query_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Query a document by its ID.

        Args:
            doc_id (str): Document ID to query.

        Returns:
            Optional[Dict[str, Any]]: Document data or None if not found.
        """
        try:
            expr = f"id == \"{doc_id}\""
            results = self.collection.query(expr=expr, output_fields=["id", "content", "metadata", "created_at"])
            if not results:
                logger.info("No document found with ID: %s", doc_id)
                return None
            result = results[0]
            result["metadata"] = json.loads(result["metadata"])
            logger.debug("Queried document ID %s: %s", doc_id, result["content"][:50])
            return result
        except Exception as e:
            logger.error("Failed to query document %s: %s", doc_id, str(e))
            return None

    def delete_by_id(self, doc_id: str) -> bool:
        """
        Delete a document by its ID.

        Args:
            doc_id (str): Document ID to delete.

        Returns:
            bool: True if deleted, False otherwise.
        """
        try:
            expr = f"id == \"{doc_id}\""
            self.collection.delete(expr)
            self.collection.flush()
            logger.info("Deleted document ID %s from collection %s", doc_id, self.collection_name)
            return True
        except Exception as e:
            logger.error("Failed to delete document %s: %s", doc_id, str(e))
            return False

    def __enter__(self):
        """Support context manager for Milvus connection."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Disconnect from Milvus on exit."""
        try:
            connections.disconnect("default")
            logger.debug("Disconnected from Milvus server")
        except Exception as e:
            logger.warning("Failed to disconnect from Milvus: %s", str(e))

if __name__ == "__main__":
    try:
        # Sample documents for testing (mimicking summarized output)
        sample_docs = [
            Document(
                page_content="Summary: Bash commands manage files and processes efficiently.",
                metadata={
                    "doc_id": str(uuid.uuid4()),
                    "content_type": "text",
                    "source": "multimodal_summarization",
                    "original_source": "doc1",
                    "page_number": 1,
                    "embedding": [0.1] * 1536,  # Mock dense embedding
                    "sparse_embedding": {1: 0.5, 2: 0.3}  # Mock sparse embedding
                }
            ),
            Document(
                page_content="Summary: Table lists commands like systemctl for system control.",
                metadata={
                    "doc_id": str(uuid.uuid4()),
                    "content_type": "table",
                    "source": "multimodal_summarization",
                    "original_source": "doc1",
                    "page_number": 3,
                    "embedding": [0.2] * 1536,
                    "sparse_embedding": {3: 0.4, 4: 0.2}
                }
            )
        ]

        with MilvusManager() as manager:
            # Insert documents
            inserted_ids = manager.insert_documents(sample_docs)
            logger.info("Inserted document IDs: %s", inserted_ids)

            # Query a document
            if inserted_ids:
                doc = manager.query_by_id(inserted_ids[0])
                if doc:
                    logger.info("Queried document: %s", doc)

            # Delete a document
            if inserted_ids:
                success = manager.delete_by_id(inserted_ids[0])
                logger.info("Deletion of ID %s: %s", inserted_ids[0], "successful" if success else "failed")

    except Exception as e:
        logger.error("MilvusManager test failed: %s", str(e))
        raise