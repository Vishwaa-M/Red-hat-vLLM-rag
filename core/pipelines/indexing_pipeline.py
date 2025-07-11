import os
import asyncio
import logging
import uuid
import json
import traceback
from typing import Dict, List, Any, TypedDict
from retrying import retry
from langgraph.graph import StateGraph, START, END
from langchain_core.documents import Document
from core.indexing_service.data_loaders.minio_document_processor import DocumentProcessor
from core.indexing_service.semantic_chunker import semantic_chunking
from core.indexing_service.multimodal_summarization import MultimodalSummarizer
from core.indexing_service.dense_embedding_generator import DenseEmbeddingGenerator
from core.indexing_service.sparse_embedding_generator import SparseEmbeddingGenerator
from core.indexing_service.milvus_manager import MilvusManager
from dotenv import load_dotenv
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def sanitize_minio_endpoint(endpoint: str, for_boto3: bool = False) -> str:
    """Sanitize MinIO endpoint, returning host:port for Minio client or full URL for boto3."""
    try:
        if not endpoint.startswith(('http://', 'https://')):
            endpoint = f"http://{endpoint}"
        parsed = urlparse(endpoint, scheme='http')
        netloc = parsed.netloc or parsed.path.split('/')[0]
        if not netloc:
            raise ValueError(f"Invalid endpoint: {endpoint}")
        if ':' not in netloc or not all(netloc.split(':')):
            raise ValueError(f"Invalid host:port format: {netloc}")
        if parsed.path and parsed.path != '/':
            logger.warning(f"Removed path component from endpoint: {parsed.path}")
        result = endpoint if for_boto3 else netloc
        logger.debug("Sanitized MinIO endpoint: %s -> %s (for_boto3=%s)", endpoint, result, for_boto3)
        return result
    except Exception as e:
        logger.error("Failed to sanitize MinIO endpoint %s: %s, returning default", endpoint, str(e))
        default = "http://192.168.190.186:9000" if for_boto3 else "192.168.190.186:9000"
        return default

# Define the state for the LangGraph pipeline
class GraphState(TypedDict):
    text_documents: List[Document]
    table_documents: List[Document]
    chunked_documents: List[Document]
    summarized_documents: List[Dict[str, Any]]
    embedded_documents: List[Document]
    error_log: List[Dict[str, Any]]
    step_count: int

@retry(stop_max_attempt_number=3, wait_fixed=2000)
async def load_documents(minio_endpoint: str, access_key: str, secret_key: str, bucket_name: str, prefix: str, secure: bool, output_dir: str) -> tuple[List[Document], List[Document]]:
    """Load and parse documents from MinIO, returning text and table documents."""
    logger.info("Loading documents from MinIO: bucket=%s, prefix=%s", bucket_name, prefix)
    try:
        processor = DocumentProcessor(
            minio_endpoint=minio_endpoint,
            minio_access_key=access_key,
            minio_secret_key=secret_key,
            bucket_name=bucket_name,
            prefix=prefix,
            secure=secure,
            output_dir=output_dir,
            store_images_on_disk=False
        )
        # Load raw files
        raw_files = await processor._load_raw_files()
        logger.info("Loaded %d raw files for sequential processing", len(raw_files))
        
        text_docs = []
        table_docs = []
        image_docs = []
        
        # Process each file sequentially to avoid Docling threading issues
        for content, source in raw_files:
            logger.debug("Processing file: %s", source)
            text_elements, table_elements, image_elements = await processor._parse_raw_file(content, source)
            
            # Convert elements to Document objects
            for content, source, metadata in text_elements:
                metadata["content_type"] = "text"
                text_docs.append(Document(page_content=content, metadata=metadata))
            for content, source, metadata in table_elements:
                metadata["content_type"] = "table"
                table_docs.append(Document(page_content=content, metadata=metadata))
            for content, source, metadata in image_elements:
                metadata["content_type"] = "image"
                image_docs.append(Document(page_content=content, metadata=metadata))
        
        processor.cleanup()
        # Assign unique doc_id and validate
        doc_ids = set()
        sources = set()
        for doc in text_docs + table_docs:
            sources.add(doc.metadata.get("source", "unknown"))
            if "doc_id" not in doc.metadata or doc.metadata["doc_id"] in doc_ids:
                doc.metadata["doc_id"] = str(uuid.uuid4())
                doc_ids.add(doc.metadata["doc_id"])
            else:
                doc_ids.add(doc.metadata["doc_id"])
        logger.info("Loaded %d unique source documents: %s", len(sources), sources)
        if len(doc_ids) != len(text_docs) + len(table_docs):
            logger.warning("Duplicate doc_ids detected: %d unique IDs for %d documents", len(doc_ids), len(text_docs) + len(table_docs))
        
        logger.info(f"Loaded {len(text_docs)} text documents, {len(table_docs)} table documents, {len(image_docs)} image documents (images ignored)")
        return text_docs, table_docs
    except Exception as e:
        logger.error("Failed to load documents: %s\n%s", str(e), traceback.format_exc())
        raise

def load_documents_node(state: GraphState) -> Dict[str, Any]:
    """Load and parse documents from MinIO."""
    logger.info("Starting document loading node")
    try:
        minio_endpoint = os.getenv("MINIO_ENDPOINT", "http://192.168.190.186:9000")
        sanitized_endpoint = sanitize_minio_endpoint(minio_endpoint, for_boto3=True)
        logger.info(f"Using MinIO endpoint for boto3: {sanitized_endpoint}")
        
        access_key = os.getenv("MINIO_ACCESS_KEY")
        secret_key = os.getenv("MINIO_SECRET_KEY")
        bucket_name = os.getenv("MINIO_BUCKET_NAME", "multi-agent-rag-data")
        prefix = os.getenv("MINIO_PREFIX", "document/")
        secure = os.getenv("MINIO_SECURE", "false").lower() == "true"
        output_dir = "scratch"
        
        text_docs, table_docs = asyncio.run(load_documents(sanitized_endpoint, access_key, secret_key, bucket_name, prefix, secure, output_dir))
        
        logger.info("Completed document loading node")
        return {
            "text_documents": text_docs,
            "table_documents": table_docs,
            "chunked_documents": [],
            "summarized_documents": [],
            "embedded_documents": [],
            "error_log": state.get("error_log", []),
            "next_node": "chunk_documents",
            "step_count": state.get("step_count", 0) + 1
        }
    except Exception as e:
        logger.error("Document loading failed: %s\n%s", str(e), traceback.format_exc())
        return {
            "error_log": state.get("error_log", []) + [{"stage": "load_documents", "error": str(e), "traceback": traceback.format_exc()}],
            "next_node": "collect_results",
            "step_count": state.get("step_count", 0) + 1
        }

def chunk_single_batch(text_docs: List[Document], table_docs: List[Document]) -> List[Document]:
    """Chunk text and table documents."""
    logger.info("Chunking batch: %d text, %d table documents", len(text_docs), len(table_docs))
    try:
        chunks = semantic_chunking(
            text_documents=text_docs,
            table_documents=table_docs,
            initial_chunk_size=3,
            min_chunk_size=180,
            max_chunk_size=500,
            overlap_sentences=1,
            batch_size=16
        )
        # Assign unique doc_id and preserve parent_doc_id
        input_doc_ids = {doc.metadata.get("doc_id", "unknown"): doc.metadata.get("source", "unknown") for doc in text_docs + table_docs}
        parent_id_counts = {}
        for chunk in chunks:
            parent_doc_id = chunk.metadata.get("parent_doc_id", chunk.metadata.get("doc_id", "unknown"))
            chunk.metadata["doc_id"] = str(uuid.uuid4())
            chunk.metadata["parent_doc_id"] = parent_doc_id
            parent_id_counts[parent_doc_id] = parent_id_counts.get(parent_doc_id, 0) + 1
            if parent_doc_id not in input_doc_ids:
                logger.warning("Chunk parent_doc_id %s not found in input documents, source: %s", parent_doc_id, chunk.metadata.get("source", "unknown"))
        logger.info("Parent ID distribution: %s", {k: v for k, v in parent_id_counts.items()})
        logger.info("Generated %d chunks", len(chunks))
        return chunks
    except Exception as e:
        logger.error("Chunking failed: %s\n%s", str(e), traceback.format_exc())
        error_docs = [
            Document(page_content=d.page_content, metadata={**d.metadata, "error": str(e)})
            for d in text_docs + table_docs
        ]
        return error_docs

def chunk_documents_node(state: GraphState) -> Dict[str, Any]:
    """Chunk text and table documents with additional validation logging."""
    step_count = state.get("step_count", 0) + 1
    logger.debug("Step %d: Starting chunking", step_count)
    
    text_docs = state.get("text_documents", [])
    table_docs = state.get("table_documents", [])
    
    if not (text_docs or table_docs):
        logger.info("No documents to chunk, moving to summarize")
        return {
            "chunked_documents": [],
            "next_node": "summarize",
            "step_count": step_count
        }
    
    logger.info(f"Chunking {len(text_docs)} text documents and {len(table_docs)} table documents")
    try:
        chunks = chunk_single_batch(text_docs, table_docs)
        
        # Validation logging
        parent_ids = {chunk.metadata.get("parent_doc_id", "unknown") for chunk in chunks}
        unique_sources = {doc.metadata.get("source", "unknown") for doc in text_docs + table_docs}
        logger.info("Generated %d chunks from %d unique parent documents (from %d source documents)", len(chunks), len(parent_ids), len(unique_sources))
        if len(parent_ids) != len(unique_sources):
            logger.warning("Parent IDs (%d) do not match unique source documents (%d), possible metadata mismatch",
                           len(parent_ids), len(unique_sources))
        
        # Log chunk size distribution
        text_chunk_sizes = [len(chunk.page_content.split()) for chunk in chunks if chunk.metadata.get("content_type") == "text"]
        if text_chunk_sizes:
            logger.info("Text chunk sizes: min=%d, max=%d, avg=%.1f", min(text_chunk_sizes), max(text_chunk_sizes), sum(text_chunk_sizes)/len(text_chunk_sizes))
        
        # Log sample chunk sizes and content
        sample_chunks = chunks[:3]  # Log first 3 chunks
        for i, chunk in enumerate(sample_chunks):
            size = len(chunk.page_content.split())  # Word count
            preview = chunk.page_content[:100] + "..." if len(chunk.page_content) > 100 else chunk.page_content
            logger.debug("Sample chunk %d: size=%d words, content='%s', doc_id=%s, parent_doc_id=%s",
                         i+1, size, preview, chunk.metadata.get("doc_id"), chunk.metadata.get("parent_doc_id"))
        
        logger.info("Completed chunking node")
        return {
            "chunked_documents": chunks,
            "next_node": "summarize",
            "step_count": step_count
        }
    except Exception as e:
        logger.error("Chunking failed: %s\n%s", str(e), traceback.format_exc())
        return {
            "error_log": state.get("error_log", []) + [{"stage": "chunk_documents", "error": str(e), "traceback": traceback.format_exc()}],
            "next_node": "summarize",
            "step_count": step_count
        }

def summarize_single_batch(documents: List[Document], summarizer: MultimodalSummarizer) -> List[Dict[str, Any]]:
    """Summarize a batch of documents with fallback for errors."""
    logger.info("Summarizing batch of %d documents", len(documents))
    try:
        summaries = summarizer.process_documents(documents)
        return summaries
    except Exception as e:
        logger.error("Summarization failed for batch: %s\n%s", str(e), traceback.format_exc())
        return [
            {
                "doc_id": doc.metadata.get("doc_id", "unknown"),
                "summary": doc.page_content[:200] + "...",  # Truncated content as fallback
                "content_type": doc.metadata.get("content_type", "unknown"),
                "source": doc.metadata.get("source", "unknown"),
                "page_number": doc.metadata.get("page_number", 1),
                "error": str(e)
            }
            for doc in documents
        ]

def summarize_node(state: GraphState) -> Dict[str, Any]:
    """Summarize chunked documents with reduced concurrency."""
    step_count = state.get("step_count", 0) + 1
    logger.debug("Step %d: Summarizing documents", step_count)
    
    documents = state.get("chunked_documents", [])
    if not documents:
        logger.info("No documents to summarize, moving to embed documents")
        return {
            "summarized_documents": [],
            "next_node": "embed_documents",
            "step_count": step_count
        }
    
    logger.info(f"Summarizing {len(documents)} documents")
    try:
        minio_endpoint = os.getenv("MINIO_ENDPOINT", "http://192.168.190.186:9000")
        sanitized_endpoint = sanitize_minio_endpoint(minio_endpoint, for_boto3=False)
        summarizer = MultimodalSummarizer(
            minio_endpoint=sanitized_endpoint,
            minio_access_key=os.getenv("MINIO_ACCESS_KEY"),
            minio_secret_key=os.getenv("MINIO_SECRET_KEY"),
            bucket_name=os.getenv("MINIO_BUCKET_NAME", "multi-agent-rag-data"),
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Increase batch size and reduce workers to avoid API overload
        batch_size = max(1, len(documents) // 2)  # Process in 2 batches
        batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]
        
        summarized_docs = []
        with ThreadPoolExecutor(max_workers=1) as executor:  # Sequential processing
            futures = [executor.submit(summarize_single_batch, batch, summarizer) for batch in batches]
            for future in as_completed(futures):
                summarized_docs.extend(future.result())
        
        logger.info("Completed summarization node: %d summaries generated", len(summarized_docs))
        return {
            "summarized_documents": summarized_docs,
            "next_node": "embed_documents",
            "step_count": step_count
        }
    except Exception as e:
        logger.error("Summarization failed: %s\n%s", str(e), traceback.format_exc())
        return {
            "error_log": state.get("error_log", []) + [{"stage": "summarize", "error": str(e), "traceback": traceback.format_exc()}],
            "next_node": "embed_documents",
            "step_count": step_count
        }

def embed_single_batch(documents: List[Document], dense_embedder: DenseEmbeddingGenerator, sparse_embedder: SparseEmbeddingGenerator) -> List[Document]:
    """Generate dense and sparse embeddings for a batch of documents."""
    logger.info("Embedding batch of %d documents", len(documents))
    try:
        dense_embedded = dense_embedder.embed_documents(documents)
        sparse_embedded = sparse_embedder.embed_documents(dense_embedded)
        return sparse_embedded
    except Exception as e:
        logger.error("Embedding failed for batch: %s\n%s", str(e), traceback.format_exc())
        return [Document(page_content=doc.page_content, metadata={**doc.metadata, "error": str(e)}) for doc in documents]

def embed_documents_node(state: GraphState) -> Dict[str, Any]:
    """Generate dense and sparse embeddings for summarized documents."""
    step_count = state.get("step_count", 0) + 1
    logger.debug("Step %d: Generating embeddings", step_count)
    
    summarized_docs = state.get("summarized_documents", [])
    if not summarized_docs:
        logger.info("No summaries to embed, moving to store in Milvus")
        return {
            "embedded_documents": [],
            "next_node": "store_in_milvus",
            "step_count": step_count
        }
    
    logger.info(f"Embedding {len(summarized_docs)} summarized documents")
    try:
        documents = [
            Document(
                page_content=doc["summary"],
                metadata={
                    "doc_id": doc["doc_id"],
                    "parent_doc_id": doc.get("parent_doc_id", doc["doc_id"]),
                    "content_type": doc.get("content_type", "summary"),
                    "source": doc.get("source", "summary"),
                    "original_source": doc.get("original_source", ""),
                    "page_number": doc.get("page_number", 1),
                    "summary_metadata": doc
                }
            )
            for doc in summarized_docs
        ]
        
        batch_size = max(1, len(documents) // 4)
        batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]
        
        embedded_docs = []
        with DenseEmbeddingGenerator() as dense_embedder, SparseEmbeddingGenerator() as sparse_embedder:
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(embed_single_batch, batch, dense_embedder, sparse_embedder) for batch in batches]
                for future in as_completed(futures):
                    embedded_docs.extend(future.result())
        
        logger.info("Completed embedding node: %d documents embedded", len(embedded_docs))
        return {
            "embedded_documents": embedded_docs,
            "next_node": "store_in_milvus",
            "step_count": step_count
        }
    except Exception as e:
        logger.error("Embedding failed: %s\n%s", str(e), traceback.format_exc())
        return {
            "error_log": state.get("error_log", []) + [{"stage": "embed_documents", "error": str(e), "traceback": traceback.format_exc()}],
            "next_node": "store_in_milvus",
            "step_count": step_count
        }

def store_in_milvus_node(state: GraphState) -> Dict[str, Any]:
    """Store embedded documents in Milvus."""
    step_count = state.get("step_count", 0) + 1
    logger.debug("Step %d: Storing in Milvus", step_count)
    
    embedded_docs = state.get("embedded_documents", [])
    if not embedded_docs:
        logger.info("No documents to store in Milvus, moving to collect results")
        return {
            "next_node": "collect_results",
            "step_count": step_count
        }
    
    logger.info(f"Storing {len(embedded_docs)} documents in Milvus")
    try:
        with MilvusManager() as milvus_manager:
            inserted_ids = milvus_manager.insert_documents(embedded_docs)
        logger.info("Completed Milvus storage: %d documents stored", len(inserted_ids))
        return {
            "next_node": "collect_results",
            "step_count": step_count
        }
    except Exception as e:
        logger.error("Milvus storage failed: %s\n%s", str(e), traceback.format_exc())
        return {
            "error_log": state.get("error_log", []) + [{"stage": "store_in_milvus", "error": str(e), "traceback": traceback.format_exc()}],
            "next_node": "collect_results",
            "step_count": step_count
        }

def collect_results_node(state: GraphState) -> Dict[str, Any]:
    """Collect final results and save summaries."""
    step_count = state.get("step_count", 0) + 1
    logger.info("Step %d: Collecting results: %d chunked, %d summarized, %d embedded, %d errors",
                step_count, len(state.get("chunked_documents", [])),
                len(state.get("summarized_documents", [])),
                len(state.get("embedded_documents", [])),
                len(state.get("error_log", [])))
    
    if state.get("summarized_documents"):
        try:
            with open("summaries_for_extraction.json", "w") as f:
                json.dump(state["summarized_documents"], f, indent=2)
            logger.info("Saved summaries to summaries_for_extraction.json")
        except Exception as e:
            logger.error("Failed to save summaries: %s\n%s", str(e), traceback.format_exc())
            state["error_log"].append({"stage": "collect_results", "error": str(e), "traceback": traceback.format_exc()})
    
    logger.info("Completed collect results node")
    return {
        "chunked_documents": state.get("chunked_documents", []),
        "summaries": state.get("summarized_documents", []),
        "embedded_documents": state.get("embedded_documents", []),
        "errors": state.get("error_log", []),
        "step_count": step_count
    }

def build_graph():
    """Build the LangGraph workflow."""
    logger.info("Building LangGraph workflow")
    graph_builder = StateGraph(GraphState)
    graph_builder.add_node("load_documents", load_documents_node)
    graph_builder.add_node("chunk_documents", chunk_documents_node)
    graph_builder.add_node("summarize", summarize_node)
    graph_builder.add_node("embed_documents", embed_documents_node)
    graph_builder.add_node("store_in_milvus", store_in_milvus_node)
    graph_builder.add_node("collect_results", collect_results_node)
    
    graph_builder.add_edge(START, "load_documents")
    graph_builder.add_edge("load_documents", "chunk_documents")
    graph_builder.add_edge("chunk_documents", "summarize")
    graph_builder.add_edge("summarize", "embed_documents")
    graph_builder.add_edge("embed_documents", "store_in_milvus")
    graph_builder.add_edge("store_in_milvus", "collect_results")
    graph_builder.add_edge("collect_results", END)
    
    return graph_builder.compile()

def run_graph() -> Dict[str, Any]:
    """Run the LangGraph pipeline."""
    logger.info("Starting LangGraph pipeline execution")
    graph = build_graph()
    try:
        initial_state: GraphState = {
            "text_documents": [],
            "table_documents": [],
            "chunked_documents": [],
            "summarized_documents": [],
            "embedded_documents": [],
            "error_log": [],
            "step_count": 0
        }
        result = graph.invoke(initial_state, config={"recursion_limit": 10000})
        chunked_docs = result.get("chunked_documents", [])
        summaries = result.get("summaries", [])
        embedded_docs = result.get("embedded_documents", [])
        errors = result.get("errors", [])
        logger.info("Pipeline completed: %d chunked, %d summaries, %d embedded, %d errors, %d steps taken",
                    len(chunked_docs), len(summaries), len(embedded_docs), len(errors), result.get("step_count", 0))
        return {
            "chunked_documents": chunked_docs,
            "summaries": summaries,
            "embedded_documents": embedded_docs,
            "errors": errors
        }
    except Exception as e:
        logger.error("Pipeline execution failed: %s\n%s", str(e), traceback.format_exc())
        return {}

if __name__ == "__main__":
    result = run_graph()
    logger.info("Pipeline execution result: %s", result)