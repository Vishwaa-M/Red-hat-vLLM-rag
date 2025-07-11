import os
import json
import io
import logging
import uuid
from typing import List, Dict, Any
from minio import Minio
from minio.error import S3Error
from openai import OpenAI
from langchain_core.documents import Document
from dotenv import load_dotenv
from urllib.parse import urlparse
import requests
import urllib3
from openai import APIError, APIConnectionError, RateLimitError
from time import time

# Initialize logger
def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)  # Set to DEBUG for detailed logs
    return logger

logger = get_logger(__name__)

def sanitize_minio_endpoint(endpoint: str) -> str:
    """Sanitize MinIO endpoint to return host:port without scheme or path."""
    try:
        if not endpoint.startswith(('http://', 'https://')):
            endpoint = f"http://{endpoint}"
        parsed = urlparse(endpoint, scheme='http')
        netloc = parsed.netloc or parsed.path.split('/')[0]
        if not netloc:
            raise ValueError(f"Invalid endpoint: {endpoint}")
        if parsed.path and parsed.path != '/':
            logger.warning(f"Removed path component from endpoint: {parsed.path}")
        logger.debug("Sanitized MinIO endpoint: %s -> %s", endpoint, netloc)
        return netloc
    except Exception as e:
        logger.error("Failed to sanitize MinIO endpoint %s: %s, returning default", endpoint, str(e))
        return "192.168.190.186:9000"  # Accessible on company network

class MultimodalSummarizer:
    def __init__(self, minio_endpoint: str, minio_access_key: str, minio_secret_key: str, bucket_name: str, openai_api_key: str):
        """Initialize the summarizer with MinIO S3 configuration and OpenAI client."""
        self.bucket_name = bucket_name
        original_endpoint = minio_endpoint
        sanitized_endpoint = sanitize_minio_endpoint(minio_endpoint)
        logger.info(f"Original MinIO endpoint: {original_endpoint}")
        logger.info(f"Sanitized MinIO endpoint: {sanitized_endpoint}")
        try:
            http_client = urllib3.PoolManager(timeout=10.0, retries=urllib3.Retry(total=3))  # 10-second timeout, 3 retries
            self.minio_client = Minio(
                sanitized_endpoint,
                access_key=minio_access_key,
                secret_key=minio_secret_key,
                secure=os.getenv("MINIO_SECURE", "false").lower() == "true",
                http_client=http_client
            )
            logger.info("MultimodalSummarizer initialized with MinIO (endpoint=%s, secure=%s).",
                        sanitized_endpoint, os.getenv("MINIO_SECURE", "false").lower() == "true")
        except Exception as e:
            logger.error(f"Failed to initialize MinIO client with endpoint {sanitized_endpoint}: {e}")
            raise
        try:
            self.openai_client = OpenAI(
                api_key=openai_api_key,
                base_url="http://192.168.190.85:8000/v1",
                max_retries=3  # Retain retries for transient errors
            )
            logger.info("OpenAI client initialized with base_url=http://192.168.190.85:8000/v1")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise

    def ensure_bucket(self) -> None:
        """Ensure the MinIO bucket exists, create it if it doesn't."""
        logger.debug("Ensuring bucket %s exists", self.bucket_name)
        try:
            if not self.minio_client.bucket_exists(self.bucket_name):
                self.minio_client.make_bucket(self.bucket_name)
                logger.info(f"Created bucket: {self.bucket_name}")
            else:
                logger.info(f"Bucket {self.bucket_name} already exists.")
        except S3Error as e:
            logger.error(f"Failed to ensure bucket {self.bucket_name}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error ensuring bucket {self.bucket_name}: {e}")
            raise

    def save_to_minio(self, data: Dict[str, Any], object_name: str) -> None:
        """Save data as JSON to MinIO S3."""
        logger.debug("Saving object %s to MinIO", object_name)
        try:
            json_data = json.dumps(data).encode('utf-8')
            self.minio_client.put_object(
                self.bucket_name,
                object_name,
                data=io.BytesIO(json_data),
                length=len(json_data),
                content_type="application/json"
            )
            logger.info(f"Saved object to MinIO: {object_name}")
        except S3Error as e:
            logger.error(f"Failed to save object {object_name} to MinIO: {e}")
            raise

    def summarize_text(self, text: str) -> str:
        """Generate a high-quality summary for text using granite-3.3-2b-instruct."""
        logger.debug("Summarizing text: %s...", text[:50])
        start_time = time()
        try:
            if not text.strip():
                logger.warning("Empty text provided, no summary generated.")
                raise ValueError("Empty text provided")
            prompt = f"""
            Summarize the following text concisely, capturing the key points and context. 
            The summary should be clear, professional English suitable for embedding in a vector database.
            Avoid external information or opinions.
            
            Text: {text}
            """
            response = self.openai_client.chat.completions.create(
                model="ibm-granite/granite-3.3-2b-instruct",
                messages=[
                    {"role": "system", "content": "You are a high quality summarization assistant that provides precise, accurate summaries."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.3
            )
            summary = response.choices[0].message.content.strip()
            logger.info(f"Generated text summary in {time() - start_time:.2f} seconds: {summary[:50]}...")
            return summary
        except (APIConnectionError, requests.exceptions.Timeout) as e:
            logger.error(f"Network error while summarizing text in {time() - start_time:.2f} seconds: {text[:50]}...: {e}")
            raise RuntimeError(f"Network error during text summarization: {e}")
        except (APIError, RateLimitError) as e:
            logger.error(f"API error while summarizing text in {time() - start_time:.2f} seconds: {text[:50]}...: {e}")
            raise RuntimeError(f"API error during text summarization: {e}")
        except Exception as e:
            logger.error(f"Unexpected error while summarizing text in {time() - start_time:.2f} seconds: {text[:50]}...: {e}")
            raise RuntimeError(f"Unexpected error during text summarization: {e}")

    def summarize_table(self, table: str) -> str:
        """Generate a summary for a table using granite-3.3-2b-instruct."""
        logger.debug("Summarizing table: %s...", table[:50])
        start_time = time()
        try:
            if not table.strip():
                logger.warning("Empty table provided, no summary generated.")
                raise ValueError("Empty table provided")
            prompt = f"""
            Summarize the following table concisely, capturing the key data points or trends.
            The table is in JSON format. The summary should be clear and suitable for vector database embedding.
            Avoid external information or opinions.
            
            Table: 
            {table}
            """
            response = self.openai_client.chat.completions.create(
                model="ibm-granite/granite-3.3-2b-instruct",
                messages=[
                    {"role": "system", "content": "You are a high quality summarization assistant that provides precise, accurate summaries."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.3
            )
            summary = response.choices[0].message.content.strip()
            logger.info(f"Generated table summary in {time() - start_time:.2f} seconds: {summary[:50]}...")
            return summary
        except (APIConnectionError, requests.exceptions.Timeout) as e:
            logger.error(f"Network error while summarizing table in {time() - start_time:.2f} seconds: {table[:50]}...: {e}")
            raise RuntimeError(f"Network error during table summarization: {e}")
        except (APIError, RateLimitError) as e:
            logger.error(f"API error while summarizing table in {time() - start_time:.2f} seconds: {table[:50]}...: {e}")
            raise RuntimeError(f"API error during table summarization: {e}")
        except Exception as e:
            logger.error(f"Unexpected error while summarizing table in {time() - start_time:.2f} seconds: {table[:50]}...: {e}")
            raise RuntimeError(f"Unexpected error during table summarization: {e}")

    def process_documents(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Process each document, generate summaries, and save raw data to MinIO after summarization."""
        logger.debug("Processing %d documents", len(documents))
        results = []
        self.ensure_bucket()

        for doc in documents:
            doc_id = doc.metadata.get("doc_id", None)
            if not doc_id:
                logger.error("No doc_id found in metadata for document from source %s", 
                             doc.metadata.get("source", "unknown"))
                raise ValueError(f"Document from source {doc.metadata.get('source', 'unknown')} missing doc_id")
            content_type = doc.metadata.get("content_type", "text")
            logger.info(f"Processing {content_type} document with doc_id: {doc_id}")

            if content_type not in ["text", "table"]:
                logger.warning(f"Skipping document with unsupported content_type '{content_type}' for doc_id: {doc_id}")
                continue

            try:
                # Generate summary first
                if content_type == "text":
                    summary = self.summarize_text(doc.page_content)
                elif content_type == "table":
                    summary = self.summarize_table(doc.page_content)

                # Save raw document to MinIO only after successful summarization
                raw_data = {
                    "doc_id": doc_id,
                    "content_type": content_type,
                    "page_content": doc.page_content,
                    "metadata": doc.metadata,
                    "source": "multimodal_summarization"
                }
                raw_object_name = f"extracted/{doc_id}_raw.json"
                self.save_to_minio(raw_data, raw_object_name)

                # Prepare summary data
                summary_data = {
                    "doc_id": doc_id,
                    "summary": summary,
                    "content_type": content_type,
                    "source": "multimodal_summarization",
                    "original_source": doc.metadata.get("source", ""),
                    "page_number": doc.metadata.get("page_number", 1)
                }
                results.append(summary_data)

            except (ValueError, RuntimeError) as e:
                logger.error(f"Failed to process document with doc_id {doc_id}: {e}")
                continue

        logger.info(f"Processed {len(results)} documents successfully.")
        return results

def main():
    """Example usage of MultimodalSummarizer."""
    logger.debug("Starting main function")
    load_dotenv()
    minio_endpoint = os.getenv("MINIO_ENDPOINT", "192.168.190.186:9000")
    minio_access_key = os.getenv("MINIO_ACCESS_KEY")
    minio_secret_key = os.getenv("MINIO_SECRET_KEY")
    bucket_name = os.getenv("MINIO_BUCKET_NAME", "multi-agent-rag-data")
    openai_api_key = os.getenv("VLLM_API_KEY")

    if not all([minio_access_key, minio_secret_key, openai_api_key]):
        missing = [k for k, v in {
            "MINIO_ACCESS_KEY": minio_access_key,
            "MINIO_SECRET_KEY": minio_secret_key,
            "VLLM_API_KEY": openai_api_key
        }.items() if not v]
        logger.error(f"Missing environment variables: {', '.join(missing)}")
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

    documents = [
        Document(
            page_content="Linux Bash commands are tools for managing files and processes.",
            metadata={"content_type": "text", "source": "doc1", "page_number": 1, "doc_id": str(uuid.uuid4())}
        ),
        Document(
            page_content='{"data": [["Name", "Age"], ["Alice", 25], ["Bob", 30]]}',
            metadata={"content_type": "table", "source": "doc2", "page_number": 1, "doc_id": str(uuid.uuid4())}
        )
    ]

    logger.debug("Initializing MultimodalSummarizer")
    summarizer = MultimodalSummarizer(
        minio_endpoint=minio_endpoint,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
        bucket_name=bucket_name,
        openai_api_key=openai_api_key
    )
    logger.debug("Processing documents")
    results = summarizer.process_documents(documents)

    logger.debug("Saving results to summaries_for_extraction.json")
    with open("summaries_for_extraction.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved summaries to summaries_for_extraction.json")

if __name__ == "__main__":
    main()