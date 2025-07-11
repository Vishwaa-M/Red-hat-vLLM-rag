import chainlit as cl
import asyncio
import logging
import json
import hashlib
from typing import List, Dict, Any
from langchain_core.documents import Document
from core.pipelines.generation_pipeline import GenerationPipeline
from dotenv import load_dotenv

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers.clear()
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)  # Set to DEBUG for detailed chunk logging
    return logger

logger = get_logger(__name__)

def normalize_query(query: str) -> str:
    """Normalize query for consistency."""
    query = query.strip()
    if "map" in query.lower() and "bash" not in query.lower():
        query += " in Bash"
    return query

def make_hashable(obj: Any) -> Any:
    """Convert objects to hashable types recursively."""
    if isinstance(obj, list):
        return tuple(make_hashable(item) for item in obj)
    elif isinstance(obj, dict):
        return tuple((k, make_hashable(v)) for k, v in sorted(obj.items()))
    elif isinstance(obj, set):
        return frozenset(make_hashable(item) for item in obj)
    return obj

@cl.on_chat_start
async def start():
    """Initialize session state on chat start."""
    load_dotenv()
    try:
        generation_pipeline = GenerationPipeline()
        cl.user_session.set("generation_pipeline", generation_pipeline)
        cl.user_session.set("history", [])
        cl.user_session.set("documents", [])
        cl.user_session.set("query_lock", asyncio.Lock())
        await cl.Message(content="Welcome to the RAG Assistant! Ask about programming or system administration.").send()
        logger.info("Initialized new chat session")
    except Exception as e:
        logger.error(f"Failed to initialize session: {e}")
        await cl.Message(content="Error initializing session. Please try again.").send()
        raise

@cl.on_message
async def main(message: cl.Message):
    """Handle incoming user queries with streaming response."""
    query = message.content.strip()
    if not query:
        await cl.Message(content="Please enter a query.").send()
        logger.warning("Empty query received")
        return

    try:
        # Normalize query
        query = normalize_query(query)
        query_hash = hashlib.sha256(query.encode()).hexdigest()
        logger.info(f"Processing query: {query[:50]}... (hash: {query_hash[:10]}...)")

        # Get session state
        generation_pipeline = cl.user_session.get("generation_pipeline")
        if not generation_pipeline:
            logger.error("GenerationPipeline not found in session")
            await cl.Message(content="Error: Session not initialized.").send()
            return

        # Prevent concurrent pipeline calls
        query_lock = cl.user_session.get("query_lock")
        async with query_lock:
            history = cl.user_session.get("history")
            session_context = "\n".join([f"Q: {q}\nA: {a[:100]}..." for q, a in history[-3:]])
            if session_context:
                logger.info(f"Session context: {session_context[:100]}...")

            # Stream response and handle documents
            msg = cl.Message(content="")
            response_text = ""
            seen_chunks = set()
            documents = []
            async for item in generation_pipeline.run_stream(query):
                if isinstance(item, str):
                    if item in seen_chunks:
                        logger.debug(f"Skipped duplicate chunk: {item[:50]}...")
                        continue
                    if item.startswith("An error occurred:") or "insufficient to fully answer" in item:
                        logger.warning(f"Error or fallback response received: {item[:100]}...")
                        response_text = item
                        await msg.stream_token(item)
                        break
                    response_text += item
                    await msg.stream_token(item)
                    seen_chunks.add(item)
                    logger.debug(f"Streamed chunk: {item[:50]}...")
                elif isinstance(item, list):
                    documents = item
                    logger.info(f"Received {len(documents)} documents from stream")
                else:
                    logger.warning(f"Unexpected item type in stream: {type(item)}")

            if not response_text.strip():
                logger.warning(f"No response generated for query: {query[:50]}...")
                await cl.Message(content="Sorry, I couldn't generate a response. Please try again.").send()
                return

            # Log final response
            logger.info(f"Final streamed response length: {len(response_text)} characters")

            # Validate and sanitize documents
            sanitized_documents = []
            for doc in documents:
                try:
                    metadata = {k: make_hashable(v) for k, v in doc.metadata.items()}
                    sanitized_doc = Document(
                        page_content=doc.page_content,
                        metadata=metadata
                    )
                    sanitized_documents.append(sanitized_doc)
                except Exception as e:
                    logger.error(f"Error sanitizing document {doc.metadata.get('doc_id', 'unknown')}: {e}")
            documents = sanitized_documents

            # Log retrieved documents
            try:
                for doc in documents[:5]:
                    logger.info(
                        f"Retrieved Doc ID: {doc.metadata.get('doc_id', 'unknown')}, "
                        f"Score: {doc.metadata.get('normalized_score', 0.0):.2f}, "
                        f"Content: {doc.page_content[:200]}..."
                    )
            except Exception as e:
                logger.error(f"Error logging documents: {e}")

            # Store sanitized documents
            cl.user_session.set("documents", documents)

            # Create JSON sources
            sources = [
                {
                    "doc_id": doc.metadata.get("doc_id", "unknown"),
                    "score": float(doc.metadata.get("normalized_score", 0.0)),
                    "content_snippet": doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                }
                for doc in documents[:5]
            ]
            sources_json = json.dumps(sources, indent=2)
            sources_element = cl.Text(name="Sources", content=sources_json, language="json", display="side")

            # Update message with sources
            msg.elements = [sources_element]
            await msg.update()
            logger.debug(f"Sources JSON: {sources_json[:500]}...")

            # Update history with hashable tuple
            history.append((query, response_text))
            cl.user_session.set("history", history)
            logger.info(f"Completed query processing: {query[:50]}...")

    except Exception as e:
        logger.error(f"Error processing query '{query[:50]}...': {e}")
        await cl.Message(content=f"An error occurred: {str(e)}").send()

@cl.on_chat_end
async def end():
    """Clean up on chat end."""
    try:
        generation_pipeline = cl.user_session.get("generation_pipeline")
        if generation_pipeline:
            await generation_pipeline.close()
            logger.info("Closed GenerationPipeline resources")
        for key in ["generation_pipeline", "history", "documents", "query_lock"]:
            cl.user_session.set(key, None)
        logger.info("Cleared session state")
    except Exception as e:
        logger.error(f"Error closing resources: {e}")

if __name__ == "__main__":
    print("This script should be run using 'chainlit run core/orchestration_unit/app.py -w'.")
    print("Running a minimal test instead...")
    import os
    os.environ["VLLM_BASE_URL"] = "https://av-parade-experiments-spin.trycloudflare.com/v1"
    os.environ["VLLM_API_KEY"] = "ydD5rcbAWF8ZlViqmoMSqjRgV6rM0G1LLgmNNZ_CUJc"
    async def test():
        logger.info("Running test initialization")
        await start()
        logger.info("Test completed")
    asyncio.run(test())