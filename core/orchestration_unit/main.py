import logging
import time
import json
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Import your fully built Generation Pipeline
from core.pipelines.generation_pipeline import GenerationPipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)

# --- API Application Setup ---
app = FastAPI(
    title="State-of-the-Art RAG API",
    description="An API for a conversational RAG system with advanced retrieval and generation capabilities, compatible with Open WebUI.",
    version="1.0.0",
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models for Request Body ---
class ChatRequest(BaseModel):
    model: str
    messages: List[Dict[str, str]]
    stream: bool = True

# --- Load Models at Startup ---
try:
    pipeline = GenerationPipeline(use_metadata_filters=False)
    logger.info("RAG Generation Pipeline initialized and models are pre-loaded.")
except Exception as e:
    logger.error(f"FATAL: Failed to initialize RAG Generation Pipeline: {e}", exc_info=True)
    pipeline = None

# --- API Endpoint ---
@app.post("/v1/chat/completions")
async def chat_endpoint(request: ChatRequest):
    """
    Handles chat requests by streaming a series of structured events to the UI,
    providing real-time feedback on the RAG process.
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="RAG Pipeline is not available.")
    
    logger.info(f"Received request for model: {request.model}")
    
    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided in the request.")
    
    query = request.messages[-1]['content']
    conversation_history = request.messages[:-1]

    async def stream_rag_response():
        """The generator function that yields responses for streaming."""
        event_id_counter = 0
        
        try:
            async for event in pipeline.run_stream(query, conversation_history):
                content_to_send = ""
                finish_reason = None 
                
                event_type = event.get('type')

                if event_type == 'status_update':
                    # --- ENHANCEMENT: Handle the new status update event ---
                    # Wrap the status message in a structured JSON object for the UI.
                    status_message = {"status": "in_progress", "message": event.get('data', '')}
                    content_to_send = json.dumps(status_message)

                elif event_type == 'llm_chunk':
                    # The final LLM response (thinking and answer) is streamed as raw text.
                    content_to_send = event.get('data', '')
                
                elif event_type == 'error':
                    # Handle custom error events from the pipeline.
                    content_to_send = json.dumps({"status": "error", "message": event.get('data', 'Unknown error')})
                    finish_reason = "stop"
                
                else:
                    logger.warning(f"Skipping unknown event type: {event_type}")
                    continue

                # Format the event into an OpenAI-compatible Server-Sent Event (SSE) chunk
                chunk = {
                    "id": f"chatcmpl-{event_id_counter}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": content_to_send},
                            "finish_reason": finish_reason,
                        }
                    ],
                }
                event_id_counter += 1
                yield f"data: {json.dumps(chunk)}\n\n"

        except Exception as e:
            logger.error(f"An unhandled error occurred during streaming: {e}", exc_info=True)
            error_message = json.dumps({"status": "error", "message": f"An unexpected server error occurred: {e}"})
            error_chunk = {
                "id": "chatcmpl-error", "object": "chat.completion.chunk", "created": int(time.time()), "model": request.model,
                "choices": [{"index": 0, "delta": {"content": error_message}, "finish_reason": "stop"}]
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
        
        # Send the final DONE message to terminate the stream
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream_rag_response(), media_type="text/event-stream")

# Add a simple root endpoint for health checks
@app.get("/")
def read_root():
    """A simple health check endpoint."""
    if pipeline:
        return {"status": "RAG API is running"}
    else:
        return {"status": "RAG API is NOT RUNNING due to initialization failure."}
