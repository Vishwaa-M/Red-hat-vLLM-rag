import logging
import asyncio
import uuid
from typing import List, Dict, Any, AsyncGenerator
import json
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate

# Import the retrieval pipeline to access its internal components
from core.pipelines.retrieval_pipeline import RetrievalPipeline
from core.generation_service.llm_ans_generator import LLMAnswerGenerator


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

class GenerationPipeline:
    """
    Orchestrates a conversational RAG pipeline that generates a dynamic,
    real-time log of its progress within the LLM's thinking process.
    """
    
    def __init__(self, use_metadata_filters: bool = True):
        self.retrieval_pipeline = RetrievalPipeline(use_metadata_filters=use_metadata_filters)
        self.answer_generator = LLMAnswerGenerator()
        
        self.summarizer_prompt = PromptTemplate(
            input_variables=["conversation_text"],
            template=(
                "You are a summarization expert. Create a very short, crisp, one-paragraph summary of the following conversation. "
                "This summary will be used as context for a future query, so focus on the key topics and outcomes."
            )
        )
        logger.info("Initialized Conversational GenerationPipeline with Full History Summarizer.")
    
    async def _summarize_conversation_history(self, conversation_history: List[Dict[str, str]]) -> str:
        """Uses an LLM to summarize the entire conversation history for robust context."""
        if not conversation_history:
            return ""
        conversation_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history])
        prompt = self.summarizer_prompt.format(conversation_text=conversation_text)
        
        logger.debug("Summarizing full conversation history.")
        try:
            summarizer_client = self.answer_generator.client.with_config({"max_tokens": 250})
            response = await summarizer_client.ainvoke(prompt)
            summary = response.content.strip()
            logger.info(f"Generated conversation summary: '{summary[:150]}...'")
            return summary
        except Exception as e:
            logger.error(f"Failed to summarize conversation history: {e}", exc_info=True)
            return ""

    async def run_stream(self, query: str, conversation_history: List[Dict[str, str]] = []) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Runs the full RAG pipeline, collecting status updates and passing them
        to the LLM to be rendered as a live log within its thinking process.
        """
        query_id = str(uuid.uuid4())[:8]
        logger.info(f"[ID: {query_id}] Starting live-log stream for query: {query[:70]}...")
        
        # --- ENHANCEMENT: Create a list to hold real-time log messages ---
        log_messages = []
        
        try:
            summary_context = await self._summarize_conversation_history(conversation_history)
            history_for_generation = []
            if summary_context:
                history_for_generation.append({
                    "role": "system", 
                    "content": f"Summary of the conversation so far: {summary_context}"
                })
            
            # --- Stage 1: Query Transformation ---
            log_messages.append("🔄 Transforming query...")
            transformed_query_data = self.retrieval_pipeline.transformer.transform(query, summary_context)
            logger.info(f"[ID: {query_id}] Query transformed.")

            # --- Stage 2: Document Retrieval & Re-ranking ---
            log_messages.append("🔍 Retrieving and re-ranking documents...")
            documents = self.retrieval_pipeline.hybrid_searcher.search(
                query_docs=transformed_query_data['query_docs'],
                original_query=transformed_query_data['original_query'],
                retrieval_params=transformed_query_data['retrieval_parameters'],
                metadata_filters=transformed_query_data['metadata_filters']
            )
            log_messages.append(f"✅ Found {len(documents)} relevant documents.")
            logger.info(f"[ID: {query_id}] Retrieval and re-ranking complete.")

            if not documents:
                logger.warning(f"[ID: {query_id}] No documents found. Sending fallback.")
                # The 'log_messages' are passed even for the fallback, so the user sees the steps taken.
                fallback_message = "<thinking>\n**Process Log:**\n- " + "\n- ".join(log_messages) + "\n\n**My Reasoning:**\nNo relevant documents were found in the knowledge base to answer the query.</thinking>\nThe provided context is insufficient to fully answer your query."
                yield {"type": "llm_chunk", "data": fallback_message}
                return

            # --- Stage 3: Final Generation ---
            # The 'log_messages' are now passed to the answer generator
            async for chunk in self.answer_generator.generate_stream(query, documents, history_for_generation, log_messages):
                yield {"type": "llm_chunk", "data": chunk}

        except Exception as e:
            logger.error(f"[ID: {query_id}] Streaming pipeline failed: {e}", exc_info=True)
            yield {"type": "error", "data": f"An error occurred: {e}"}

    async def close(self):
        await self.answer_generator.close()
        logger.info("Closed GenerationPipeline resources.")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    async def test_final_pipeline():
        pipeline = GenerationPipeline(use_metadata_filters=False)
        conversation_history = []
        
        # --- Turn 1 ---
        print("\n--- Turn 1: Initial Query ---")
        query1 = "What is the difference between RAG and fine-tuning?"
        print(f"User: {query1}\n")
        
        full_response_1 = ""
        print("Assistant: ")
        async for event in pipeline.run_stream(query1, conversation_history):
            # The test script no longer checks for 'status_update' as it's handled internally by the LLM now.
            if event['type'] == 'llm_chunk':
                full_response_1 += event['data']
                print(event['data'], end="", flush=True)
            elif event['type'] == 'error':
                 print(f"\n[ERROR: {event['data']}]")
        
        conversation_history.append({"role": "user", "content": query1})
        conversation_history.append({"role": "assistant", "content": full_response_1})
        print("\n\n" + "="*50 + "\n")

        # --- Turn 2 ---
        print("--- Turn 2: Follow-up Query ---")
        query2 = "Which one is better for keeping knowledge up to date?"
        print(f"User: {query2}\n")
        
        full_response_2 = ""
        print("Assistant: ")
        async for event in pipeline.run_stream(query2, conversation_history):
            if event['type'] == 'llm_chunk':
                 full_response_2 += event['data']
                 print(event['data'], end="", flush=True)
            elif event['type'] == 'error':
                 print(f"\n[ERROR: {event['data']}]")
        
        print()
        await pipeline.close()
    
    asyncio.run(test_final_pipeline())
