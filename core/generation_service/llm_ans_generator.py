import os
import logging
import asyncio
import re
from typing import List, AsyncGenerator, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from retrying import retry
from dotenv import load_dotenv
import tiktoken

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

class LLMAnswerGenerator:
    """
    Generates intelligent, context-aware answers, rendering a real-time
    process log within its chain-of-thought.
    """
    
    def __init__(self):
        load_dotenv()
        self.base_url = os.getenv("VLLM_BASE_URL", "https://av-parade-experiments-spin.trycloudflare.com/v1")
        self.api_key = os.getenv("VLLM_API_KEY", "dummy_key")
        self.model = os.getenv("VLLM_MODEL", "ibm-granite/granite-3.2-8b-instruct")
        self.temperature = float(os.getenv("VLLM_TEMPERATURE", 0.05))
        # --- FIX: Increased token limit to prevent premature cutoff ---
        self.max_tokens_completion = int(os.getenv("VLLM_MAX_TOKENS", 2048))
        self.top_p = float(os.getenv("VLLM_TOP_P", 0.95))
        self.timeout = int(os.getenv("VLLM_TIMEOUT", 60))
        self.model_context_limit = 4096

        try:
            self.client = ChatOpenAI(
                model=self.model,
                openai_api_key=self.api_key,
                openai_api_base=self.base_url,
                max_tokens=self.max_tokens_completion,
                temperature=self.temperature,
                model_kwargs={"top_p": self.top_p},
                request_timeout=self.timeout,
                streaming=True
            )
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            logger.info(f"Initialized LLMAnswerGenerator with model={self.model}, context_limit={self.model_context_limit}, max_tokens={self.max_tokens_completion}")
        except Exception as e:
            logger.error(f"Failed to initialize ChatOpenAI client: {e}", exc_info=True)
            self.client = None

    def _pack_context_with_budget(self, fixed_prompt_parts: str, documents: List[Document]) -> str:
        """Intelligently packs document content into a prompt, respecting a token budget."""
        fixed_tokens = len(self.tokenizer.encode(fixed_prompt_parts))
        document_token_budget = self.model_context_limit - fixed_tokens - self.max_tokens_completion - 100 
        
        if document_token_budget <= 0:
            logger.warning("No token budget remaining for documents.")
            return ""

        context_parts = []
        current_tokens = 0
        for doc in documents:
            doc_header = f"Source: {doc.metadata.get('source', 'unknown')} (Score: {doc.metadata.get('rerank_score', 0.0):.2f})\n"
            doc_content_str = f"Content: ```{doc.page_content}```"
            full_doc_str = doc_header + doc_content_str
            doc_tokens = len(self.tokenizer.encode(full_doc_str))

            if current_tokens + doc_tokens <= document_token_budget:
                context_parts.append(full_doc_str)
                current_tokens += doc_tokens
            else:
                break
        
        logger.info(f"Packed {len(context_parts)} documents into a context of {current_tokens} tokens.")
        return "\n\n---\n\n".join(context_parts)

    def _build_messages(
        self, 
        query: str, 
        documents: List[Document], 
        conversation_history: List[Dict[str, str]],
        log_messages: List[str]
    ) -> List[SystemMessage | HumanMessage | AIMessage]:
        """Builds the final list of messages, including the live process log."""
        
        # --- ENHANCEMENT: Final prompt instructs the LLM to render the live log ---
        system_prompt = (
            "You are an expert AI assistant. Your response MUST follow this exact format:\n\n"
            "1.  **Thinking Process:** Start with `<thinking>`. Inside, FIRST, you MUST repeat the `Process Log` provided by the system, line by line. AFTER the log, add your own step-by-step reasoning for how you will answer the user's query using the provided context documents. Then close with `</thinking>`.\n"
            "2.  **Final Answer:** After the thinking block, provide the final, user-facing answer in clear markdown.\n"
            "3.  **Sources:** At the very end, add a `**Sources:**` section. List the full `Source` path for each document you used, formatting it as a markdown link where the filename is the link text.\n\n"
            "**Example:**\n"
            "<thinking>\n"
            "**Process Log:**\n"
            "- 🔄 Transforming query...\n"
            "- 🔍 Retrieving and re-ranking documents...\n"
            "- ✅ Found 5 relevant documents.\n\n"
            "**My Reasoning:**\n"
            "The user is asking about X. I will use Source Y and Z to construct the answer.\n"
            "</thinking>\n"
            "This is the markdown answer to the user's question.\n\n"
            "**Sources:**\n"
            "- [filename1.pdf](s3://.../filename1.pdf)\n"
            "- [filename2.txt](s3://.../filename2.txt)\n"
        )
        messages = [SystemMessage(content=system_prompt)]
        
        if conversation_history:
            messages.extend([
                HumanMessage(content=msg['content']) if msg['role'] == 'user' else AIMessage(content=msg['content'])
                for msg in conversation_history
            ])

        # Format the log messages for the prompt
        process_log_str = "**Process Log:**\n- " + "\n- ".join(log_messages) if log_messages else ""

        history_for_budgeting = "\n".join([f"{m['role']}: {m['content']}" for m in conversation_history])
        fixed_prompt_for_budgeting = f"{system_prompt}{history_for_budgeting}{process_log_str}User: {query}"
        context_str = self._pack_context_with_budget(fixed_prompt_for_budgeting, documents)
        
        final_user_content = (
            f"{process_log_str}\n\n"
            "--- Context Documents ---\n"
            f"{context_str}\n\n"
            "--- Current Query ---\n"
            f"User: {query}"
        )
        messages.append(HumanMessage(content=final_user_content))
        return messages

    @retry(
        stop_max_attempt_number=3, 
        wait_exponential_multiplier=1000, 
        wait_exponential_max=10000,
        retry_on_exception=lambda e: isinstance(e, Exception)
    )
    async def generate_stream(
        self, 
        query: str, 
        documents: List[Document], 
        conversation_history: List[Dict[str, str]] = [], 
        log_messages: List[str] = []
    ) -> AsyncGenerator[str, None]:
        """Generates the final response, including the live log in the thinking process."""
        if not self.client:
            yield "The answer generation service is not available."
            return
            
        try:
            messages = self._build_messages(query, documents, conversation_history, log_messages)
            logger.debug(f"Generating streaming response for query: {query}")
            async for chunk in self.client.astream(messages):
                yield chunk.content
        except Exception as e:
            logger.error(f"Failed to generate streaming response: {e}", exc_info=True)
            yield f"An error occurred while streaming the response: {e}"

    async def close(self):
        logger.info("Closing LLMAnswerGenerator resources.")
