import re
import os
import logging
import json
from typing import List, Dict, Any, Tuple, Optional
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
# --- ENHANCEMENT: Switched to the modern, more reliable Chat model client ---
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

from time import time

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
    return logger

logger = get_logger(__name__)

class QueryTransformer:
    def __init__(self):
        try:
            # --- ENHANCEMENT: Using ChatOpenAI for better instruction following ---
            self.llm_client = ChatOpenAI(
                model="ibm-granite/granite-3.2-8b-instruct",
                openai_api_key=os.getenv("VLLM_API_KEY", "dummy_key"),
                openai_api_base=os.getenv("VLLM_BASE_URL", "https://av-parade-experiments-spin.trycloudflare.com/v1"),
                max_tokens=1024,
                temperature=0.0,
                model_kwargs={"top_p": 0.95}
            )
            logger.info("QueryTransformer initialized with ChatOpenAI client.")
        except Exception as e:
            logger.error(f"Failed to initialize ChatOpenAI for QueryTransformer: {e}")
            self.llm_client = None
        
        # Prompts remain the same, as they are templates. The change is in how we use them.
        self.decompose_prompt_template_text = (
            "Given the user's original query and session context, your task is to break down complex queries into 2–4 simpler, unique, and specific sub-queries. "
            "Each sub-query should be optimized for a vector search system. "
            "If the original query is already simple and specific, return only one sub-query, which must be identical to the original query. "
            "DO NOT include any conversational filler, introductory/concluding remarks, or examples. "
            "Output ONLY a numbered list of sub-queries (e.g., '1. ...'). Each sub-query must be on a new line."
        )
        self.hyde_prompt_template_text = (
            "You are an expert AI assistant. Generate a concise, factual, hypothetical document that directly answers the given query. "
            "Do not include any conversational phrases or disclaimers. Just the hypothetical content itself. "
            "Focus on providing information that would be found in a relevant retrieved document."
        )
        self.parameter_recommendation_prompt_template_text = (
            "You are an expert RAG system optimizer. Based on the user's query, recommend optimal retrieval parameters. "
            "Output ONLY a valid JSON object. For a 'long_tail' query, prioritize high recall (larger top_k). "
            "For a 'factual' query, prioritize precision (smaller top_k)."
            "\n\nStrictly adhere to the following JSON schema:\n"
            "```json\n"
            "{{\n"
            "  \"dense_top_k\": <integer>,\n"
            "  \"sparse_top_k\": <integer>,\n"
            "  \"rerank_top_k\": <integer>,\n"
            "  \"final_top_k\": <integer>,\n"
            "  \"mmr_lambda\": <float between 0.0 and 1.0>\n"
            "}}\n"
            "```"
        )
        self.metadata_filters: Dict[str, str] = {} 
    
    def _invoke_llm(self, messages: List[SystemMessage | HumanMessage]) -> str:
        if self.llm_client is None: raise ValueError("Chat LLM not initialized.")
        try:
            response = self.llm_client.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"Error invoking Chat LLM: {e}", exc_info=True)
            raise

    def decompose_query(self, query: str, session_context: str = "") -> List[str]:
        start_time = time()
        if not query.strip(): return []
        
        try:
            logger.debug(f"Attempting query decomposition for query: '{query[:50]}...'")
            messages = [
                SystemMessage(content=self.decompose_prompt_template_text),
                HumanMessage(content=f"Original query: {query}\nSession context: {session_context or 'None'}")
            ]
            response = self._invoke_llm(messages)
            
            sub_queries = [re.sub(r'^\d+\.\s*', '', line).strip() for line in response.strip().split('\n') if re.match(r'^\d+\.\s*', line.strip())]
            
            if not sub_queries:
                logger.warning("No valid sub-queries extracted. Falling back to original query.")
                return [query]
            
            logger.info(f"Decomposed query in {time() - start_time:.2f}s: '{query}' -> {sub_queries}")
            return sub_queries[:4]
        except Exception as e:
            logger.error(f"Decomposition failed: {e}. Falling back to original query.")
            return [query]
        
    def generate_hyde_document(self, query: str) -> Optional[str]:
        start_time = time()
        if not query.strip():
            # Handle filter-only queries
            return "This document is relevant to the specified filters."

        try:
            logger.debug(f"Attempting HyDE generation for query: '{query[:50]}...'")
            messages = [
                SystemMessage(content=self.hyde_prompt_template_text),
                HumanMessage(content=f"Query: {query}")
            ]
            hyde_document = self._invoke_llm(messages)
            
            if hyde_document:
                logger.info(f"Generated hypothetical document in {time() - start_time:.2f}s.")
                return hyde_document
            return f"General information related to: '{query}'."
        except Exception as e:
            logger.error(f"HyDE generation failed: {e}. Falling back to basic HyDE.")
            return f"General information related to: '{query}'."
        
    def _recommend_retrieval_parameters(self, query: str) -> Dict[str, Any]:
        start_time = time()
        default_params = {"dense_top_k": 50, "sparse_top_k": 50, "rerank_top_k": 50, "final_top_k": 10, "mmr_lambda": 0.7}

        try:
            logger.debug(f"Attempting parameter recommendation for query: '{query[:50]}...'")
            messages = [
                SystemMessage(content=self.parameter_recommendation_prompt_template_text),
                HumanMessage(content=f"Query: {query}")
            ]
            response = self._invoke_llm(messages)
            
            json_match = re.search(r"\{[\s\S]*\}", response)
            if not json_match: raise json.JSONDecodeError("No JSON object found", response, 0)
            
            recommended_params = json.loads(json_match.group(0))
            validated_params = {
                "dense_top_k": int(recommended_params.get("dense_top_k", default_params["dense_top_k"])),
                "sparse_top_k": int(recommended_params.get("sparse_top_k", default_params["sparse_top_k"])),
                "rerank_top_k": int(recommended_params.get("rerank_top_k", default_params["rerank_top_k"])),
                "final_top_k": int(recommended_params.get("final_top_k", default_params["final_top_k"])),
                "mmr_lambda": float(recommended_params.get("mmr_lambda", default_params["mmr_lambda"]))
            }
            logger.info(f"Recommended retrieval parameters in {time() - start_time:.2f}s: {validated_params}")
            return validated_params
        except Exception as e:
            logger.error(f"Parameter recommendation failed: {e}. Returning default parameters.")
            return default_params

    def extract_metadata_filters(self, query: str) -> Tuple[Dict[str, str], str]:
        filters, remaining_query = {}, query
        patterns = {
            "source": r"\b(?:source|from):\s*([^\s]+)\b",
            "date": r"\b(?:date|year):\s*([\w\-]+)\b"
        }
        for key, pattern in patterns.items():
            if match := re.search(pattern, remaining_query, re.IGNORECASE):
                filters[key] = match.group(1)
                remaining_query = re.sub(pattern, "", remaining_query, flags=re.IGNORECASE).strip()
        
        return filters, re.sub(r"\s+", " ", remaining_query).strip()
    
    def transform(self, query: str, session_context: str = "") -> Dict[str, Any]:
        """Transforms the user query into multiple optimized representations."""
        filters, cleaned_query_for_llm = self.extract_metadata_filters(query)
        
        query_docs = []
        if cleaned_query_for_llm:
            sub_queries = self.decompose_query(cleaned_query_for_llm, session_context)
            query_docs.extend([Document(page_content=q) for q in sub_queries])
        
        hyde_doc_content = self.generate_hyde_document(cleaned_query_for_llm or query)
        if hyde_doc_content:
            query_docs.append(Document(page_content=hyde_doc_content, metadata={"is_hyde": True}))
        
        retrieval_parameters = self._recommend_retrieval_parameters(query)
        
        return {
            "query_docs": query_docs,
            "metadata_filters": filters,
            "original_query": query,
            "retrieval_parameters": retrieval_parameters
        }

if __name__ == "__main__":
    load_dotenv()
    if not os.getenv("VLLM_API_KEY"): raise ValueError("Missing VLLM_API_KEY.")
    transformer = QueryTransformer()
    complex_query = "What are the latest security threats to cloud infrastructure and how can we mitigate them using zero-trust principles? source:cloud_report year:2023"
    result = transformer.transform(complex_query, session_context="Previous discussion was on cybersecurity.")
    print(json.dumps(result, indent=2, default=lambda o: o.__dict__))