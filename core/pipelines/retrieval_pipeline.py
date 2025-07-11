import logging
import uuid
from typing import List, Dict, Any, TypedDict, Optional
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from core.retrieval_service.query_transformer import QueryTransformer
from core.retrieval_service.hybrid_search import HybridSearch
# --- CHANGE: Dense and Sparse searchers are no longer called directly by the graph ---

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

# --- CHANGE: Updated state to include retrieval_parameters ---
class RetrievalState(TypedDict):
    query: str
    session_context: Optional[str] # Added for passing context
    query_id: str
    original_query: str
    query_docs: List[Document]
    metadata_filters: Dict[str, str]
    retrieval_parameters: Dict[str, Any] # Crucial new field
    # dense_docs and sparse_docs are no longer needed in the final state
    hybrid_docs: List[Document]

class RetrievalPipeline:
    """
    Orchestrates a state-of-the-art retrieval pipeline using LangGraph.
    This pipeline uses a two-step process: Transform Query, then execute a Hybrid Search.
    """
    
    def __init__(self, use_metadata_filters: bool = True):
        self.transformer = QueryTransformer()
        # --- CHANGE: The HybridSearcher is now the only search component we need to initialize here ---
        # It internally manages the dense, sparse, and reranker components.
        self.hybrid_searcher = HybridSearch(use_metadata_filters=use_metadata_filters)
        self.graph = self._build_graph()
        logger.info("Initialized RetrievalPipeline with a simplified two-step graph.")
    
    def _transform_query(self, state: RetrievalState) -> RetrievalState:
        """
        Transforms the user query and populates the state with all necessary
        components for the search and re-ranking steps.
        """
        query_id = state.get('query_id', 'unknown')
        session_context = state.get('session_context', '')
        try:
            logger.info(f"[ID: {query_id}] Step 1: Transforming query: {state['query'][:70]}...")
            result = self.transformer.transform(state['query'], session_context)
            
            # --- CHANGE: Populate the state with all outputs from the transformer ---
            state['query_docs'] = result['query_docs']
            state['metadata_filters'] = result['metadata_filters']
            state['original_query'] = result['original_query']
            state['retrieval_parameters'] = result['retrieval_parameters']
            
            logger.info(f"[ID: {query_id}] Query transformed. Generated {len(state['query_docs'])} query docs. Recommended params: {state['retrieval_parameters']}")
        except Exception as e:
            logger.error(f"[ID: {query_id}] Query transformation failed: {e}", exc_info=True)
            # Create a fallback state
            state['query_docs'] = [Document(page_content=state['query'])]
            state['metadata_filters'] = {}
            state['original_query'] = state['query']
            state['retrieval_parameters'] = {} # Use defaults in hybrid search
        return state
    
    # --- REMOVED: _dense_search and _sparse_search are no longer needed as graph nodes ---
    
    # --- CHANGE: Rewritten hybrid_search node to be the main search orchestrator ---
    def _hybrid_search_and_rerank(self, state: RetrievalState) -> RetrievalState:
        """
        Executes the entire hybrid search and multi-stage re-ranking pipeline.
        """
        query_id = state.get('query_id', 'unknown')
        try:
            logger.info(f"[ID: {query_id}] Step 2: Executing hybrid search and reranking pipeline...")
            state['hybrid_docs'] = self.hybrid_searcher.search(
                query_docs=state['query_docs'],
                original_query=state['original_query'],
                retrieval_params=state['retrieval_parameters'],
                metadata_filters=state['metadata_filters']
            )
            logger.info(f"[ID: {query_id}] Hybrid search and rerank complete. Retrieved {len(state['hybrid_docs'])} final documents.")
        except Exception as e:
            logger.error(f"[ID: {query_id}] Hybrid search or re-ranking failed: {e}", exc_info=True)
            state['hybrid_docs'] = []
        return state
    
    def _build_graph(self) -> StateGraph:
        """Build the simplified LangGraph workflow."""
        workflow = StateGraph(RetrievalState)
        
        # --- CHANGE: Simplified graph with only two main steps ---
        workflow.add_node("query_transformer", self._transform_query)
        workflow.add_node("hybrid_search_and_rerank", self._hybrid_search_and_rerank)
        
        # Define the new, simpler workflow
        workflow.set_entry_point("query_transformer")
        workflow.add_edge("query_transformer", "hybrid_search_and_rerank")
        workflow.add_edge("hybrid_search_and_rerank", END)
        
        return workflow.compile()
    
    def run(self, query: str, session_context: str = "") -> List[Document]:
        """Run the retrieval pipeline for a given query."""
        query_id = str(uuid.uuid4())[:8]
        try:
            initial_state: RetrievalState = {
                "query": query,
                "session_context": session_context,
                "query_id": query_id,
                "original_query": "",
                "query_docs": [],
                "metadata_filters": {},
                "retrieval_parameters": {},
                "hybrid_docs": []
            }
            logger.info(f"[ID: {query_id}] Pipeline started for query: {query[:70]}...")
            result = self.graph.invoke(initial_state)
            logger.info(f"[ID: {query_id}] Pipeline completed successfully.")
            return result['hybrid_docs']
        except Exception as e:
            logger.error(f"[ID: {query_id}] Pipeline failed for query '{query[:70]}...': {e}", exc_info=True)
            return []

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    # Instantiate the pipeline with filters deactivated for this test run
    pipeline = RetrievalPipeline(use_metadata_filters=False)
    
    query = "explain themap in bash and related commands about it"
    
    results = pipeline.run(query)
    
    print("\n--- Final State-of-the-Art Retrieval Results ---")
    print(f"Retrieved {len(results)} documents.")
    for i, doc in enumerate(results):
        print(f"\n--- Document {i+1} ---")
        print(f"ID: {doc.metadata.get('doc_id')}")
        print(f"Re-rank Score: {doc.metadata.get('rerank_score'):.4f}")
        print(f"Content: {doc.page_content[:250]}...")

    assert len(results) > 0, "Pipeline should return documents for a valid query."
    print("\nPipeline execution test completed.")