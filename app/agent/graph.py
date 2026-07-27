"""
LangGraph state machine wiring the supervisor-worker agent nodes.

Implements the supervisor pattern where:
- Supervisor node makes ALL routing decisions
- Worker nodes (fetch_data, analyst, viz, report) execute their tasks
- No self-loops on supervisor; workers always return to supervisor
"""

import logging
from typing import Literal, TypedDict, Any
from langgraph.graph import StateGraph, END

from app.agent.nodes import (
    AgentState,
    thinker_node,
    data_fetch_node,
    analyst_node,
    viz_node,
    report_node,
    get_agent_max_iterations,
)
from app.agent.model_client import ModelClient
from app.agent.sandbox_client import SandboxClient
from app.agent.data_layer import DataLayer


logger = logging.getLogger(__name__)


class AgentGraph:
    """
    LangGraph-based agent orchestrator with supervisor-worker architecture.

    Manages the state machine where supervisor routes to specialists.
    """

    def __init__(
        self,
        data_dir: str,
        model_path: str | None = None,
        scratch_dir: str | None = None,
    ):
        """
        Initialize the agent graph with all dependencies.

        Args:
            data_dir: Path to directory containing dataset files.
            model_path: Optional path to GGUF model file.
            scratch_dir: Optional path for sandbox scratch space.
        """
        logger.info("Initializing AgentGraph with data_dir=%s", data_dir)

        # Initialize dependencies
        try:
            self.data_layer = DataLayer(data_dir=data_dir)
            logger.info("DataLayer initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize DataLayer: %s", e)
            raise

        try:
            self.model_client = ModelClient(model_path=model_path)
            logger.info("ModelClient initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize ModelClient: %s", e)
            raise

        try:
            self.sandbox = SandboxClient(data_dir=data_dir, scratch_dir=scratch_dir)
            logger.info("SandboxClient initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize SandboxClient: %s", e)
            self.data_layer.close()
            raise

        # Build the graph
        self.graph = self._build_graph()
        logger.info("AgentGraph initialization complete")

    def _build_graph(self) -> StateGraph:
        """Construct the LangGraph state machine."""
        logger.info("Building LangGraph state machine")

        # Define node functions with bound dependencies
        def thinker(state: AgentState) -> AgentState:
            try:
                result = thinker_node(state, self.model_client)
                # Log model output from thinker
                logger.info("Thinker model output: action=%s", result.get('current_action'))
                if 'conversation_history' in result and result['conversation_history']:
                    last_msg = result['conversation_history'][-1]
                    logger.debug("Thinker reasoning: %s", last_msg.get('content', '')[:200])
                return result
            except Exception as e:
                logger.error("thinker_node failed: %s", e)
                return {'errors': state.get('errors', []) + [f'Thinker error: {e}'], 'current_action': 'report'}

        def fetch_data(state: AgentState) -> AgentState:
            try:
                return data_fetch_node(state, self.data_layer)
            except Exception as e:
                logger.error("data_fetch_node failed: %s", e)
                return {'errors': state.get('errors', []) + [f'Data fetch error: {e}']}

        def analyze(state: AgentState) -> AgentState:
            try:
                result = analyst_node(state, self.model_client, self.sandbox, self.data_layer)
                logger.info("Analyst model output: findings=%s", result.get('findings'))
                gc = result.get('generated_code')
                if isinstance(gc, dict):
                    logger.debug("Analyst SQL: %s", gc.get('sql', '')[:200])
                    logger.debug("Analyst Python: %s", gc.get('python', '')[:200])
                elif gc:
                    logger.debug("Analyst generated code: %s", gc[:300])
                return result
            except Exception as e:
                logger.error("analyst_node failed: %s", e)
                return {
                    'errors': state.get('errors', []) + [f'Analyst error: {e}'],
                    'generated_code': None,
                    'findings': 'Analysis failed.',
                    'success': False,   # was missing — stale success flag from prior iteration otherwise persists
                }

        def visualize(state: AgentState) -> AgentState:
            try:
                result = viz_node(state, self.model_client, self.sandbox)
                # Log model output from visualization (code generation)
                if result.get('generated_code'):
                    logger.info("Visualize model output: generated_code (first 300 chars) = %s", result['generated_code'][:300])
                else:
                    logger.info("Visualize model output: no code generated")
                return result
            except Exception as e:
                logger.error("viz_node failed: %s", e)
                return {'errors': state.get('errors', []) + [f'Visualization error: {e}']}

        def report(state: AgentState) -> AgentState:
            try:
                result = report_node(state, self.model_client)
                # Log final model response
                logger.info("Report model output: final_response=%s", result.get('final_response'))
                if result.get('interpretation'):
                    logger.info("Report model output: interpretation=%s", result.get('interpretation'))
                return result
            except Exception as e:
                logger.error("report_node failed: %s", e)
                return {'final_response': f"Error generating response: {e}", 'findings': state.get('findings', 'Unknown'), 'interpretation': 'Error occurred during analysis', 'current_action': 'done'}

        # Create the graph
        workflow = StateGraph(AgentState)
        workflow.add_node("thinker", thinker)
        workflow.add_node("fetch_data", fetch_data)
        workflow.add_node("analyst", analyze)
        workflow.add_node("visualize", visualize)
        workflow.add_node("report", report)

        workflow.set_entry_point("thinker")

        # FIX C: Route function checks iteration count
        def route_from_thinker(state: AgentState) -> Literal["fetch_data", "analyst", "visualize", "report", "__end__"]:
            action = state.get('current_action', 'analyze')
            iteration = state.get('iteration_count', 0)
            
            # Hard stop if max iterations hit
            max_iterations = get_agent_max_iterations()
            if iteration >= max_iterations:
                logger.warning("Max iterations hit in routing, forcing END")
                return 'report'  # Route to report to explain failure
            
            logger.debug("Routing action: %s", action)
            
            if action == 'fetch_data':
                return 'fetch_data'
            elif action in ('extract', 'analyze'):
                return 'analyst'
            elif action == 'visualize':
                return 'visualize'
            elif action in ('report', 'done'):
                return 'report'
            else:
                return 'analyst'

        workflow.add_conditional_edges(source="thinker", path=route_from_thinker)

        workflow.add_edge("fetch_data", "thinker")
        workflow.add_edge("analyst", "thinker")
        workflow.add_edge("visualize", "thinker")
        workflow.add_edge("report", END)

        return workflow.compile()


    def run(self, query: str) -> dict:
        """
        Run a query through the agent loop.

        Args:
            query: Natural language question about the dataset.

        Returns:
            Dict with final_response, findings, interpretation, artifacts.
        """
        logger.info("Running query: %s", query[:100])
        initial_state: AgentState = {
            'query': query,
            'data_profile': None,
            'table_names': [],
            'schema_summary': None,
            'conversation_history': [],
            'current_action': '',
            'generated_code': None,
            'code_language': None,
            'execution_result': None,
            'findings': None,
            'interpretation': None,
            'final_response': None,
            'iteration_count': 0,
            'errors': [],
            'artifacts': [],
        }

        try:
            result = self.graph.invoke(initial_state)
            logger.info("Query completed successfully")
            return result
        except Exception as e:
            logger.error("Graph execution failed: %s", e)
            return {
                'final_response': f"Error during analysis: {str(e)}",
                'findings': None,
                'interpretation': None,
                'artifacts': [],
                'errors': [str(e)]
            }

    def get_artifact(self, artifact_path: str) -> bytes | None:
        """Retrieve a generated artifact (chart PNG) from the sandbox."""
        try:
            return self.sandbox.get_artifact(artifact_path)
        except Exception as e:
            logger.error("Failed to get artifact %s: %s", artifact_path, e)
            return None

    def cleanup(self):
        """Stop the sandbox container and clean up resources."""
        logger.info("Cleaning up agent resources")
        try:
            self.sandbox.stop()
            logger.info("Sandbox stopped successfully")
        except Exception as e:
            logger.warning("Failed to stop sandbox: %s", e)
        try:
            self.data_layer.close()
            logger.info("DataLayer closed successfully")
        except Exception as e:
            logger.warning("Failed to close DataLayer: %s", e)


# Convenience function for Streamlit integration
def create_agent(data_dir: str, model_path: str | None = None) -> AgentGraph:
    """Create an agent instance for a given data directory."""
    return AgentGraph(data_dir=data_dir, model_path=model_path)