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
    supervisor_node,
    data_fetch_node,
    analyst_node,
    viz_node,
    report_node,
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
        def supervisor(state: AgentState) -> AgentState:
            try:
                return supervisor_node(state, self.model_client)
            except Exception as e:
                logger.error("supervisor_node failed: %s", e)
                return {'errors': state.get('errors', []) + [f'Supervisor error: {e}'], 'current_action': 'report'}

        def fetch_data(state: AgentState) -> AgentState:
            try:
                return data_fetch_node(state, self.data_layer)
            except Exception as e:
                logger.error("data_fetch_node failed: %s", e)
                return {'errors': state.get('errors', []) + [f'Data fetch error: {e}']}

        def analyze(state: AgentState) -> AgentState:
            try:
                return analyst_node(state, self.model_client, self.sandbox, self.data_layer)
            except Exception as e:
                logger.error("analyst_node failed: %s", e)
                return {'errors': state.get('errors', []) + [f'Analyst error: {e}'], 'generated_code': None, 'findings': 'Analysis failed.'}

        def visualize(state: AgentState) -> AgentState:
            try:
                return viz_node(state, self.model_client, self.sandbox)
            except Exception as e:
                logger.error("viz_node failed: %s", e)
                return {'errors': state.get('errors', []) + [f'Visualization error: {e}']}

        def report(state: AgentState) -> AgentState:
            try:
                return report_node(state, self.model_client)
            except Exception as e:
                logger.error("report_node failed: %s", e)
                return {'final_response': f"Error generating response: {e}", 'findings': state.get('findings', 'Unknown'), 'interpretation': 'Error occurred during analysis', 'current_action': 'done'}

        # Create the graph
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("supervisor", supervisor)
        workflow.add_node("fetch_data", fetch_data)
        workflow.add_node("analyst", analyze)
        workflow.add_node("visualize", visualize)
        workflow.add_node("report", report)

        # Set entry point
        workflow.set_entry_point("supervisor")

        # Define conditional edges from supervisor
        def route_from_supervisor(state: AgentState) -> Literal["fetch_data", "analyst", "visualize", "report", "__end__"]:
            action = state.get('current_action', 'analyze')
            logger.debug("route_from_supervisor: action=%s", action)
            
            if action == 'fetch_data':
                return 'fetch_data'
            elif action == 'analyze':
                return 'analyst'
            elif action == 'visualize':
                return 'visualize'
            elif action == 'report' or action == 'done':
                return 'report'
            else:
                # Default to analyst for unknown actions
                logger.warning("Unknown action '%s', defaulting to analyst", action)
                return 'analyst'

        workflow.add_conditional_edges(
            source="supervisor",
            path=route_from_supervisor,
        )

        # Workers always return to supervisor (except report which ends)
        workflow.add_edge("fetch_data", "supervisor")
        workflow.add_edge("analyst", "supervisor")
        workflow.add_edge("visualize", "supervisor")
        
        # Report ends the loop
        workflow.add_edge("report", END)

        logger.info("LangGraph state machine built successfully")
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