"""
LangGraph state machine wiring the agent nodes.

Implements the think/act/observe/verify loop with conditional routing
based on verification results and visualization decisions.
"""

import logging
from typing import Literal, TypedDict, Any
from langgraph.graph import StateGraph, END

from app.agent.nodes import (
    AgentState,
    explore_node,
    think_node,
    act_node,
    observe_node,
    verify_node,
    viz_decision_node,
    respond_node,
)
from app.agent.model_client import ModelClient
from app.agent.sandbox_client import SandboxClient
from app.agent.data_layer import DataLayer

logger = logging.getLogger(__name__)


class AgentGraph:
    """
    LangGraph-based agent orchestrator.

    Manages the state machine loop and provides a simple interface
    for running queries against a connected dataset.
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
        def explore(state: AgentState) -> AgentState:
            try:
                return explore_node(state, self.data_layer)
            except Exception as e:
                logger.error("explore_node failed: %s", e)
                return {'errors': state.get('errors', []) + [f'Explore error: {e}']}

        def think(state: AgentState) -> AgentState:
            try:
                return think_node(state, self.model_client)
            except Exception as e:
                logger.error("think_node failed: %s", e)
                return {'errors': state.get('errors', []) + [f'Think error: {e}'], 'current_action': 'respond'}

        def act(state: AgentState) -> AgentState:
            try:
                return act_node(state, self.model_client, self.sandbox, self.data_layer)
            except Exception as e:
                logger.error("act_node failed: %s", e)
                return {'errors': state.get('errors', []) + [f'Act error: {e}'], 'generated_code': None}

        def observe(state: AgentState) -> AgentState:
            try:
                return observe_node(state)
            except Exception as e:
                logger.error("observe_node failed: %s", e)
                return {'errors': state.get('errors', []) + [f'Observe error: {e}']}

        def verify(state: AgentState) -> AgentState:
            try:
                return verify_node(state, self.model_client)
            except Exception as e:
                logger.error("verify_node failed: %s", e)
                return {'verification_result': {'pass': False, 'issues': [f'Verify error: {e}']}, 'current_action': 'think'}

        def viz_decision(state: AgentState) -> AgentState:
            try:
                return viz_decision_node(state, self.model_client)
            except Exception as e:
                logger.error("viz_decision_node failed: %s", e)
                return {'visualization_decision': {'should_visualize': False, 'rationale': f'Error: {e}'}, 'current_action': 'respond'}

        def respond(state: AgentState) -> AgentState:
            try:
                return respond_node(state, self.model_client)
            except Exception as e:
                logger.error("respond_node failed: %s", e)
                return {'final_response': f"Error generating response: {e}", 'findings': state.get('findings', 'Unknown'), 'interpretation': 'Error occurred during analysis'}

        # Create the graph
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("explore", explore)
        workflow.add_node("think", think)
        workflow.add_node("act", act)
        workflow.add_node("observe", observe)
        workflow.add_node("verify", verify)
        workflow.add_node("viz_decision", viz_decision)
        workflow.add_node("respond", respond)

        # Set entry point
        workflow.set_entry_point("explore")

        # Define edges with conditional routing
        # After explore, always go to think
        workflow.add_edge("explore", "think")

        # After think, route based on action decision
        def route_from_think(state: AgentState) -> Literal["act", "verify", "viz_decision", "respond", "explore"]:
            action = state.get('current_action', 'think')
            logger.debug("route_from_think: action=%s", action)
            if action == 'act':
                return 'act'
            elif action == 'verify':
                return 'verify'
            elif action == 'visualize':
                return 'viz_decision'  # Map 'visualize' action to 'viz_decision' node
            elif action == 'respond':
                return 'respond'
            else:
                return 'think'  # Default to more thinking

        workflow.add_conditional_edges(
            source="think",
            path=route_from_think,
        )

        # After act, always observe
        workflow.add_edge("act", "observe")

        # After observe, go to verify
        workflow.add_edge("observe", "verify")

        # After verify, route based on pass/fail
        def route_from_verify(state: AgentState) -> Literal["think", "viz_decision"]:
            current = state.get('current_action', 'think')
            verification = state.get('verification_result', {})
            passed = verification.get('pass', False)
            logger.debug("route_from_verify: current=%s, passed=%s", current, passed)
            if not passed:
                return 'think'
            elif current == 'think':
                return 'think'
            else:
                return 'viz_decision'

        workflow.add_conditional_edges(
            source="verify",
            path=route_from_verify,
        )

        # After viz_decision, route based on decision
        def route_from_viz(state: AgentState) -> Literal["act", "respond"]:
            decision = state.get('visualization_decision', {})
            should_viz = decision.get('should_visualize', False)
            logger.debug("route_from_viz: should_visualize=%s", should_viz)
            if should_viz:
                return 'act'  # Generate chart code
            else:
                return 'respond'

        workflow.add_conditional_edges(
            source="viz_decision",
            path=route_from_viz,
        )

        # Respond ends the loop
        workflow.add_edge("respond", END)

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
            'conversation_history': [],
            'current_action': '',
            'generated_code': None,
            'code_language': None,
            'execution_result': None,
            'verification_result': None,
            'visualization_decision': None,
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
            logger.error("Failed to stop sandbox: %s", e)
        try:
            self.data_layer.close()
            logger.info("DataLayer closed successfully")
        except Exception as e:
            logger.error("Failed to close DataLayer: %s", e)


# Convenience function for Streamlit integration
def create_agent(data_dir: str, model_path: str | None = None) -> AgentGraph:
    """Create an agent instance for a given data directory."""
    return AgentGraph(data_dir=data_dir, model_path=model_path)