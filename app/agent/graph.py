"""
LangGraph state machine wiring the agent nodes.

Implements the think/act/observe/verify loop with conditional routing
based on verification results and visualization decisions.
"""

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
        # Initialize dependencies
        self.data_layer = DataLayer(data_dir=data_dir)
        self.model_client = ModelClient(model_path=model_path)
        self.sandbox = SandboxClient(data_dir=data_dir, scratch_dir=scratch_dir)
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Construct the LangGraph state machine."""
        
        # Define node functions with bound dependencies
        def explore(state: AgentState) -> AgentState:
            return explore_node(state, self.data_layer)
        
        def think(state: AgentState) -> AgentState:
            return think_node(state, self.model_client)
        
        def act(state: AgentState) -> AgentState:
            return act_node(state, self.model_client, self.sandbox, self.data_layer)
        
        def observe(state: AgentState) -> AgentState:
            return observe_node(state)
        
        def verify(state: AgentState) -> AgentState:
            return verify_node(state, self.model_client)
        
        def viz_decision(state: AgentState) -> AgentState:
            return viz_decision_node(state, self.model_client)
        
        def respond(state: AgentState) -> AgentState:
            return respond_node(state, self.model_client)
        
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
        def route_from_think(state: AgentState) -> Literal["act", "verify", "visualize", "respond", "explore"]:
            action = state.get('current_action', 'think')
            if action == 'act':
                return 'act'
            elif action == 'verify':
                return 'verify'
            elif action == 'visualize':
                return 'viz_decision'
            elif action == 'respond':
                return 'respond'
            else:
                return 'think'  # Default to more thinking
        
        workflow.add_conditional_edges(
            source="think",
            conditional=route_from_think,
        )
        
        # After act, always observe
        workflow.add_edge("act", "observe")
        
        # After observe, go to verify
        workflow.add_edge("observe", "verify")
        
        # After verify, route based on pass/fail
        def route_from_verify(state: AgentState) -> Literal["think", "viz_decision"]:
            current = state.get('current_action', 'think')
            if current == 'think':
                return 'think'
            else:
                return 'viz_decision'
        
        workflow.add_conditional_edges(
            source="verify",
            conditional=route_from_verify,
        )
        
        # After viz_decision, route based on decision
        def route_from_viz(state: AgentState) -> Literal["act", "respond"]:
            decision = state.get('visualization_decision', {})
            if decision.get('should_visualize', False):
                return 'act'  # Generate chart code
            else:
                return 'respond'
        
        workflow.add_conditional_edges(
            source="viz_decision",
            conditional=route_from_viz,
        )
        
        # Respond ends the loop
        workflow.add_edge("respond", END)
        
        return workflow.compile()
    
    def run(self, query: str) -> dict:
        """
        Run a query through the agent loop.
        
        Args:
            query: Natural language question about the dataset.
        
        Returns:
            Dict with final_response, findings, interpretation, artifacts.
        """
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
        
        result = self.graph.invoke(initial_state)
        return result
    
    def get_artifact(self, artifact_path: str) -> bytes | None:
        """Retrieve a generated artifact (chart PNG) from the sandbox."""
        return self.sandbox.get_artifact(artifact_path)
    
    def cleanup(self):
        """Stop the sandbox container and clean up resources."""
        self.sandbox.stop()
        self.data_layer.close()


# Convenience function for Streamlit integration
def create_agent(data_dir: str, model_path: str | None = None) -> AgentGraph:
    """Create an agent instance for a given data directory."""
    return AgentGraph(data_dir=data_dir, model_path=model_path)
