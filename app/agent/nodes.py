"""
LangGraph agent nodes for the think/act/observe/verify loop.

Each node is a function that takes the current state and returns updates.
Nodes are wired together in graph.py to form the complete agent loop.
"""

import json
import logging
from typing import TypedDict, List, Dict, Any, Optional
from datetime import datetime

from app.agent.model_client import ModelClient
from app.agent.sandbox_client import SandboxClient
from app.agent.data_layer import DataLayer
from app.config import AGENT_MAX_ITERATIONS, AGENT_CONTEXT_MAX_TOKENS

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State passed between nodes in the LangGraph."""
    query: str
    data_profile: Optional[Dict[str, Any]]
    table_names: List[str]
    conversation_history: List[Dict[str, str]]
    current_action: str
    generated_code: Optional[str]
    code_language: Optional[str]
    execution_result: Optional[Dict[str, Any]]
    verification_result: Optional[Dict[str, Any]]
    visualization_decision: Optional[Dict[str, Any]]
    findings: Optional[str]
    interpretation: Optional[str]
    final_response: Optional[str]
    iteration_count: int
    errors: List[str]
    artifacts: List[str]


def load_rulebase() -> str:
    """Load the rulebase.md file as system prompt."""
    from pathlib import Path
    rulebase_path = Path(__file__).parent.parent.parent / 'rulebase.md'
    if rulebase_path.exists():
        content = rulebase_path.read_text()
        logger.info("Rulebase loaded from %s (%d chars)", rulebase_path, len(content))
        return content
    logger.warning("Rulebase not found at %s", rulebase_path)
    return ""  # Fallback: empty rules (agent will have minimal guidance)


RULEBASE = load_rulebase()


def explore_node(state: AgentState, data_layer: DataLayer) -> AgentState:
    """
    Explore node: Profile the dataset once per session.

    Extracts schema, row counts, column stats, and sample values.
    Caches result in state for subsequent turns.
    """
    logger.info("Running explore_node")
    if state.get('data_profile') is not None:
        # Already explored
        logger.debug("Dataset already explored, skipping")
        return {}

    try:
        table_names = data_layer.get_table_names()
    except Exception as e:
        logger.error("Failed to get table names: %s", e)
        return {
            'errors': state.get('errors', []) + [f'Failed to list tables: {e}'],
            'table_names': []
        }

    if not table_names:
        logger.warning("No tables found in dataset")
        return {
            'errors': state.get('errors', []) + ['No tables found in dataset'],
            'table_names': []
        }

    logger.info("Found %d tables: %s", len(table_names), table_names)
    profiles = {}
    for table_name in table_names:
        try:
            profiles[table_name] = data_layer.get_profile(table_name)
            logger.debug("Profiled table %s", table_name)
        except Exception as e:
            logger.error("Failed to profile table %s: %s", table_name, e)
            profiles[table_name] = {'error': str(e)}

    # Create compact summary for LLM context
    summary_parts = []
    for table_name, profile in profiles.items():
        if 'error' in profile:
            continue
        summary_parts.append(f"Table: {table_name}")
        summary_parts.append(f"  Rows: {profile['row_count']}")
        summary_parts.append("  Columns:")
        for col in profile['columns'][:10]:  # Limit columns shown
            summary_parts.append(
                f"    - {col['name']} ({col['type']}): "
                f"{col['distinct_count']} distinct, "
                f"{col['null_count']} nulls"
            )
            if col['sample_values']:
                summary_parts.append(f"      Samples: {', '.join(col['sample_values'][:5])}")

    logger.info("Explore complete, profile summary: %d chars", len("\n".join(summary_parts)))
    return {
        'data_profile': profiles,
        'table_names': table_names,
        'conversation_history': state.get('conversation_history', []) + [{
            'role': 'system',
            'content': f"Dataset profile:\n" + "\n".join(summary_parts)
        }]
    }


def think_node(state: AgentState, model_client: ModelClient) -> AgentState:
    """
    Think node: Decide the next micro-step from fixed action set.

    Actions: explore, think, act, verify, visualize, respond

    Uses grammar-constrained decoding to ensure valid action selection.
    """
    logger.debug("Running think_node, iteration=%d", state.get('iteration_count', 0))
    if state.get('iteration_count', 0) >= AGENT_MAX_ITERATIONS:
        logger.warning("Max iterations reached")
        return {
            'current_action': 'respond',
            'errors': state.get('errors', []) + [f'Max iterations ({AGENT_MAX_ITERATIONS}) reached']
        }

    # Build context from state
    context_parts = []
    if state.get('data_profile'):
        context_parts.append("Dataset is loaded and profiled.")
    if state.get('findings'):
        context_parts.append(f"Current findings: {state['findings']}")
    if state.get('execution_result'):
        result = state['execution_result']
        if result.get('result_repr'):
            context_parts.append(f"Last execution result: {result['result_repr'][:500]}")

    context = "\n".join(context_parts) or "Starting fresh analysis."

    try:
        # Call model with grammar constraints
        decision = model_client.think_step(
            system_prompt=RULEBASE,
            context=context,
            user_query=state['query']
        )
    except Exception as e:
        logger.error("think_step failed: %s", e)
        return {
            'current_action': 'respond',
            'errors': state.get('errors', []) + [f'Model error: {e}']
        }

    action = decision.get('action', 'think')
    reason = decision.get('reason', 'No reason provided')
    logger.info("Think decided: action=%s, reason=%s", action, reason[:100] if reason else '')

    return {
        'current_action': action,
        'conversation_history': state.get('conversation_history', []) + [{
            'role': 'assistant',
            'content': f"Thinking: Next action is '{action}' because {reason}"
        }],
        'iteration_count': state.get('iteration_count', 0) + 1
    }


def act_node(state: AgentState, model_client: ModelClient, sandbox: SandboxClient, data_layer: DataLayer) -> AgentState:
    """
    Act node: Generate and execute code in the sandbox.

    Generates Python or SQL based on task, executes via docker exec,
    captures results and any generated artifacts (charts).
    """
    task_description = f"Answer the query: {state['query']}"
    if state.get('findings'):
        task_description += f"\nPrevious findings: {state['findings']}"

    # Build data context
    data_context = f"Available tables: {', '.join(state.get('table_names', []))}"
    if state.get('data_profile'):
        data_context += "\nSchema info available via data_layer.get_schema()"

    # Generate code with grammar constraints
    code_result = model_client.generate_code(
        system_prompt=RULEBASE,
        task_description=task_description,
        data_context=data_context
    )

    code = code_result.get('code', '')
    language = code_result.get('language', 'python')

    if not code:
        return {
            'errors': state.get('errors', []) + ['Failed to generate code'],
            'generated_code': None
        }

    # Execute in sandbox
    execution_result = sandbox.execute(code, timeout=30)

    # Collect artifacts
    new_artifacts = execution_result.get('artifacts', [])

    return {
        'generated_code': code,
        'code_language': language,
        'execution_result': execution_result,
        'artifacts': state.get('artifacts', []) + new_artifacts,
        'conversation_history': state.get('conversation_history', []) + [{
            'role': 'assistant',
            'content': f"Executed {language} code. Result: {execution_result.get('result_repr', 'N/A')[:200]}"
        }]
    }


def observe_node(state: AgentState) -> AgentState:
    """
    Observe node: Parse and summarize sandbox execution result.

    Extracts stdout, stderr, result_repr, and error status.
    Formats for downstream nodes (verify, respond).
    """
    execution_result = state.get('execution_result')
    if not execution_result:
        return {'errors': state.get('errors', []) + ['No execution result to observe']}

    error = execution_result.get('error')
    stdout = execution_result.get('stdout', '')
    stderr = execution_result.get('stderr', '')
    result_repr = execution_result.get('result_repr', '')

    observation = {
        'success': error is None,
        'output': stdout[:1000] if stdout else '',
        'error': error,
        'result_summary': result_repr[:500] if result_repr else ''
    }

    if stderr:
        observation['warnings'] = stderr[:500]

    return {
        'conversation_history': state.get('conversation_history', []) + [{
            'role': 'system',
            'content': f"Observation: {'Success' if observation['success'] else 'Error'} - {observation.get('error', observation.get('result_summary', 'N/A'))[:200]}"
        }]
    }


def verify_node(state: AgentState, model_client: ModelClient) -> AgentState:
    """
    Verify node: Check execution results against quality checklist.

    Routes back to think_node on failure, forward on pass.
    Uses grammar-constrained JSON output for reliable parsing.
    """
    if not state.get('execution_result'):
        return {
            'verification_result': {'pass': False, 'issues': ['No result to verify']},
            'current_action': 'think'
        }

    task = state['query']
    code = state.get('generated_code', 'N/A')
    result = state['execution_result'].get('result_repr', 'N/A')

    if state['execution_result'].get('error'):
        result = f"Error: {state['execution_result']['error']}"

    verification = model_client.verify_result(
        system_prompt=RULEBASE,
        task=task,
        code=code,
        result=result
    )

    passed = verification.get('pass', False)
    issues = verification.get('issues', [])

    return {
        'verification_result': verification,
        'current_action': 'think' if not passed else 'visualize',
        'errors': state.get('errors', []) + issues if not passed else state.get('errors', [])
    }


def viz_decision_node(state: AgentState, model_client: ModelClient) -> AgentState:
    """
    Visualization decision node: Determine if a chart would help.

    Uses rule table from rulebase.md §6.
    Returns chart type recommendation if visualization is warranted.
    """
    findings = state.get('findings', '')
    if not findings and state.get('execution_result'):
        findings = state['execution_result'].get('result_repr', '')

    decision = model_client.decide_visualization(
        system_prompt=RULEBASE,
        findings=findings[:1000],
        query=state['query']
    )

    should_viz = decision.get('should_visualize', False)
    chart_type = decision.get('chart_type')

    return {
        'visualization_decision': decision,
        'current_action': 'act' if should_viz else 'respond'
    }


def respond_node(state: AgentState, model_client: ModelClient) -> AgentState:
    """
    Respond node: Format final answer per rulebase.md §7.

    Includes findings, business-relevant interpretation, and follow-up questions.
    """
    findings = state.get('findings', 'Analysis completed.')
    if not findings and state.get('execution_result'):
        findings = state['execution_result'].get('result_repr', 'No results captured.')

    # Generate business interpretation
    interpretation = f"Based on the analysis of {', '.join(state.get('table_names', ['the dataset']))}, "
    interpretation += f"the key finding is: {findings[:500]}..."

    response = model_client.generate_response(
        system_prompt=RULEBASE,
        query=state['query'],
        findings=findings,
        interpretation=interpretation
    )

    return {
        'findings': findings,
        'interpretation': interpretation,
        'final_response': response
    }