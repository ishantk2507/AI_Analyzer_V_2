"""
LangGraph agent nodes for the supervisor-worker architecture.

Each node is a function that takes the current state and returns updates.
Nodes are wired together in graph.py to form the complete agent loop.

Agent Roles:
1. Supervisor: Routes to specialists (fetch_data, analyze, visualize, report, done)
2. Data Fetch: Loads dataset schema and profiles tables (pure Python, no LLM)
3. Analyst: Generates SQL/Python code, executes in sandbox
4. Visualization: Generates matplotlib charts
5. Reporting: Generates final natural language response
"""

import json
import logging
import re
from typing import TypedDict, List, Dict, Any, Optional
from datetime import datetime

from app.agent.model_client import ModelClient, _extract_json_fields_regex
from app.agent.sandbox_client import SandboxClient
from app.agent.data_layer import DataLayer
from app.config import AGENT_MAX_ITERATIONS, AGENT_CONTEXT_MAX_TOKENS

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State passed between nodes in the LangGraph."""
    query: str
    data_profile: Optional[Dict[str, Any]]
    table_names: List[str]
    schema_summary: Optional[str]
    conversation_history: List[Dict[str, str]]
    current_action: str
    generated_code: Optional[str]
    code_language: Optional[str]
    execution_result: Optional[Dict[str, Any]]
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


def _extract_digit_starting_columns(data_profile: Optional[Dict[str, Any]]) -> List[str]:
    """Extract column names that start with digits from the data profile."""
    digit_cols = []
    if not data_profile:
        return digit_cols
    
    for table_name, profile in data_profile.items():
        if isinstance(profile, dict) and 'columns' in profile:
            for col in profile['columns']:
                col_name = col.get('name', '')
                if col_name and col_name[0].isdigit():
                    digit_cols.append(col_name)
    return digit_cols


def _quote_sql_identifier(name: str) -> str:
    """Wrap a column name in double quotes for SQL."""
    return f'"{name}"'


def supervisor_node(state: AgentState, model_client: ModelClient) -> AgentState:
    """
    Supervisor node: The ONLY routing decision maker.
    
    Sees full state and decides which specialist to invoke next.
    Does NOT write SQL, code, or verification logic.
    
    Returns: {'current_action': <next>, 'conversation_history': [...]}
    
    If model returns unknown action (e.g., "explore"), defaults to "report" to terminate.
    """
    logger.debug("Running supervisor_node, iteration=%d", state.get('iteration_count', 0))
    
    try:
        # Build context from state
        context_parts = []
        
        # User query
        context_parts.append(f"User Query: {state.get('query', 'Unknown')}")
        
        # Schema summary
        if state.get('schema_summary'):
            context_parts.append(f"\nDataset Schema:\n{state['schema_summary']}")
        elif state.get('table_names'):
            context_parts.append(f"\nAvailable Tables: {', '.join(state['table_names'])}")
        
        # Current findings
        if state.get('findings'):
            context_parts.append(f"\nCurrent Findings:\n{state['findings']}")
        
        # Last execution error (if any)
        if state.get('errors') and state['errors'][-1]:
            context_parts.append(f"\nLast Error: {state['errors'][-1]}")
        
        # Completed steps
        completed_steps = []
        if state.get('data_profile'):
            completed_steps.append("Data loaded and profiled")
        if state.get('generated_code'):
            completed_steps.append("Code generated and executed")
        if state.get('artifacts'):
            completed_steps.append(f"Artifacts generated: {len(state['artifacts'])} files")
        
        if completed_steps:
            context_parts.append(f"\nCompleted Steps: {'; '.join(completed_steps)}")
        
        context = "\n".join(context_parts)
        
        # Call supervisor_step
        decision = model_client.supervisor_step(
            system_prompt=RULEBASE,
            context=context,
            user_query=state['query']
        )
        
        next_action = decision.get('next', 'analyze')
        reason = decision.get('reason', 'No reason provided')
        
        # Validate action: only allow known actions
        valid_actions = {'fetch_data', 'analyze', 'visualize', 'report', 'done'}
        if next_action not in valid_actions:
            logger.warning("Supervisor returned invalid action '%s', defaulting to 'report'", next_action)
            next_action = 'report'
            reason = f"Invalid action '{next_action}' was returned by model, forcing report to terminate"
            # Add error to state so we know this happened
            state.setdefault('errors', []).append(f"Supervisor hallucinated action: {next_action}")
        
        # Hard guardrails
        # If data_profile exists, force next != fetch_data
        if state.get('data_profile') and next_action == 'fetch_data':
            logger.warning("Supervisor tried to fetch_data but data already loaded, forcing analyze")
            next_action = 'analyze'
        
        # If max iterations reached, force done
        if state.get('iteration_count', 0) >= AGENT_MAX_ITERATIONS:
            logger.warning("Max iterations reached, forcing done")
            next_action = 'done'
        
        logger.info("Supervisor decided: next=%s, reason=%s", next_action, reason[:100] if reason else '')
        
        return {
            'current_action': next_action,
            'conversation_history': state.get('conversation_history', []) + [{
                'role': 'assistant',
                'content': f"Supervisor: Next action is '{next_action}' because {reason}"
            }],
            'iteration_count': state.get('iteration_count', 0) + 1
        }
        
    except Exception as e:
        logger.error("supervisor_node failed: %s", e)
        return {
            'errors': state.get('errors', []) + [f'Supervisor error: {e}'],
            'current_action': 'report'
        }


def data_fetch_node(state: AgentState, data_layer: DataLayer) -> AgentState:
    """
    Data fetch node: Load dataset schema and profile tables.
    
    Pure Python, no LLM call.
    Guardrail: Skip if data_profile already exists.
    
    Returns: {'data_profile', 'table_names', 'schema_summary'}
    """
    logger.info("Running data_fetch_node")
    
    # Guardrail: skip if already fetched
    if state.get('data_profile') is not None:
        logger.debug("Data already fetched, skipping")
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
    
    # Profile each table
    profiles = {}
    for table_name in table_names:
        try:
            profiles[table_name] = data_layer.get_profile(table_name)
            logger.debug("Profiled table %s", table_name)
        except Exception as e:
            logger.error("Failed to profile table %s: %s", table_name, e)
            profiles[table_name] = {'error': str(e)}
    
    # Build compact schema summary for LLM context
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
            if col.get('sample_values'):
                summary_parts.append(f"      Samples: {', '.join(col['sample_values'][:5])}")
    
    schema_summary = "\n".join(summary_parts)
    logger.info("Data fetch complete, schema summary: %d chars", len(schema_summary))
    
    return {
        'data_profile': profiles,
        'table_names': table_names,
        'schema_summary': schema_summary,
        'conversation_history': state.get('conversation_history', []) + [{
            'role': 'system',
            'content': f"Dataset loaded:\n{schema_summary}"
        }]
    }


def analyst_node(state: AgentState, model_client: ModelClient, sandbox: SandboxClient, data_layer: DataLayer) -> AgentState:
    """
    Analyst node: Generate SQL/Python based on query and schema.
    
    Handles its own error reporting.
    Post-processes generated code to quote digit-starting column names.
    
    Returns: {'generated_code', 'code_language', 'execution_result', 'findings', 'conversation_history'}
    """
    logger.info("Running analyst_node")
    
    task_description = f"Answer the query: {state['query']}"
    if state.get('findings'):
        task_description += f"\nPrevious findings: {state['findings']}"
    
    # Build data context with explicit rules for digit-starting columns
    data_context_parts = []
    data_context_parts.append(f"Available tables: {', '.join(state.get('table_names', []))}")
    
    # Extract digit-starting columns and add explicit rule
    digit_cols = _extract_digit_starting_columns(state.get('data_profile'))
    if digit_cols:
        data_context_parts.append(f"\nIMPORTANT: Column names starting with digits MUST be double-quoted in SQL.")
        data_context_parts.append(f"Digit-starting columns found: {', '.join(digit_cols)}")
        data_context_parts.append(f"Example: SELECT \"{digit_cols[0]}\" FROM table_name")
    
    if state.get('schema_summary'):
        data_context_parts.append(f"\nSchema Info:\n{state['schema_summary']}")
    
    data_context = "\n".join(data_context_parts)
    
    # Generate code
    code_result = model_client.generate_code(
        system_prompt=RULEBASE,
        task_description=task_description,
        data_context=data_context
    )
    
    # Handle error response from generate_structured
    if 'error' in code_result:
        logger.warning("Code generation returned error: %s", code_result.get('error'))
        # Try regex fallback to extract code from raw response
        raw = code_result.get('raw', '')
        code = ''
        language = 'python'
        
        if raw:
            # First try extracting from markdown code block (most reliable for multi-line SQL)
            code_match = re.search(r'```(?:sql|python)?\s*([\s\S]*?)```', raw, re.IGNORECASE)
            if code_match:
                code = code_match.group(1).strip()
                language = 'sql' if 'SELECT' in code.upper() or 'FROM' in code.upper() else 'python'
                logger.info("Regex fallback extracted code from markdown block (%d chars)", len(code))
            
            # If no markdown block, try field extraction
            if not code:
                extracted = _extract_json_fields_regex(raw, ['code', 'language'])
                if extracted and extracted.get('code'):
                    code = extracted['code']
                    language = extracted.get('language', 'python')
                    logger.info("Regex fallback extracted code from JSON fields (%d chars)", len(code))
            
            # Last resort: look for SQL keywords and extract everything that looks like SQL
            if not code:
                sql_match = re.search(r'(SELECT[\s\S]*?FROM[\s\S]*?(?:WHERE[\s\S]*?)?)(?:$|\"|\})', raw, re.IGNORECASE)
                if sql_match:
                    code = sql_match.group(1).strip()
                    language = 'sql'
                    logger.info("Regex fallback extracted SQL query (%d chars)", len(code))
    else:
        code = code_result.get('code', '')
        language = code_result.get('language', 'python')
    
    if not code:
        logger.warning("Code generation returned empty code")
        return {
            'errors': state.get('errors', []) + ['Failed to generate code'],
            'generated_code': None,
            'findings': 'Code generation failed: ' + code_result.get('error', 'Unknown error'),
            'conversation_history': state.get('conversation_history', []) + [{
                'role': 'assistant',
                'content': 'Analyst: Code generation failed.'
            }]
        }
    
    # Post-process: wrap bare digit-starting column names in double quotes
    # This regex finds digit-starting identifiers that are NOT already quoted
    for col_name in digit_cols:
        # Match the column name when NOT preceded by a double quote
        pattern = r'(?<!")\b' + re.escape(col_name) + r'\b(?!")'
        code = re.sub(pattern, _quote_sql_identifier(col_name), code)
    
    logger.debug("Generated code (post-processed):\n%s", code[:500])
    
    # Execute in sandbox
    execution_result = sandbox.execute(code, timeout=30)
    
    # Handle execution result
    error = execution_result.get('error')
    if error:
        logger.warning("Code execution failed: %s", error)
        findings = f"Execution failed: {error}"
    else:
        result_repr = execution_result.get('result_repr', '')
        stdout = execution_result.get('stdout', '')
        findings = result_repr or stdout or "Code executed successfully."
    
    # Collect artifacts
    new_artifacts = execution_result.get('artifacts', [])
    
    return {
        'generated_code': code,
        'code_language': language,
        'execution_result': execution_result,
        'findings': findings,
        'artifacts': state.get('artifacts', []) + new_artifacts,
        'conversation_history': state.get('conversation_history', []) + [{
            'role': 'assistant',
            'content': f"Analyst: Executed {language} code. Result: {findings[:200]}"
        }]
    }


def viz_node(state: AgentState, model_client: ModelClient, sandbox: SandboxClient) -> AgentState:
    """
    Visualization node: Generate matplotlib charts from findings.
    
    Only runs when supervisor routes to 'visualize'.
    
    Returns: {'artifacts', 'execution_result', 'conversation_history'}
    """
    logger.info("Running viz_node")
    
    findings = state.get('findings', 'No findings available.')
    query = state.get('query', 'Unknown query')
    
    task_description = (
        f"You are a visualization expert. Write Python matplotlib code to chart these findings.\n\n"
        f"Query: {query}\n"
        f"Findings:\n{findings}\n\n"
        f"Create a clear, professional chart. Save it to /scratch/chart.png"
    )
    
    data_context = "Use matplotlib for plotting. Save figures to /scratch/ directory."
    
    # Generate code
    code_result = model_client.generate_code(
        system_prompt=RULEBASE,
        task_description=task_description,
        data_context=data_context
    )
    
    code = code_result.get('code', '')
    language = code_result.get('language', 'python')
    
    if not code:
        logger.warning("Viz code generation returned empty code")
        return {
            'errors': state.get('errors', []) + ['Failed to generate visualization code'],
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
            'content': f"Visualization: Generated chart. Artifacts: {new_artifacts}"
        }]
    }


def report_node(state: AgentState, model_client: ModelClient) -> AgentState:
    """
    Reporting node: Generate final natural language response.
    
    Returns: {'final_response', 'interpretation', 'current_action': 'done'}
    """
    logger.info("Running report_node")
    
    findings = state.get('findings', 'Analysis completed.')
    errors = state.get('errors', [])
    
    # Build interpretation
    table_names = state.get('table_names', ['the dataset'])
    interpretation = f"Based on the analysis of {', '.join(table_names)}, "
    
    if errors:
        interpretation += f"some issues occurred: {', '.join(errors[-2:])}. "
    
    interpretation += f"The key finding is: {findings[:500]}..."
    
    # Generate response
    response = model_client.generate_response(
        system_prompt=RULEBASE,
        query=state['query'],
        findings=findings,
        interpretation=interpretation
    )
    
    logger.info("Report generated: %s", response[:200])
    
    return {
        'findings': findings,
        'interpretation': interpretation,
        'final_response': response,
        'current_action': 'done',
        'conversation_history': state.get('conversation_history', []) + [{
            'role': 'assistant',
            'content': f"Report: {response[:200]}"
        }]
    }