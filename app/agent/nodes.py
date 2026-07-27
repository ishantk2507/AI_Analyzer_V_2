"""
LangGraph agent nodes for the supervisor-worker architecture.

FIXES APPLIED:
1. Analyst Node: Implements TRUE two-phase execution (SQL → df → Python).
   - Calls analyst_generate() for {sql, python} schema.
   - Executes SQL via DataLayer to populate 'df'.
   - Injects 'df' into sandbox for Python execution.
2. Thinker Node: Fixed guardrail ordering. Max iterations now forces 'report' explicitly.
3. Report Node: Removed hallucinated follow-up questions. Strictly summarizes findings.
"""

import json
import logging
import re
from typing import TypedDict, List, Dict, Any, Optional
from datetime import datetime

from app.agent.model_client import ModelClient, _extract_json_fields_regex
from app.agent.sandbox_client import SandboxClient
from app.agent.data_layer import DataLayer
from app.config import RULEBASE_PATH

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State passed between nodes in the LangGraph."""
    query: str
    data_profile: Optional[Dict[str, Any]]
    table_names: List[str]
    schema_summary: Optional[str]
    conversation_history: List[Dict[str, str]]
    current_action: str
    generated_code: Optional[Dict[str, str]]  # Changed to dict {sql, python}
    code_language: Optional[str]
    execution_result: Optional[Dict[str, Any]]
    findings: Optional[str]
    interpretation: Optional[str]
    final_response: Optional[str]
    iteration_count: int
    errors: List[str]
    artifacts: List[str]
    success: bool
    analyst_feedback: Optional[str]
    thinker_reason: Optional[str]


def load_rulebase() -> str:
    """Load the rulebase.md file as system prompt."""
    if RULEBASE_PATH.exists():
        content = RULEBASE_PATH.read_text()
        logger.info("Rulebase loaded from %s (%d chars)", RULEBASE_PATH, len(content))
        return content
    logger.warning("Rulebase not found at %s", RULEBASE_PATH)
    return ""

def _parse_max_iterations(rulebase_text: str, default: int = 6) -> int:
    m = re.search(r'Max iterations:\s*(\d+)', rulebase_text)
    return int(m.group(1)) if m else default


# Global cache for rulebase with file modification time tracking
_rulebase_cache = {"content": "", "mtime": 0.0}

def get_rulebase() -> str:
    """
    Get the current rulebase content, reloading if the file has changed.
    
    This enables hot-reloading: any edits to rulebase.md are picked up
    automatically on the next query without restarting the application.
    """
    global _rulebase_cache
    
    try:
        current_mtime = RULEBASE_PATH.stat().st_mtime if RULEBASE_PATH.exists() else 0.0
        
        if current_mtime != _rulebase_cache["mtime"]:
            # File has changed, reload it
            _rulebase_cache["content"] = load_rulebase()
            _rulebase_cache["mtime"] = current_mtime
            logger.info("Rulebase reloaded due to file change")
        
        return _rulebase_cache["content"]
    except Exception as e:
        logger.warning("Failed to check rulebase file status: %s", e)
        return _rulebase_cache["content"] or load_rulebase()


def get_agent_max_iterations() -> int:
    """Get the current max iterations from rulebase, with hot-reload support."""
    rulebase = get_rulebase()
    return _parse_max_iterations(rulebase)

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


def thinker_node(state: AgentState, model_client: ModelClient) -> AgentState:
    """
    FIX #2: Guardrail ordering fixed. Max iterations check happens LAST and overrides all.
    """
    iteration = state.get('iteration_count', 0)
    
    # Build context
    context_parts = [
        f"Query: {state['query']}",
        f"Available Tables: {state.get('table_names', [])}",
    ]
    
    if state.get('execution_result'):
        err = state['execution_result'].get('error')
        if err:
            context_parts.append(f"Last Error: {str(err)[:200]}")
        res = state['execution_result'].get('result_repr')
        if res:
            context_parts.append(f"Last Result: {res[:200]}")
            
    if state.get('generated_code'):
        code = state['generated_code']
        if isinstance(code, dict):
            if code.get('sql'): context_parts.append(f"Last SQL: {code['sql'][:100]}")
            if code.get('python'): context_parts.append(f"Last Python: {code['python'][:100]}")

    context = "\n".join(context_parts)
    
    logger.info("=== THINKER INPUT ===\n%s", context)

    # Call LLM
    try:
        rulebase = get_rulebase()
        decision = model_client.thinker_decide(rulebase, context)
        logger.info("=== THINKER RAW OUTPUT ===\n%s", decision)
    except Exception as e:
        logger.error("Thinker LLM call failed: %s", e)
        decision = {'action': 'report', 'reason': 'LLM error'}

    action = decision.get('action', 'report')
    reason = decision.get('reason', '')
    
    # Guardrail 1: Validate action
    VALID = {'extract', 'analyze', 'visualize', 'report', 'done'}
    if action not in VALID:
        logger.warning("Invalid action '%s' from LLM, defaulting to report", action)
        action = 'report'
    
    # Guardrail 2: Handle failure retry (ONLY if not at max iterations)
    if not state.get('success') and action not in {'extract', 'analyze', 'report', 'done'}:
        action = 'analyze'
    
    # Guardrail 3: MAX ITERATIONS CHECK (HAPPENS LAST - OVERRIDES ALL)
    max_iterations = get_agent_max_iterations()
    if iteration >= max_iterations:
        logger.warning("Max iterations (%d) reached. Forcing 'report' to terminate.", max_iterations)
        action = 'report'  # Explicitly route to report to explain failure
        reason = "Max iterations reached"

    # Construct feedback in Python (NOT LLM)
    feedback = None
    if action in ('extract', 'analyze') and iteration < max_iterations:
        last_error = (state.get('execution_result') or {}).get('error', '')
        
        if action == 'extract':
            if last_error and 'does not exist' in str(last_error):
                real_tables = state.get('table_names', [])
                feedback = f"Use ONLY these tables: {real_tables}. Do not invent names."
            else:
                feedback = "Extract data using SQL. Filter rows, select columns. No aggregation."
        
        elif action == 'analyze':
            if last_error:
                feedback = f"Code failed: {str(last_error)[:150]}. Fix the error."
            else:
                feedback = "Analyze the DataFrame using pandas."

    logger.info("Thinker Decision: action=%s, reason=%s", action, reason)
    
    return {
        'current_action': action,
        'thinker_reason': reason,
        'analyst_feedback': feedback,
        'iteration_count': iteration + 1
    }


def analyst_node(state: AgentState, model_client: ModelClient, sandbox: SandboxClient, data_layer: DataLayer) -> AgentState:
    """
    FIX #1: TRUE Two-Phase Execution.
    1. Call analyst_generate() to get {sql, python}.
    2. Execute SQL via DataLayer to get pandas DataFrame 'df'.
    3. Inject 'df' into sandbox and execute Python code.
    """
    iteration = state.get('iteration_count', 0)
    logger.info("=== ANALYST NODE START (Iteration %d) ===", iteration)
    
    # Build Table Manifest (Prevent Hallucination)
    table_names = state.get('table_names', [])
    table_manifest = "REGISTERED TABLES (YOU MUST USE ONLY THESE):\n"
    for t in table_names:
        table_manifest += f"- user_data.{t}\n"
    table_manifest += "\nRULE: Do NOT invent table names like 'sales', 'states_2024', etc.\n"
    
    # Get Digit Columns
    digit_cols = _extract_digit_starting_columns(state.get('data_profile'))
    quoting_rule = ""
    if digit_cols:
        quoted_list = ', '.join([f'"{c}"' for c in digit_cols])
        quoting_rule = f"\nMUST QUOTE THESE COLUMNS IN SQL: {quoted_list}\n"
    
    # Build Context
    context_parts = [table_manifest, quoting_rule]
    if state.get('schema_summary'):
        context_parts.append(f"Schema:\n{state['schema_summary']}")
    context_parts.append(f"Query: {state['query']}")
    
    if state.get('analyst_feedback'):
        context_parts.append(f"Instruction: {state['analyst_feedback']}")
    
    context = "\n".join(context_parts)
    logger.info("=== ANALYST PROMPT CONTEXT ===\n%s", context[:1000])

    # CALL ANALYST GENERATE (Two-Phase Schema)
    try:
        rulebase = get_rulebase()
        result = model_client.analyst_generate(rulebase, context)
        logger.info("=== ANALYST RAW OUTPUT ===\n%s", result)
    except Exception as e:
        logger.error("Analyst LLM call failed: %s", e)
        return {
            'errors': state.get('errors', []) + [f'LLM error: {e}'],
            'findings': 'Code generation failed.',
            'success': False,
            'iteration_count': iteration + 1
        }

    sql = result.get('sql', '')
    python_code = result.get('python', '')
    
    # Fallback parsing if needed
    if not sql and not python_code:
        fallback = _extract_json_fields_regex(str(result), ['sql', 'python'])
        if fallback:
            sql = fallback.get('sql', '')
            python_code = fallback.get('python', '')

    sql = _fix_table_references(sql, table_names)
    sql = re.sub(r'`([^`]+)`', r'"\1"', sql)  # backtick backstop, still worth keeping
    for col in digit_cols:
        sql = re.sub(rf'(?<!")\b{re.escape(col)}\b(?!")', f'"{col}"', sql)
    
    logger.info("Generated SQL: %s", sql[:200] if sql else "None")
    logger.info("Generated Python: %s", python_code[:200] if python_code else "None")

    # EXECUTE TWO-PHASE
    execution_result = {}
    findings = ""
    success = False

    if not sql.strip() and not python_code.strip():
        return {
            'generated_code': {'sql': '', 'python': ''},
            'findings': 'Analyst produced no SQL or Python — generation failed.',
            'errors': state.get('errors', []) + ['Empty analyst generation'],
            'success': False,
            'iteration_count': iteration + 1
        }
    
    try:
        # PHASE 1: SQL Extraction via DataLayer
        df = None
        if sql.strip():
            logger.info("Executing SQL Phase...")
            df = data_layer.conn.execute(sql).fetchdf()
            execution_result['df_shape'] = df.shape
            logger.info("SQL Phase Complete. Rows: %d", len(df))
        else:
            # Empty DF if no SQL
            import pandas as pd
            df = pd.DataFrame()
            logger.info("No SQL provided. Using empty DataFrame.")
        
        # PHASE 2: Python Analysis in Sandbox with 'df' injected
        if python_code.strip():
            logger.info("Executing Python Phase...")
            # Serialize df to JSON for safe injection into sandbox
            df_json = df.to_json(orient='split')
            bootstrap_code = f"""
                import pandas as pd
                import json
                df = pd.read_json('''{df_json}''', orient='split')
                """
            full_code = bootstrap_code + "\n" + python_code
            execution_result = sandbox.execute(full_code, timeout=30)
            
            # Check for result variable
            if execution_result.get('error'):
                raise Exception(execution_result['error'])
                
            findings = execution_result.get('result_repr', 'Analysis complete.')
            success = True
        else:
            # If only SQL, return DF info
            findings = f"SQL Execution Successful. Shape: {df.shape}\nHead:\n{df.head().to_string()}"
            execution_result['result_repr'] = findings
            success = True
        
    except Exception as e:
        logger.error("Execution failed: %s", e)
        execution_result['error'] = str(e)
        findings = f"Execution failed: {e}"
        success = False
        return {
            'generated_code': {'sql': sql, 'python': python_code},
            'execution_result': execution_result,
            'findings': findings,
            'errors': state.get('errors', []) + [str(e)],
            'success': success,
            'iteration_count': iteration + 1
        }

    return {
        'generated_code': {'sql': sql, 'python': python_code},
        'code_language': 'python',
        'execution_result': execution_result,
        'findings': findings,
        'success': success,
        'iteration_count': iteration + 1
    }


def data_fetch_node(state: AgentState, data_layer: DataLayer) -> AgentState:
    """Data fetch node: Load dataset schema and profile tables."""
    logger.info("Running data_fetch_node")

    if state.get('data_profile') is not None:
        logger.debug("Data already fetched, skipping")
        return {}

    try:
        table_names = data_layer.get_table_names()
    except Exception as e:
        logger.error("Failed to get table names: %s", e)
        return {'errors': state.get('errors', []) + [f'Failed to list tables: {e}'], 'table_names': []}

    if not table_names:
        logger.warning("No tables found in dataset")
        return {'errors': state.get('errors', []) + ['No tables found in dataset'], 'table_names': []}

    logger.info("Found %d tables: %s", len(table_names), table_names)

    profiles = {}
    for table_name in table_names:
        try:
            profiles[table_name] = data_layer.get_profile(table_name)
        except Exception as e:
            logger.error("Failed to profile table %s: %s", table_name, e)
            profiles[table_name] = {'error': str(e)}

    summary_parts = []
    for table_name, profile in profiles.items():
        if 'error' in profile:
            continue
        summary_parts.append(f"Table: user_data.{table_name}")
        summary_parts.append(f"  Rows: {profile['row_count']}")
        summary_parts.append("  Columns:")
        for col in profile['columns'][:10]:
            summary_parts.append(f"    - {col['name']} ({col['type']}): {col['distinct_count']} distinct")
            if col.get('sample_values'):
                summary_parts.append(f"      Samples: {', '.join(col['sample_values'][:5])}")

    schema_summary = "\n".join(summary_parts)
    logger.info("Data fetch complete, schema summary: %d chars", len(schema_summary))

    return {
        'data_profile': profiles,
        'table_names': table_names,
        'schema_summary': schema_summary,
        'conversation_history': state.get('conversation_history', []) + [{
            'role': 'system', 'content': f"Dataset loaded:\n{schema_summary}"
        }]
    }


def viz_node(state: AgentState, model_client: ModelClient, sandbox: SandboxClient) -> AgentState:
    """Visualization node: Generate matplotlib charts from findings."""
    logger.info("Running viz_node")

    findings = state.get('findings', 'No findings available.')
    query = state.get('query', 'Unknown query')

    task_description = (
        f"You are a visualization expert. Write Python matplotlib code to chart these findings.\n\n"
        f"Query: {query}\nFindings:\n{findings}\n\n"
        f"Create a clear, professional chart. Save it to /scratch/chart.png"
    )

    data_context = "Use matplotlib for plotting. Save figures to /scratch/ directory."

    rulebase = get_rulebase()
    code_result = model_client.generate_code(rulebase, task_description, data_context)
    logger.info("Viz model response: %s", code_result)

    code = code_result.get('code', '')
    language = code_result.get('language', 'python')

    if not code:
        logger.warning("Viz code generation returned empty code")
        return {'errors': state.get('errors', []) + ['Failed to generate visualization code'], 'generated_code': None}

    logger.info("Generated viz code:\n%s", code[:500])

    execution_result = sandbox.execute(code, timeout=30)
    new_artifacts = execution_result.get('artifacts', [])

    if execution_result.get('error'):
        logger.warning("Viz execution failed: %s", execution_result.get('error'))
    else:
        logger.info("Viz execution succeeded, artifacts: %s", new_artifacts)

    return {
        'generated_code': code,
        'code_language': language,
        'execution_result': execution_result,
        'artifacts': state.get('artifacts', []) + new_artifacts,
        'conversation_history': state.get('conversation_history', []) + [{
            'role': 'assistant', 'content': f"Visualization: Generated chart. Artifacts: {new_artifacts}"
        }]
    }


def report_node(state: AgentState, model_client: ModelClient) -> AgentState:
    """
    FIX #3: Reporting node strictly summarizes findings.
    NO hallucinated follow-up questions or next steps.
    """
    logger.info("Running report_node")

    findings = state.get('findings', 'No findings available.')
    errors = state.get('errors', [])
    query = state.get('query', 'Unknown query')
    
    # Build honest interpretation
    table_names = state.get('table_names', ['the dataset'])
    
    if errors:
        interpretation = f"Analysis of {', '.join(table_names)} encountered issues: {', '.join(errors[-2:])}. Findings: {findings[:200]}"
    else:
        interpretation = f"Based on analysis of {', '.join(table_names)}: {findings[:500]}"

    logger.info("Report Interpretation Context: %s", interpretation)

    # Generate response (No follow-up questions instruction)
    try:
        rulebase = get_rulebase()
        response = model_client.generate_response(
            system_prompt=rulebase,
            query=query,
            findings=findings,
            interpretation=interpretation
        )
        logger.info("=== REPORT GENERATED ===\n%s", response)
    except Exception as e:
        logger.error("Report generation failed: %s", e)
        response = f"Unable to generate final report due to error: {e}. Raw findings: {findings}"

    return {
        'findings': findings,
        'interpretation': interpretation,
        'final_response': response,
        'current_action': 'done',
        'conversation_history': state.get('conversation_history', []) + [{
            'role': 'assistant', 'content': f"Report: {response[:500]}"
        }]
    }

def _fix_table_references(sql: str, table_names: List[str]) -> str:
    """Rewrite hallucinated/malformed table names to the real registered table.
    Small local models frequently ignore explicit naming instructions —
    correct deterministically rather than re-prompting."""
    if not table_names:
        return sql

    norm_lookup = {t.lower().replace('_', ''): t for t in table_names}

    def replace_ref(m):
        keyword, raw_name = m.group(1), m.group(2)
        clean = raw_name.split('.')[-1].strip('"\'`')
        key = clean.lower().replace('_', '')
        if key in norm_lookup:
            return f'{keyword} user_data.{norm_lookup[key]}'
        return m.group(0)

    return re.sub(r'\b(FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_.]*)', replace_ref, sql, flags=re.IGNORECASE)