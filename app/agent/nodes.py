"""
LangGraph agent nodes for the supervisor-worker architecture.

FIXES APPLIED:
1. Analyst Node: Implements TRUE two-phase execution (SQL → df → Python).
   - Calls analyst_generate() for {sql, python} schema.
   - Executes SQL via DataLayer to populate 'df'.
   - Injects 'df' into sandbox for Python execution.
2. Thinker Node: Fixed guardrail ordering. Max iterations now forces 'report' explicitly.
3. Report Node: Removed hallucinated follow-up questions. Strictly summarizes findings.
4. Analyst Node: Retry context now echoes the previous SQL/Python back to the
   model alongside the error feedback, so it has something concrete to
   diverge from instead of regenerating the same mistake blind.
5. Rulebase: split into role-scoped '##' sections via get_rulebase_section().
   Each agent now receives only its own instructions instead of the entire
   file concatenated together — critical on a 4096-token model, where the
   old approach spent ~1700+ tokens of system prompt per call regardless of
   which agent was calling.
"""

import json
import logging
import re
import pandas as pd
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


def _parse_rulebase_sections(rulebase_text: str) -> Dict[str, str]:
    """
    Split the rulebase into named sections keyed by their '## name' header.

    Content before the first '##' header (title + preamble) is intentionally
    dropped here — it's documentation for humans editing the file, never
    meant to be sent to the model. A '###' sub-heading (e.g. "### Column
    Disambiguation") does NOT start a new top-level section; it stays inside
    whichever '##' section it's nested under.
    """
    sections: Dict[str, str] = {}
    current_name: Optional[str] = None
    current_lines: List[str] = []
    for line in rulebase_text.split('\n'):
        m = re.match(r'^##\s+(\S+)\s*$', line.strip())
        if m:
            if current_name:
                sections[current_name] = '\n'.join(current_lines).strip()
            current_name = m.group(1).strip().lower()
            current_lines = []
        elif current_name:
            current_lines.append(line)
    if current_name:
        sections[current_name] = '\n'.join(current_lines).strip()
    return sections


def get_rulebase_section(*names: str) -> str:
    """
    Return only the named section(s) of the rulebase, concatenated in the
    order given. This — NOT get_rulebase() — is what should be sent as the
    system prompt for a specific agent role: each role gets just its own
    instructions instead of every other role's prompt along with it, which
    matters a lot on a 4096-token model where the system prompt competes
    with the schema context and the completion budget for the same window.

    get_rulebase() (whole raw file) still exists and is still used for the
    max-iterations regex lookup, since that value isn't role-scoped.
    """
    sections = _parse_rulebase_sections(get_rulebase())
    parts = []
    for name in names:
        key = name.strip().lower()
        if key in sections:
            parts.append(sections[key])
        else:
            logger.warning("Rulebase section '%s' not found in rulebase.md", name)
    return '\n\n'.join(parts)

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


# Keyword -> subtype codes that must be SELECTed and SUMed together. Keep this
# in sync with the "SUBTYPE COLUMN GROUPS" block in rulebase.md's COLUMN
# GLOSSARY — that's the source of truth for a human reading the rulebase;
# this is the same rule re-expressed so it can be injected dynamically,
# right next to the query, rather than only living in the static system
# prompt where it competes with the schema block for the model's attention
# and is the first thing to fall out of a near-full context window.
_CATEGORY_SUBTYPE_GROUPS: List[tuple] = [
    (r'\btwo[\s-]*wheel', ['2WT', '2WN', '2WIC'], 'Two Wheeler'),
    (r'\bthree[\s-]*wheel', ['3WT', '3WN'], 'Three Wheeler'),
]


def _category_group_hint(query: str, table_columns: Dict[str, List[str]]) -> str:
    """
    If the query names a broad category that maps to more than one subtype
    column on the table(s) actually in play, return a concrete SELECT-all
    reminder. Only fires when 2+ of the group's codes are real columns here
    (so it stays silent on tables that don't have this subtype split).
    """
    all_cols = {c for cols in table_columns.values() for c in cols}
    hints = []
    for pattern, codes, label in _CATEGORY_SUBTYPE_GROUPS:
        if not re.search(pattern, query, re.IGNORECASE):
            continue
        present = [c for c in codes if c in all_cols]
        if len(present) > 1:
            quoted = ' + '.join(f'"{c}"' for c in present)
            hints.append(
                f'NOTE: the query asks about "{label}". On this table that total is '
                f'{quoted} SUMMED together. Your SQL SELECT list must include ALL of '
                f'{present} — selecting only one of them is wrong.'
            )
    return '\n'.join(hints)


def _detect_missing_sql_columns(state: AgentState) -> Optional[str]:
    """
    Detect a specific, previously-unhandled failure mode: the Python phase
    raised a KeyError on a column name that IS a real column in the live
    schema, but the SQL SELECT list simply didn't include it (e.g. Python
    does df.groupby(['STATE', 'YEAR']) but the SQL only selected "2WT").
    This is neither "the column doesn't exist" (already handled by
    _validate_and_fix_columns / the is_column_error branch below) nor a
    genuine Python logic bug (the generic 'analyze' feedback) — it's a
    mismatch between what the SQL fetched and what the Python code assumes
    it fetched. Returns a concrete diagnosis naming the exact missing
    column(s) and what the SQL actually returned, or None if this pattern
    doesn't apply.
    """
    execution_result = state.get('execution_result') or {}
    err = str(execution_result.get('error') or '')
    m = re.search(r"KeyError:\s*'([^']+)'", err)
    if not m:
        return None
    missing_col = m.group(1)

    # The sandbox bootstrap always prints the actual columns it received
    # (see analyst_node's `schema_check`) — read that back rather than
    # re-deriving it from the SQL text, since it reflects what actually
    # reached pandas.
    stdout = execution_result.get('stdout', '') or ''
    cols_match = re.search(r"COLUMNS:\s*(\[[^\]]*\])", stdout)
    returned_cols: List[str] = []
    if cols_match:
        try:
            returned_cols = eval(cols_match.group(1), {"__builtins__": {}})
        except Exception:
            returned_cols = []

    if missing_col in returned_cols:
        return None  # column WAS returned — some other kind of KeyError

    # Only blame the SQL if the missing name is a real column somewhere in
    # the live schema — otherwise this is a typo/hallucinated name, which
    # is a different problem the generic branch already covers.
    data_profile = state.get('data_profile') or {}
    is_real_column = any(
        any(c.get('name') == missing_col for c in profile.get('columns', []))
        for profile in data_profile.values()
        if isinstance(profile, dict)
    )
    if not is_real_column:
        return None

    return (
        "YOUR SQL DID NOT SELECT A COLUMN YOUR PYTHON CODE NEEDS. "
        f"Your SQL only returned these columns: {returned_cols or '(unknown)'}. "
        f"Your Python code references '{missing_col}', which is a real column but was "
        f"left out of the SQL SELECT list. Rewrite the SQL to also SELECT \"{missing_col}\" "
        f"(keep it quoted if it starts with a digit). Do not rename or guess a different "
        f"column — '{missing_col}' is correct, it just needs to be added to SELECT."
    )


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
        shape = state['execution_result'].get('df_shape')
        if shape is not None:
            context_parts.append(f"Last SQL row count: {shape[0]}")
            if shape[0] == 0:
                context_parts.append("The last SQL extraction matched ZERO rows — the filter value was likely wrong, not the column/table.")
        res = state['execution_result'].get('result_repr')
        if res:
            context_parts.append(f"Last Result: {res[:200]}")

    # Give the thinker the real diagnosis (not just the raw truncated
    # traceback) when we can determine one deterministically — it has been
    # observed inventing a plausible-sounding but factually wrong reason
    # otherwise (e.g. blaming a rename(column) typo that wasn't in the code).
    missing_cols_diagnosis = _detect_missing_sql_columns(state)
    if missing_cols_diagnosis:
        context_parts.append(f"Diagnosis: {missing_cols_diagnosis}")

    if state.get('generated_code'):
        code = state['generated_code']
        if isinstance(code, dict):
            if code.get('sql'): context_parts.append(f"Last SQL: {code['sql'][:100]}")
            if code.get('python'): context_parts.append(f"Last Python: {code['python'][:100]}")

    context = "\n".join(context_parts)

    logger.info("=== THINKER INPUT ===\n%s", context)

    # Call LLM
    try:
        rulebase = get_rulebase_section('thinker_system_prompt')
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
        err_str_top = str(last_error) if last_error else ''

        # A crashed/killed sandbox process is never a schema or column
        # problem — it means the query loaded or materialized far more data
        # than it needed to (e.g. SELECT * across many wide string columns,
        # then .to_dict('records') on the full un-aggregated table). This is
        # checked before the extract/analyze branches below regardless of
        # which action the thinker picked, because it has been observed
        # choosing 'extract' and inventing a plausible-sounding schema fix
        # (e.g. "table is missing STATE/YEAR") for what was actually a crash
        # — the crashed table printed right above the claim clearly had
        # those columns. Don't let a wrong action choice lose this signal.
        if (
            'terminated unexpectedly' in err_str_top
            or 'Failed to write to sandbox' in err_str_top
            or 'Failed to read from sandbox' in err_str_top
        ):
            feedback = (
                "SANDBOX CRASHED. This is a resource/memory problem, not a "
                "schema problem — do not rename or guess at column names. "
                "The likely cause is loading too much data at once: SELECT "
                "only the specific columns the query needs (never SELECT *), "
                "and aggregate or filter down to a small result BEFORE "
                "calling to_dict(), to_dict('records'), or similar on the "
                "DataFrame."
            )
        elif missing_cols_diagnosis:
            # Deterministic override: this specific failure has a known,
            # concrete fix (add the column to SELECT). Don't let the generic
            # 'analyze' branch below reduce it to a vague "Code failed:
            # KeyError... Fix the error." message, and don't let a
            # misclassified action route it to the wrong feedback branch —
            # force it to 'extract' since the fix is in the SQL.
            feedback = missing_cols_diagnosis
            action = 'extract'
        elif action == 'extract':
            err_str = str(last_error) if last_error else ''
            zero_rows_hint = (state.get('execution_result') or {}).get('zero_rows_hint')

            # Column errors come from two places: our own pre-flight
            # validator ("Column(s) not found: [...]. Available columns:
            # [...]") or DuckDB itself if something slipped past it
            # ("Referenced column \"X\" not found ... Candidate bindings:
            # ..."). Check for these FIRST — they also contain the word
            # "not found", so they'd otherwise be misclassified as table
            # errors below and lose the real column list.
            is_column_error = (
                'Column(s) not found' in err_str
                or 'Referenced column' in err_str
            )
            # DuckDB phrases a missing/mis-qualified table two different ways:
            # "does not exist" (bad table name) and "Referenced table ... not
            # found ... Candidate tables: ..." (usually a column wrongly
            # qualified with the "user_data" schema). Both need the same
            # kind of corrective feedback, not the generic fallback.
            is_table_error = (not is_column_error) and err_str and (
                'does not exist' in err_str
                or 'Referenced table' in err_str
                or 'Candidate tables' in err_str
            )

            if is_column_error:
                available_match = re.search(r"Available columns:\s*(\[[^\]]*\])", err_str)
                candidates_match = re.search(r'Candidate bindings:\s*(.+)$', err_str)
                if available_match:
                    cols_hint = f" Columns that actually exist on this table: {available_match.group(1)}."
                elif candidates_match:
                    cols_hint = f" Closest real columns on this table: {candidates_match.group(1)}."
                else:
                    cols_hint = " Check the live schema context for the real column list."
                feedback = (
                    "COLUMN NOT FOUND ERROR. A column referenced in the SQL does not "
                    f"exist on this table.{cols_hint} Rewrite the query using ONLY "
                    "columns confirmed to exist — do not guess a column name from a "
                    "common naming pattern."
                )
            elif is_table_error:
                real_tables = state.get('table_names', [])
                did_you_mean = ""
                if "Did you mean" in err_str:
                    match = re.search(r'Did you mean "([^"]+)"', err_str)
                    if match:
                        suggested_table = match.group(1)
                        # Extract just the table name without schema prefix
                        if '.' in suggested_table:
                            suggested_table = suggested_table.split('.')[-1]
                        did_you_mean = f" Use the table '{suggested_table}' (suggested by database). THIS IS THE CORRECT NAME - USE IT EXACTLY."
                elif "Referenced table \"user_data\"" in err_str:
                    did_you_mean = (
                        " This means a COLUMN was incorrectly qualified with the "
                        "schema name, e.g. 'user_data.CATEGORY'. 'user_data' is a "
                        "schema, not a table — it can only prefix the table in "
                        "FROM/JOIN. Rewrite all column references bare, e.g. "
                        "'CATEGORY' not 'user_data.CATEGORY'."
                    )

                # CRITICAL: Emphasize lowercase requirement
                feedback = f"TABLE NAME ERROR. Registered tables: {real_tables}.{did_you_mean}"
            elif not err_str and zero_rows_hint:
                feedback = zero_rows_hint
            else:
                feedback = "Extract data using SQL. Filter rows, select columns. No aggregation."

        elif action == 'analyze':
            if last_error and 'SyntaxError' in str(last_error):
                feedback = (
                    f"Your last code had a syntax error: {str(last_error)[:150]}. "
                    f"Do NOT include any print() debugging or inspection statements — "
                    f"only code that computes `result` directly."
                )
            elif last_error:
                feedback = f"Code failed: {str(last_error)[:150]}. Fix the error."
            else:
                feedback = "Analyze the DataFrame using pandas."

    logger.info("Thinker Decision: action=%s, reason=%s", action, reason)

    print(f"Thinker Decision: action={action}, reason={reason}, feedback={feedback}")
    print(f"Iteration: {iteration}/{max_iterations}")

    return {
        'current_action': action,
        'thinker_reason': reason,
        'analyst_feedback': feedback,
        'iteration_count': iteration + 1
    }

def _sanitize_python_code(code: str) -> str:
    """Strip markdown fences and import statements. No content-altering
    regex substitutions — those corrupt already-valid code."""
    code = code.strip()
    code = re.sub(r'^```(?:python)?\s*', '', code, flags=re.IGNORECASE)
    code = re.sub(r'\s*```$', '', code, flags=re.IGNORECASE)
    code = re.sub(r'^\s*(import|from)\s+\w.*$', '', code, flags=re.MULTILINE)
    return code.strip()


import sqlglot
from sqlglot import exp
import difflib

def _validate_and_fix_columns(sql: str, table_columns: Dict[str, List[str]]) -> tuple[str, Optional[str]]:
    """
    Parse the SQL and check every referenced column against the real,
    live schema. Auto-corrects near-misses (case, minor typos) the same
    way _fix_table_references does for table names. Anything unfixable
    is rejected here, before it ever reaches DuckDB — the model gets the
    exact real column list back, not a raw parser error to misinterpret.
    """
    try:
        parsed = sqlglot.parse_one(sql, read='duckdb')
    except Exception:
        return sql, None  # unparseable — let DuckDB's own error surface normally

    table_nodes = list(parsed.find_all(exp.Table))
    tables = [t.name.lower() for t in table_nodes]
    aliases = {t.alias.lower() for t in table_nodes if t.alias}
    valid_qualifiers = set(tables) | aliases

    available = set()
    for full_name, cols in table_columns.items():
        if full_name.split('.')[-1].lower() in tables:
            available |= set(cols)

    if not available:
        return sql, None  # unknown table — table-name fix path handles this

    available_lower = {c.lower(): c for c in available}
    missing = []
    for col in parsed.find_all(exp.Column):
        # A column qualified with something that isn't a real table name or
        # alias (most commonly the schema "user_data") is malformed — DuckDB
        # rejects it as "Referenced table 'user_data' not found" because
        # user_data is a schema, not something a column can be qualified
        # with. Strip the bogus qualifier here rather than let it reach
        # DuckDB; the bare column name is what actually needs validating.
        qualifier = col.table
        if qualifier and qualifier.lower() not in valid_qualifiers:
            col.set('table', None)

        name = col.name
        if name.lower() in available_lower:
            if name != available_lower[name.lower()]:
                col.set('this', exp.to_identifier(available_lower[name.lower()]))
            continue
        close = difflib.get_close_matches(name.lower(), available_lower.keys(), n=1, cutoff=0.75)
        if close:
            col.set('this', exp.to_identifier(available_lower[close[0]]))
        else:
            missing.append(name)

    if missing:
        return sql, f"Column(s) not found: {missing}. Available columns: {sorted(available)}"

    return parsed.sql(dialect='duckdb'), None


def _validate_no_aggregation(sql: str) -> Optional[str]:
    """
    The rulebase requires SQL to be extraction-only (filter + select) —
    all aggregation happens in the Python phase. The model has been
    observed adding GROUP BY on retry when told to fix a Python KeyError
    (reaching for aggregation instead of the actual fix: adding a missing
    column to SELECT). Reject that deterministically here, before it hits
    DuckDB as a Binder Error and burns another iteration, with a message
    that points at the real fix.
    """
    try:
        parsed = sqlglot.parse_one(sql, read='duckdb')
    except Exception:
        return None  # unparseable — let DuckDB's own error surface normally

    if parsed.find(exp.Group):
        return (
            "SQL contains GROUP BY, which is not allowed — this phase is extraction "
            "only (filter + select). Remove GROUP BY entirely. If a KeyError sent you "
            "here, the real fix is almost always to add the missing column(s) to your "
            "SELECT list, not to aggregate in SQL. All grouping/aggregation belongs in "
            "the Python phase using df.groupby(...)."
        )

    agg_funcs = list(parsed.find_all(exp.AggFunc))
    if agg_funcs:
        names = sorted({type(f).__name__ for f in agg_funcs})
        return (
            f"SQL contains aggregate function(s) {names}, which is not allowed — this "
            "phase is extraction only (filter + select). Move aggregation to the "
            "Python phase using pandas."
        )

    return None


def _build_zero_row_hint(sql: str, data_profile: Dict[str, Any]) -> str:
    """
    The SQL ran fine but matched no rows — almost always a wrong filter
    *value* on a real column, not a wrong column/table (those are caught
    separately). Rather than let the model guess again from the same
    schema block it already ignored once, pull the actual sample values
    for whatever column(s) it filtered on and hand them back directly,
    the same way _validate_and_fix_columns hands back the real column
    list on a column-not-found error.
    """
    generic = (
        "ZERO ROWS RETURNED. The query ran without error but matched no data. "
        "Re-check the live sample values for the filtered column(s) in the schema "
        "context and use one of those exact values, or broaden with ILIKE."
    )
    try:
        parsed = sqlglot.parse_one(sql, read='duckdb')
    except Exception:
        return generic

    table_nodes = list(parsed.find_all(exp.Table))
    table_name = None
    for t in table_nodes:
        candidate = t.name.lower()
        if candidate in data_profile:
            table_name = candidate
            break
    if not table_name:
        return generic

    profile_cols = {
        c['name']: c.get('sample_values', [])
        for c in data_profile[table_name].get('columns', [])
    }

    where = parsed.find(exp.Where)
    if not where:
        return generic

    filtered_cols = []
    for comparison in where.find_all((exp.EQ, exp.In, exp.Like, exp.ILike)):
        col = comparison.find(exp.Column)
        if col and col.name not in filtered_cols:
            filtered_cols.append(col.name)

    hints = [
        f"{col} real values seen in this table: {profile_cols[col]}"
        for col in filtered_cols
        if col in profile_cols and profile_cols[col]
    ]
    if not hints:
        return generic

    return (
        "ZERO ROWS RETURNED for your filter. " + "; ".join(hints) + ". "
        "Use one of these exact values instead of the one you just tried, "
        "or broaden the filter with ILIKE if none match closely."
    )

def _auto_close_brackets(code: str) -> str:
    """Append closing )/]/} for anything left open at EOF. Only appends —
    never removes/reorders — so it can't corrupt valid code."""
    import tokenize, io
    pairs = {'(': ')', '[': ']', '{': '}'}
    closers = {')': '(', ']': '[', '}': '{'}
    stack = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(code).readline):
            if tok.type == tokenize.OP:
                if tok.string in pairs:
                    stack.append(tok.string)
                elif tok.string in closers and stack and stack[-1] == closers[tok.string]:
                    stack.pop()
    except (tokenize.TokenError, IndentationError):
        pass
    if not stack:
        return code
    return code.rstrip() + ''.join(pairs[c] for c in reversed(stack)) + '\n'


def _flatten_indentation(code: str) -> str:
    """Strip leading whitespace from every line.

    Phase 2 analyst code is always a flat sequence of top-level pandas
    statements — the rulebase never asks for if/for/def blocks — so any
    indentation the model adds outside a bracket is spurious. That's
    exactly what produces 'unexpected indent' / IndentationError: a line
    has more indentation than the (nonexistent) enclosing block expects.
    Indentation inside an open ( [ { is unaffected by this either way,
    since Python's tokenizer ignores it there regardless.

    Caveat: if the code contains a multi-line triple-quoted string, this
    also strips leading whitespace from lines inside it, altering that
    string's content. That's an acceptable tradeoff for a last-resort
    repair — it only runs after the code has already failed to compile
    and bracket-repair hasn't fixed it, and it trades a possible content
    change for a working script over a guaranteed failure.
    """
    return '\n'.join(line.lstrip() for line in code.split('\n'))


def _validate_python(code: str) -> tuple[str, str | None]:
    """Returns (usable_code, error_or_None). If error is set, do NOT execute.

    Tries increasingly aggressive, non-destructive repairs before giving up:
    1. Compile as-is.
    2. Auto-close brackets left open at EOF.
    3. On top of that, flatten indentation (see _flatten_indentation) —
       covers 'unexpected indent' from stray leading whitespace.
    Returns the first candidate that compiles; if none do, returns the
    original code alongside the error from the last (most-repaired) attempt.
    """
    candidates = [code]

    bracket_fixed = _auto_close_brackets(code)
    if bracket_fixed != code:
        candidates.append(bracket_fixed)

    flattened = _flatten_indentation(bracket_fixed)
    if flattened not in candidates:
        candidates.append(flattened)

    last_error = None
    for candidate in candidates:
        try:
            compile(candidate, '<analyst>', 'exec')
            return candidate, None
        except SyntaxError as e:
            last_error = e
            continue

    return code, str(last_error)


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

    # Computed early (not just post-generation) so it can also feed the
    # category-group hint below — both uses are pure functions of the live
    # schema, not of anything the model just generated.
    data_profile = state.get('data_profile') or {}
    table_columns = {
        t: [c['name'] for c in data_profile[t].get('columns', [])]
        for t in table_names if t in data_profile
    }

    # Build Context
    context_parts = []
    if state.get('analyst_feedback'):
        context_parts.append(f"FIX THIS FIRST: {state['analyst_feedback']}")

        # CRITICAL: without seeing what it actually wrote last time, the
        # model has nothing to diverge from. analyst_generate() is a
        # stateless call (system_prompt + this context string only — no
        # conversation history), so if we don't echo the previous attempt
        # back here, the model reconstructs a fresh query from the same
        # schema block + feedback description each retry and frequently
        # regenerates the exact same broken SQL/Python. Showing the prior
        # code explicitly gives it a concrete thing to change.
        prev_code = state.get('generated_code')
        if isinstance(prev_code, dict):
            if prev_code.get('sql'):
                context_parts.append(
                    "YOUR PREVIOUS SQL (this is what caused the error above — "
                    f"do not repeat it verbatim, write corrected SQL):\n{prev_code['sql']}"
                )
            if prev_code.get('python'):
                context_parts.append(
                    "YOUR PREVIOUS PYTHON (this is what caused the error above — "
                    f"do not repeat it verbatim, write corrected code):\n{prev_code['python']}"
                )

    context_parts += [
        table_manifest,
        quoting_rule,
    ]
    if state.get('schema_summary'):
        context_parts.append(f"Schema:\n{state['schema_summary']}")

    # Dynamic, per-query reminder placed immediately before the query text —
    # see _category_group_hint docstring for why this isn't left to the
    # static COLUMN GLOSSARY alone.
    category_hint = _category_group_hint(state['query'], table_columns)
    if category_hint:
        context_parts.append(category_hint)

    context_parts.append(f"Query: {state['query']}")

    context = "\n".join(context_parts)
    logger.info("=== ANALYST PROMPT CONTEXT ===\n%s", context[:1000])

    # CALL ANALYST GENERATE (Two-Phase Schema)
    try:
        rulebase = get_rulebase_section('dataset_manifest', 'analyst_system_prompt')
        result = model_client.analyst_generate(rulebase, context)
        logger.info("=== ANALYST RAW OUTPUT ===\n%s", result)
    except Exception as e:
        logger.error("Analyst LLM call failed: %s", e)
        return {
            'errors': state.get('errors', []) + [f'LLM error: {e}'],
            'findings': 'Code generation failed.',
            'success': False,
        }

    sql = result.get('sql', '')
    python_code = result.get('python', '')

    print(f"sql: {sql}")
    print(f"python_code: {python_code}")

    # Fallback parsing if needed
    if not sql and not python_code:
        fallback = _extract_json_fields_regex(str(result), ['sql', 'python'])
        if fallback:
            sql = fallback.get('sql', '')
            python_code = fallback.get('python', '')

    sql = _fix_table_references(sql, table_names)
    sql = re.sub(r'`([^`]+)`', r'"\1"', sql)
    for col in digit_cols:
        sql = re.sub(rf'(?<!")\b{re.escape(col)}\b(?!")', f'"{col}"', sql)

    # NEW: validate every referenced column against the live schema.
    # (table_columns and data_profile were already computed above, before
    # the LLM call, so the category hint could use them too.)
    sql, column_error = _validate_and_fix_columns(sql, table_columns)
    if column_error:
        logger.warning("Column validation failed: %s", column_error)
        return {
            'generated_code': {'sql': sql, 'python': python_code},
            'execution_result': {'error': column_error},
            'findings': f"SQL references invalid column(s): {column_error}",
            'errors': state.get('errors', []) + [column_error],
            'success': False,
        }

    # NEW: reject SQL that violates the extraction-only rule (GROUP BY /
    # aggregate functions) — see _validate_no_aggregation for why.
    aggregation_error = _validate_no_aggregation(sql)
    if aggregation_error:
        logger.warning("SQL aggregation check failed: %s", aggregation_error)
        return {
            'generated_code': {'sql': sql, 'python': python_code},
            'execution_result': {'error': aggregation_error},
            'findings': f"SQL violates extraction-only rule: {aggregation_error}",
            'errors': state.get('errors', []) + [aggregation_error],
            'success': False,
        }

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
        }

    try:
        # PHASE 1: SQL Extraction via DataLayer
        df = None
        if sql.strip():
            logger.info("Executing SQL Phase...")
            try:
                df = data_layer.conn.execute(sql).fetchdf()
                print(f"df: {df}")
            except Exception as sql_err:
                corrected_sql = _autocorrect_from_duckdb_error(sql, str(sql_err))
                if corrected_sql:
                    logger.info("Auto-correcting table name: %s -> %s", sql[:80], corrected_sql[:80])
                    sql = corrected_sql
                    df = data_layer.conn.execute(sql).fetchdf()
                else:
                    raise
            execution_result['df_shape'] = df.shape
            logger.info("SQL Phase Complete. Rows: %d", len(df))
            if len(df) == 0:
                execution_result['zero_rows_hint'] = _build_zero_row_hint(sql, data_profile)
                logger.warning("SQL matched 0 rows: %s", execution_result['zero_rows_hint'])
        else:
            # Empty DF if no SQL
            df = pd.DataFrame()
            logger.info("No SQL provided. Using empty DataFrame.")

        # PHASE 2: Python Analysis in Sandbox with 'df' injected
        if python_code.strip():
            logger.info("Executing Python Phase...")

            # Strip markdown code fences from LLM output (common issue)
            # First strip leading/trailing whitespace, then remove fences
            python_code_stripped = python_code.strip()
            python_code_cleaned = re.sub(r'^```(?:python)?\s*', '', python_code_stripped, flags=re.IGNORECASE)
            python_code_cleaned = re.sub(r'\s*```$', '', python_code_cleaned, flags=re.IGNORECASE)
            python_code_cleaned = python_code_cleaned.strip()

            # CRITICAL: Fix unterminated string literals and balance all brackets/quotes
            python_code_cleaned = _sanitize_python_code(python_code)
            python_code_cleaned, syntax_error = _validate_python(python_code_cleaned)

            if syntax_error:
                logger.warning("Analyst Python failed to compile: %s", syntax_error)
                execution_result['error'] = f'SyntaxError: {syntax_error}'
                findings = f"Generated Python failed to parse: {syntax_error}"
                return {
                    'generated_code': {'sql': sql, 'python': python_code},
                    'execution_result': execution_result,
                    'findings': findings,
                    'errors': state.get('errors', []) + [execution_result['error']],
                    'success': False,
                }

            python_code_no_imports = python_code_cleaned

            if python_code_no_imports != python_code_cleaned:
                logger.info("Stripped import statements and/or fixed syntax errors from LLM code")

            # Serialize df for safe injection into sandbox as a plain Python
            # literal — NOT via `import json`. The sandbox execs code with
            # restricted builtins (no __import__ — see persistent_repl.py),
            # so ANY `import` statement fails with
            # "ImportError: __import__ not found", including one we inject
            # ourselves. That was happening on every single Phase 2 run,
            # independent of whatever code the model generated: the bootstrap
            # always used to start with `import json as _json`.
            # repr() of a list of dicts is valid Python source by itself —
            # no import or parsing call needed to reconstruct it at exec time.
            df_safe = df.copy()
            for col in df_safe.columns:
                if pd.api.types.is_datetime64_any_dtype(df_safe[col]):
                    df_safe[col] = df_safe[col].astype(str)
            # Replace NaN/NaT with None so repr() emits valid Python literals.
            # repr(float('nan')) is the bare word `nan`, which is not valid
            # Python syntax on its own and would raise NameError at exec time.
            df_safe = df_safe.astype(object).where(pd.notnull(df_safe), None)
            df_records = df_safe.to_dict(orient='records')
            bootstrap_code = f"df = pd.DataFrame({df_records!r})\n"
            # Add schema inspection helper to show columns immediately
            schema_check = "print('COLUMNS:', df.columns.tolist())\n"
            full_code = bootstrap_code + schema_check + "\n" + python_code_no_imports
            logger.debug("Full code sent to sandbox (first 500 chars): %s", full_code[:500])
            sandbox_result = sandbox.execute(full_code, timeout=30)
            execution_result.update(sandbox_result)

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
        }

    print(f"Execution Result: {execution_result}")
    print(f"Findings: {findings}")
    return {
        'generated_code': {'sql': sql, 'python': python_code},
        'code_language': 'python',
        'execution_result': execution_result,
        'findings': findings,
        'success': success,
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
        for col in profile['columns']:
            summary_parts.append(f"    - {col['name']} ({col['type']}): {col['distinct_count']} distinct")
            if col.get('sample_values'):
                summary_parts.append(f"      Values: {', '.join(col['sample_values'])}")

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

    rulebase = get_rulebase_section('viz_system_prompt')
    code_result = model_client.generate_code(rulebase, task_description, data_context)
    logger.info("Viz model response: %s", code_result)

    code = code_result.get('code', '')
    language = code_result.get('language', 'python')

    if not code:
        logger.warning("Viz code generation returned empty code")
        return {'errors': state.get('errors', []) + ['Failed to generate visualization code'], 'generated_code': None}

    # Strip markdown code fences from LLM output (common issue)
    # First strip leading/trailing whitespace, then remove fences
    code_stripped = code.strip()
    code_cleaned = re.sub(r'^```(?:python)?\s*', '', code_stripped, flags=re.IGNORECASE)
    code_cleaned = re.sub(r'\s*```$', '', code_cleaned, flags=re.IGNORECASE)
    code_cleaned = code_cleaned.strip()

    logger.info("Generated viz code (cleaned):\n%s", code_cleaned[:500])

    execution_result = sandbox.execute(code_cleaned, timeout=30)
    new_artifacts = execution_result.get('artifacts', [])

    if execution_result.get('error'):
        logger.warning("Viz execution failed: %s", execution_result.get('error'))
    else:
        logger.info("Viz execution succeeded, artifacts: %s", new_artifacts)

    print(f"Viz Execution Result: {execution_result}")
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
        rulebase = get_rulebase_section('report_system_prompt')
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

def _autocorrect_from_duckdb_error(sql: str, error_msg: str) -> Optional[str]:
    """DuckDB's own 'Did you mean' is authoritative for table-not-found errors —
    apply it directly rather than relying on the model to read a manifest it
    has proven it doesn't reliably follow."""
    m = re.search(r'Did you mean "([^"]+)"', error_msg)
    if not m:
        return None
    suggested = m.group(1).split('.')[-1]
    corrected = re.sub(
        r'\b(FROM|JOIN)\s+[a-zA-Z_][a-zA-Z0-9_.-]*',
        lambda mm: f'{mm.group(1)} user_data.{suggested}',
        sql, count=1, flags=re.IGNORECASE,
    )
    return corrected if corrected != sql else None

def _fix_table_references(sql: str, table_names: List[str]) -> str:
    """Rewrite hallucinated/malformed table names to the real registered table.
    Small local models frequently ignore explicit naming instructions —
    correct deterministically rather than re-prompting.

    This function handles multiple variations:
    - Vehicle_Registrations → user_data.vehicle_registrations
    - vehicle-registrations → user_data.vehicle_registrations
    - vehicleregistrations → user_data.vehicle_registrations
    - Any case/separator variation that normalizes to the same key

    CRITICAL: Also handles cases where the model omits the schema prefix
    or uses incorrect casing like "Vehicle_Registrations" instead of
    "user_data.vehicle_registrations".
    """
    if not table_names:
        return sql

    # Build lookup: normalized name (no separators, lowercase) → actual table name
    norm_lookup = {re.sub(r'[\s_-]', '', t).lower(): t for t in table_names}

    def replace_ref(m):
        keyword, raw_name = m.group(1), m.group(2)
        # Strip schema prefix and quotes/backticks
        clean = raw_name.split('.')[-1].strip('"\'`')
        # Normalize: lowercase, remove ALL separators (underscore, hyphen, space)
        key = re.sub(r'[\s_-]', '', clean).lower()
        if key in norm_lookup:
            return f'{keyword} user_data.{norm_lookup[key]}'
        return m.group(0)

    # Match FROM or JOIN followed by table name (with optional schema prefix, allowing hyphens)
    sql = re.sub(r'\b(FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_.-]*)', replace_ref, sql, flags=re.IGNORECASE)

    return sql