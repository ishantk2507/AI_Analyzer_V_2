#!/usr/bin/env python3
"""
Persistent REPL for sandbox container.

Reads JSON commands from stdin, executes code in a persistent namespace,
and writes JSON results to stdout. Enforces per-call timeout and blocks
dangerous operations.

Protocol:
  Input (stdin):  {"code": "<python or sql string>", "timeout": <int>}
  Output (stdout): {"stdout": "...", "stderr": "...", "result_repr": "...",
                    "error": null|"<message>", "artifacts": ["/scratch/..."],
                    "result_type": "DataFrame"|"int"|"str"|"NoneType"|...,
                    "result_value": <JSON-safe value, when the result type
                        supports one - scalars, and short lists/dicts, or a
                        bounded preview of records for DataFrame/Series>,
                    "result_shape": [rows, cols]  # DataFrame/ndarray/Series only
                    "result_columns": [...]        # DataFrame only
                    "result_dtypes": {...}         # DataFrame only
                    "result_truncated": true        # present only if the
                        preview was cut short of the full result

  result_repr is always a string (kept for backward compatibility with
  callers that slice/read it as text). The other result_* fields are
  type-aware and are only present when applicable - a plain int result
  will have result_type/result_value/result_repr but no result_shape.
"""

import sys
import json
import threading
import traceback
import re
import os
import ast
import logging
from io import StringIO
from types import ModuleType

logger = logging.getLogger(__name__)

# Pre-import safe modules for the execution namespace
import duckdb
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pyarrow as pa
import json as _json_module  # Pre-import json for use in bootstrap code

# Blocked modules/functions for defense in depth
BLOCKED_MODULES = {'subprocess', 'socket', 'os.system', 'os.popen', 'os.spawn'}
BLOCKED_BUILTINS = {'__import__', 'eval', 'exec', 'compile', 'open'}

class TimeoutError(Exception):
    pass


class SafeNamespace(dict):
    """A restricted namespace that blocks dangerous operations."""

    def __init__(self):
        # Start with safe builtins only
        safe_builtins = {
            '__builtins__': {
                'abs': abs, 'all': all, 'any': any, 'bin': bin, 'bool': bool,
                'bytes': bytes, 'callable': callable, 'chr': chr, 'complex': complex,
                'dict': dict, 'dir': dir, 'divmod': divmod, 'enumerate': enumerate,
                'filter': filter, 'float': float, 'format': format, 'frozenset': frozenset,
                'getattr': getattr, 'hasattr': hasattr, 'hash': hash, 'hex': hex,
                'int': int, 'isinstance': isinstance, 'issubclass': issubclass,
                'iter': iter, 'len': len, 'list': list, 'map': map, 'max': max,
                'min': min, 'next': next, 'object': object, 'oct': oct, 'ord': ord,
                'pow': pow, 'print': print, 'range': range, 'repr': repr,
                'reversed': reversed, 'round': round, 'set': set, 'slice': slice,
                'sorted': sorted, 'str': str, 'sum': sum, 'super': super,
                'tuple': tuple, 'type': type, 'zip': zip,
                'True': True, 'False': False, 'None': None,
                'Exception': Exception, 'ValueError': ValueError, 'TypeError': TypeError,
                'KeyError': KeyError, 'IndexError': IndexError,
            }
        }
        super().__init__(safe_builtins)

        # Add pre-imported safe modules
        self['duckdb'] = duckdb
        self['pd'] = pd
        self['np'] = np
        self['plt'] = plt
        self['pa'] = pa
        self['json'] = _json_module  # Make json available for bootstrap code

        # Create a restricted open() that only allows /data and /scratch
        def safe_open(file, mode='r', *args, **kwargs):
            abs_path = os.path.abspath(file)
            if not (abs_path.startswith('/data/') or
                    abs_path.startswith('/scratch/') or
                    abs_path == '/data' or
                    abs_path == '/scratch'):
                raise PermissionError(f"Access denied: {file}. Only /data and /scratch are allowed.")
            return __builtins__['open'](file, mode, *args, **kwargs)

        self['open'] = safe_open


# Bounds on how much of a result we ever try to carry back across the
# stdin/stdout boundary. These exist because the eventual consumer is an
# LLM prompt with a small token budget (not just a display widget), so an
# uncapped DataFrame or list dump would silently eat the whole context.
MAX_RESULT_CHARS = 4000
MAX_ROWS_PREVIEW = 20
MAX_COLS_PREVIEW = 40
MAX_ARRAY_PREVIEW = 100


def _json_default(o):
    """Fallback encoder for objects json.dumps doesn't natively handle.

    Used everywhere we call json.dumps in this module so that a stray
    numpy scalar, Timestamp, set, or other common pandas/numpy citizen
    never turns into an unhandled exception that kills the REPL process.
    """
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (pd.Timestamp, pd.Timedelta)):
        return str(o)
    if hasattr(o, 'isoformat'):  # datetime.date/datetime, etc.
        return o.isoformat()
    if isinstance(o, (set, frozenset)):
        return list(o)
    if isinstance(o, bytes):
        return o.decode('utf-8', errors='replace')
    return str(o)


def summarize_result(val) -> dict:
    """
    Build a type-aware summary of an execution result instead of blindly
    repr()-ing everything or flattening every DataFrame into raw JSON
    records. Returns a dict of fields to merge into the response - see the
    module docstring for the full field list.

    Design goals:
      - result_repr is ALWAYS a plain string (backward compatible with
        callers that do result_repr[:200] etc.), but it's built per-type
        so a DataFrame reads like a printed table (compact, familiar to an
        LLM) rather than a repeated-keys JSON blob.
      - result_value carries a JSON-safe, still-structured version of the
        value for callers that want to do more than read text, capped in
        size so it can't blow up the response.
      - Nothing in here should ever raise - a completely unknown object
        just falls back to a bounded repr() with its type name attached.
    """
    out: dict = {}

    if val is None:
        out['result_type'] = 'NoneType'
        out['result_repr'] = ''
        return out

    # --- pandas DataFrame ---
    if isinstance(val, pd.DataFrame):
        out['result_type'] = 'DataFrame'
        out['result_shape'] = list(val.shape)
        out['result_columns'] = [str(c) for c in val.columns]
        out['result_dtypes'] = {str(c): str(t) for c, t in val.dtypes.items()}

        row_trunc = val.shape[0] > MAX_ROWS_PREVIEW
        col_trunc = val.shape[1] > MAX_COLS_PREVIEW
        preview_df = val.iloc[:MAX_ROWS_PREVIEW, :MAX_COLS_PREVIEW]

        # Text form: printed table, not JSON - far more token-efficient
        # (column names aren't repeated per row) and reads naturally to an LLM.
        header = f"DataFrame: {val.shape[0]} rows x {val.shape[1]} cols"
        try:
            body = preview_df.to_string(index=True)
        except Exception:
            body = preview_df.astype(str).to_string(index=True)
        text = f"{header}\n{body}"
        if row_trunc or col_trunc:
            text += f"\n... (showing {preview_df.shape[0]} of {val.shape[0]} rows, {preview_df.shape[1]} of {val.shape[1]} cols)"
            out['result_truncated'] = True
        out['result_repr'] = text[:MAX_RESULT_CHARS]

        # Structured form: bounded JSON-safe records
        try:
            out['result_value'] = json.loads(
                preview_df.to_json(orient='records', date_format='iso', default_handler=str)
            )
        except Exception:
            out['result_value'] = preview_df.astype(str).to_dict(orient='records')
        return out

    # --- pandas Series ---
    if isinstance(val, pd.Series):
        out['result_type'] = 'Series'
        out['result_shape'] = [val.shape[0]]
        truncated = val.shape[0] > MAX_ROWS_PREVIEW
        preview = val.iloc[:MAX_ROWS_PREVIEW]
        header = f"Series '{val.name}': len={val.shape[0]}, dtype={val.dtype}"
        try:
            body = preview.to_string()
        except Exception:
            body = str(preview)
        text = f"{header}\n{body}"
        if truncated:
            text += f"\n... (showing first {MAX_ROWS_PREVIEW} of {val.shape[0]})"
            out['result_truncated'] = True
        out['result_repr'] = text[:MAX_RESULT_CHARS]
        try:
            out['result_value'] = json.loads(preview.to_json(date_format='iso', default_handler=str))
        except Exception:
            out['result_value'] = {str(k): str(v) for k, v in preview.items()}
        return out

    # --- numpy array ---
    if isinstance(val, np.ndarray):
        out['result_type'] = 'ndarray'
        out['result_shape'] = list(val.shape)
        flat = val.flatten()
        preview = flat[:MAX_ARRAY_PREVIEW].tolist()
        out['result_value'] = preview
        text = f"ndarray shape={val.shape} dtype={val.dtype}: {preview}"
        if flat.size > MAX_ARRAY_PREVIEW:
            text += f" ... ({flat.size} total elements)"
            out['result_truncated'] = True
        out['result_repr'] = text[:MAX_RESULT_CHARS]
        return out

    # --- numpy scalar types (np.int64, np.float64, np.bool_, ...) ---
    if isinstance(val, np.generic):
        py_val = val.item()
        out['result_type'] = type(val).__name__
        out['result_value'] = py_val
        out['result_repr'] = str(py_val)
        return out

    # --- matplotlib figure/axes: point at the artifact convention, don't try to inline it ---
    if isinstance(val, (plt.Figure, plt.Axes)):
        out['result_type'] = type(val).__name__
        out['result_repr'] = (
            f"<{type(val).__name__}> (not inlined - save it with "
            f"plt.savefig('/scratch/name.png') to return it as an artifact)"
        )
        return out

    # --- native JSON-safe scalars ---
    if isinstance(val, (bool, int, float, str)):
        out['result_type'] = type(val).__name__
        out['result_value'] = val
        out['result_repr'] = str(val)[:MAX_RESULT_CHARS]
        return out

    # --- dict / list / tuple / set: keep structure via JSON rather than repr() ---
    if isinstance(val, (dict, list, tuple, set, frozenset)):
        out['result_type'] = type(val).__name__
        try:
            text = json.dumps(val, default=_json_default)
        except Exception:
            out['result_repr'] = repr(val)[:MAX_RESULT_CHARS]
            return out
        if len(text) > MAX_RESULT_CHARS:
            out['result_truncated'] = True
            out['result_repr'] = text[:MAX_RESULT_CHARS] + ' ...'
            # Still surface a value, just not the full (oversized) structure.
        else:
            out['result_repr'] = text
            try:
                out['result_value'] = json.loads(text)
            except Exception:
                pass
        return out

    # --- fallback: arbitrary object, e.g. a custom class instance ---
    out['result_type'] = type(val).__name__
    try:
        out['result_repr'] = repr(val)[:MAX_RESULT_CHARS]
    except Exception as e:
        out['result_repr'] = f"<unrepresentable {type(val).__name__}: {e}>"
    return out


def _exec_capturing_last_expr(code: str, namespace: dict):
    """
    Execute code the way a notebook cell does: if the final top-level
    statement is a bare expression (e.g. `df.groupby('x').sum()` as the
    last line, with no assignment), evaluate it separately so its value is
    captured automatically - mirroring Jupyter/IPython "last expression is
    the cell's result" semantics.

    Without this, code that doesn't explicitly do `result = ...` on the
    last line silently returns nothing, which is the main reason "the
    result" so often ended up missing or forced into a lossy string.

    Falls back to a plain exec() (returning None) for code that doesn't
    end in a bare expression - e.g. it ends with an assignment, a for
    loop, or a def. Callers should fall back to the `result`/`_` variable
    convention in that case, exactly as before.
    """
    tree = ast.parse(code, mode='exec')

    if not tree.body:
        return None

    last = tree.body[-1]
    if isinstance(last, ast.Expr):
        # Run everything except the last statement normally...
        head = ast.Module(body=tree.body[:-1], type_ignores=[])
        ast.fix_missing_locations(head)
        exec(compile(head, '<sandbox>', 'exec'), namespace)
        # ...then evaluate the last expression to capture its value.
        expr = ast.Expression(body=last.value)
        ast.fix_missing_locations(expr)
        return eval(compile(expr, '<sandbox>', 'eval'), namespace)

    exec(compile(tree, '<sandbox>', 'exec'), namespace)
    return None


def execute_with_timeout(code: str, namespace: SafeNamespace, timeout_sec: int) -> dict:
    """Execute code with a timeout using a watchdog thread."""
    result = {
        'stdout': '',
        'stderr': '',
        'result_repr': '',
        'error': None,
        'artifacts': []
    }

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    stdout_capture = StringIO()
    stderr_capture = StringIO()

    execution_done = threading.Event()
    execution_error = [None]  # Use list to allow modification in nested function
    execution_result = [None]

    def run_code():
        try:
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture

            # Strip markdown code fences if present
            code_cleaned = code.strip()
            code_cleaned = re.sub(r'^```(?:sql|python)?\s*', '', code_cleaned, flags=re.IGNORECASE)
            code_cleaned = re.sub(r'\s*```$', '', code_cleaned, flags=re.IGNORECASE)
            code_cleaned = code_cleaned.strip()

            # Remove SQL comments from the beginning to properly detect SQL
            # Also strip leading/trailing whitespace after comment removal
            code_for_detection = re.sub(r'^\s*--.*$', '', code_cleaned, flags=re.MULTILINE).strip()
            code_for_detection_upper = code_for_detection.upper()

            logger.debug("Code detection: first 100 chars of cleaned='%s', after comment removal='%s'",
                        code_cleaned[:100], code_for_detection[:100])

            # Check for SQL-like query (heuristic: starts with SQL keywords after stripping comments)
            is_sql = bool(code_for_detection_upper) and code_for_detection_upper.startswith(
                ('SELECT', 'WITH', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'ALTER', 'DESCRIBE')
            )

            logger.debug("Code detected as SQL: %s", is_sql)

            if is_sql:
                # Execute as DuckDB SQL
                conn = namespace.get('duckdb_conn')
                if conn is None:
                    conn = duckdb.connect(':memory:')
                    namespace['duckdb_conn'] = conn

                # Execute SQL and capture result
                df_result = conn.execute(code_cleaned).fetchdf()
                execution_result[0] = df_result
            else:
                # Execute as Python code. Try to auto-capture the value of a
                # trailing bare expression (notebook-style); fall back to the
                # explicit `result`/`_` variable convention for code that
                # ends in an assignment, loop, etc.
                captured = _exec_capturing_last_expr(code_cleaned, namespace)
                if captured is not None:
                    execution_result[0] = captured
                else:
                    execution_result[0] = namespace.get('result', namespace.get('_'))

        except Exception as e:
            execution_error[0] = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            execution_done.set()

    # Start execution in a separate thread
    exec_thread = threading.Thread(target=run_code, daemon=True)
    exec_thread.start()

    # Wait for completion or timeout
    if not execution_done.wait(timeout=timeout_sec):
        result['error'] = f"Timeout after {timeout_sec} seconds"
        # Note: The thread will continue running in background but is daemonized
        return result

    # Collect results
    result['stdout'] = stdout_capture.getvalue()
    result['stderr'] = stderr_capture.getvalue()

    if execution_error[0]:
        result['error'] = execution_error[0]
    elif execution_result[0] is not None:
        try:
            result.update(summarize_result(execution_result[0]))
        except Exception as e:
            # summarize_result() is written to never raise, but if some
            # exotic object still manages to break it, degrade gracefully
            # instead of losing the whole response.
            logger.warning("Result summarization failed: %s", e)
            val = execution_result[0]
            result['result_type'] = type(val).__name__
            try:
                result['result_repr'] = str(val)[:MAX_RESULT_CHARS]
            except Exception:
                result['result_repr'] = f"<unrepresentable {type(val).__name__}>"

    # Scan for generated artifacts (PNG files in /scratch)
    if os.path.exists('/scratch'):
        for fname in os.listdir('/scratch'):
            if fname.endswith('.png'):
                fpath = f'/scratch/{fname}'
                if fpath not in result['artifacts']:
                    result['artifacts'].append(fpath)

    return result


def main():
    namespace = SafeNamespace()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            cmd = json.loads(line)
        except json.JSONDecodeError as e:
            error_response = {
                'stdout': '',
                'stderr': '',
                'result_repr': '',
                'error': f"Invalid JSON input: {e}",
                'artifacts': []
            }
            print(json.dumps(error_response), flush=True)
            continue

        code = cmd.get('code', '')
        timeout = cmd.get('timeout', 10)

        # Validate timeout
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            timeout = 10

        result = execute_with_timeout(code, namespace, int(timeout))
        try:
            output_line = json.dumps(result, default=_json_default)
        except Exception as e:
            # Last-resort safety net: never let an odd value stop the REPL
            # from responding, since sandbox_client.py is blocked reading a
            # line and would otherwise have to time out / restart the process.
            logger.error("Failed to serialize result to JSON: %s", e)
            output_line = json.dumps({
                'stdout': result.get('stdout', ''),
                'stderr': result.get('stderr', ''),
                'result_repr': f"<result not serializable: {e}>",
                'result_type': result.get('result_type'),
                'error': result.get('error'),
                'artifacts': result.get('artifacts', []),
            }, default=str)
        print(output_line, flush=True)


if __name__ == '__main__':
    main()