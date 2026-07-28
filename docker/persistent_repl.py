#!/usr/bin/env python3
"""
Persistent REPL for sandbox container.

Reads JSON commands from stdin, executes code in a persistent namespace,
and writes JSON results to stdout. Enforces per-call timeout and blocks
dangerous operations.

Protocol:
  Input (stdin):  {"code": "<python or sql string>", "timeout": <int>}
  Output (stdout): {"stdout": "...", "stderr": "...", "result_repr": "...",
                    "error": null|"<message>", "artifacts": ["/scratch/..."]}
"""

import sys
import json
import threading
import traceback
import re
import os
import logging
from io import StringIO
from types import ModuleType

logger = logging.getLogger(__name__)

# Pre-import safe modules for the execution namespace
import duckdb
import pandas as pd
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
                # Execute as Python code
                exec(code_cleaned, namespace)
                # Get last expression result if any
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
            val = execution_result[0]
            if hasattr(val, 'to_dict'):
                result['result_repr'] = json.dumps(val.to_dict(orient='records') if len(val) <= 100 else val.head(100).to_dict(orient='records'), default=str)
            else:
                result['result_repr'] = repr(val)
        except Exception as e:
            result['result_repr'] = str(val)

    # Scan for generated artifacts (PNG files in /scratch)
    if os.path.exists('/scratch'):
        for fname in os.listdir('/scratch'):
            if fname.endswith('.png'):
                fpath = f'/scratch/{fname}'
                if fpath not in result['artifacts']:
                    result['artifacts'].append(fpath)

    print(result)
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
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()