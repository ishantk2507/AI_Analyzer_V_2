"""
Configuration for the local agentic data analysis system.

All magic numbers and tunable parameters live here.
Change this file to adjust resource limits, timeouts, model settings.
"""

import os
from pathlib import Path

# === Model Configuration ===
MODEL_PATH = os.environ.get(
    'MODEL_PATH',
    '/models/ministral-3b-instruct-2125-q4_k_m.gguf'
)
MODEL_N_CTX = int(os.environ.get('MODEL_N_CTX', '4096'))
MODEL_N_GPU_LAYERS = int(os.environ.get('MODEL_N_GPU_LAYERS', '-1'))  # -1 = offload all

# === DuckDB Configuration ===
DUCKDB_MEMORY_LIMIT = os.environ.get('DUCKDB_MEMORY_LIMIT', '2GB')

# === Sandbox Container Configuration ===
SANDBOX_IMAGE_NAME = 'local-data-sandbox:latest'
SANDBOX_MEMORY = os.environ.get('SANDBOX_MEMORY', '1g')
SANDBOX_MEMORY_SWAP = os.environ.get('SANDBOX_MEMORY_SWAP', '1g')
SANDBOX_CPUS = float(os.environ.get('SANDBOX_CPUS', '1.5'))
SANDBOX_TIMEOUT_DEFAULT = int(os.environ.get('SANDBOX_TIMEOUT_DEFAULT', '10'))
SANDBOX_TIMEOUT_MAX = int(os.environ.get('SANDBOX_TIMEOUT_MAX', '60'))

# === LangGraph / Agent Configuration ===
AGENT_MAX_ITERATIONS = int(os.environ.get('AGENT_MAX_ITERATIONS', '10'))
AGENT_CONTEXT_MAX_TOKENS = int(os.environ.get('AGENT_CONTEXT_MAX_TOKENS', '3000'))

# === Paths ===
PROJECT_ROOT = Path(__file__).parent.parent
DOCKER_DIR = PROJECT_ROOT / 'docker'
DATA_MOUNT_DEFAULT = os.environ.get('DATA_MOUNT_DEFAULT', '/tmp/data_mount')
SCRATCH_MOUNT_DEFAULT = os.environ.get('SCRATCH_MOUNT_DEFAULT', '/tmp/scratch_mount')

# === Environment flags for offline operation ===
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['PIP_NO_INDEX'] = '1'
os.environ['PIP_DISABLE_PIP_VERSION_CHECK'] = '1'
os.environ['MPLBACKEND'] = 'Agg'
os.environ['NO_COLOR'] = '1'
