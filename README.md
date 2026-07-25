# Local Agentic Data Analysis System

A fully offline, local data analysis agent that answers natural language questions about your datasets. The agent explores data, reasons about next steps, executes code in an isolated Docker sandbox, verifies its results, and responds with findings plus business-relevant interpretations.

## Features

- **Zero internet dependency**: All models, packages, and assets resolved from local disk
- **Isolated code execution**: Agent code runs in a Docker container with `--network none`, memory/CPU limits, and restricted filesystem access
- **Resource-efficient**: Designed for laptops with 6GB VRAM; uses quantized 3B model with full GPU offloading
- **Grammar-constrained decoding**: GBNF grammars ensure structured outputs (action selection, code generation) are always valid
- **Rule-driven behavior**: Agent behavior governed entirely by `rulebase.md` — change rules without editing code

## Prerequisites

1. **Docker** installed and running
2. **Python 3.10+** with pip
3. **Model file**: Download `ministral-3b-instruct-2125-q4_k_m.gguf` and place at `/models/ministral-3b-instruct-2125-q4_k_m.gguf` (or configure via `MODEL_PATH` env var)

## Setup

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Build the sandbox Docker image

```bash
cd docker
docker build -t local-data-sandbox:latest -f Dockerfile.sandbox .
```

### 3. Download the model

Using Ollama or direct download:

```bash
# Option A: Via Ollama (then export to GGUF)
ollama pull ministral:3b-instruct-2125-q4_k_m

# Option B: Direct download from HuggingFace (do this on a machine with internet)
# Download ministral-3b-instruct-2125-q4_k_m.gguf to /models/
```

Set environment variable if model is elsewhere:

```bash
export MODEL_PATH=/path/to/your/ministral-3b-instruct-2125-q4_k_m.gguf
```

### 4. Create rulebase.md

The `rulebase.md` file should be in the project root. This file contains all agent behavior rules and is loaded as the system prompt. See the architecture spec for expected sections (§5 verification checklist, §6 visualization rules, §7 response format).

## Running

Start the Streamlit UI:

```bash
streamlit run app/main.py --server.address localhost --server.port 8501
```

Open http://localhost:8501 in your browser.

## Usage

1. **Upload a dataset**: Drag & drop CSV, Parquet, or XLSX files via the sidebar
2. **Ask questions**: Type natural language queries like:
   - "What are the top 5 categories by revenue?"
   - "Show me the distribution of customer ages"
   - "Is there a correlation between price and quantity sold?"
3. **Review results**: The agent will explore, analyze, verify, and respond with findings and optional charts

## Configuration

Environment variables (all optional):

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `/models/ministral-3b-instruct-2125-q4_k_m.gguf` | Path to GGUF model file |
| `MODEL_N_CTX` | `4096` | Context window size (tokens) |
| `MODEL_N_GPU_LAYERS` | `-1` | GPU layers to offload (-1 = all) |
| `DUCKDB_MEMORY_LIMIT` | `2GB` | DuckDB memory limit |
| `SANDBOX_MEMORY` | `1g` | Sandbox container memory limit |
| `SANDBOX_CPUS` | `1.5` | Sandbox CPU cores |
| `AGENT_MAX_ITERATIONS` | `10` | Max think/act cycles per query |

## Architecture

```
Streamlit UI (app/main.py)
    │
    ▼
LangGraph Agent (app/agent/graph.py)
    ├─ explore_node → profiles dataset via DuckDB
    ├─ think_node   → llama-cpp-python with GBNF grammar
    ├─ act_node     → sends code to sandbox
    ├─ observe_node → parses sandbox JSON response
    ├─ verify_node  → quality checklist, routes on pass/fail
    ├─ viz_decision_node → chart decision rules
    └─ respond_node → formats final answer
    │
    ▼
Sandbox Container (docker exec -i)
    --network none, --memory 1g, --cpus 1.5
    /data (ro) → mounted dataset
    /scratch (rw) → generated charts
```

## File Structure

```
/project-root
  rulebase.md              # Agent behavior rules (system prompt)
  docker/
    Dockerfile.sandbox     # Sandbox container definition
    persistent_repl.py     # In-container REPL for code execution
  app/
    main.py                # Streamlit entrypoint
    agent/
      graph.py             # LangGraph state machine
      nodes.py             # Node implementations (explore/think/act/etc.)
      model_client.py      # llama-cpp-python wrapper with GBNF grammars
      sandbox_client.py    # Docker exec wrapper, JSON-lines protocol
      data_layer.py        # DuckDB connection and query interface
    config.py              # Configuration constants
  requirements.txt
  README.md
```

## Security Notes

- Code execution is isolated in a Docker container with no network access
- Container has read-only access to dataset, read-write only to `/scratch`
- Dangerous builtins (`subprocess`, `socket`, `os.system`) are blocked in the REPL
- Per-call timeouts prevent runaway execution

## Troubleshooting

**Model not found error**: Ensure the GGUF file exists at the configured path. Set `MODEL_PATH` env var if needed.

**Docker permission denied**: Run `docker ps` to verify Docker daemon is accessible. On Linux, you may need to add your user to the `docker` group.

**Out of memory**: Reduce `MODEL_N_GPU_LAYERS` to offload fewer layers to GPU, or reduce `SANDBOX_MEMORY`.

**Slow responses**: The 3B model at Q4 should be fast on most GPUs. If using CPU-only, expect slower inference. Consider a smaller quantization or more GPU layers.

## License

MIT