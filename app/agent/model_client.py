"""
LLM client using llama-cpp-python for in-process model inference.

Architecture for 2-Agent Thinker-Analyst system:
- Analyst: Two-phase generation (SQL extraction + Python analysis) using structured JSON
- Thinker: Validates results and routes to next action (extract|analyze|visualize|report|done)

Platform handling:
- Windows: Uses outlines backend or prompt-only fallback (avoids LlamaGrammar segfaults)
- Linux/Mac: Uses LlamaGrammar with sanitized GBNF for constrained decoding
"""

import json
import logging
import sys
import threading
import re
from typing import Optional, Dict, Any, List, Literal
from pathlib import Path

from app.config import (
    MODEL_PATH,
    MODEL_N_CTX,
    MODEL_N_GPU_LAYERS,
)

logger = logging.getLogger(__name__)


# GBNF grammars for llama.cpp backend (Linux/Mac only)
GRAMMAR_ANALYST = r'''
root ::= output
output ::= "{" ws "\"sql\"" ws ":" ws oneline_string ws "," ws "\"python\"" ws ":" ws string ws "," ws "\"language\"" ws ":" ws "\"python\"" ws "}"
oneline_string ::= "\"" oneline_char* "\""
oneline_char ::= [a-zA-Z0-9 .,;:='*()_<>!@#$%^&+/-] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4})
string ::= "\"" char* "\""
char ::= [^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4})
ws ::= [ \t\n]*
'''

# FIX A: Simplified Thinker Grammar - No feedback field, restricted reason chars
GRAMMAR_THINKER = r'''
root ::= decision
decision ::= "{" ws "\"action\"" ws ":" ws action_val ws "," ws "\"reason\"" ws ":" ws short_string ws "}"
action_val ::= "\"extract\"" | "\"analyze\"" | "\"visualize\"" | "\"report\"" | "\"done\""
short_string ::= "\"" short_char* "\""
short_char ::= [a-zA-Z0-9 .,;:!?()-] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4})
ws ::= [ \t\n]*
'''


GRAMMAR_CODE_BLOCK = r'''
root ::= output
output ::= "{" ws "\"code\"" ws ":" ws string ws "," ws "\"language\"" ws ":" ws language_val ws "}"
language_val ::= "\"python\"" | "\"sql\""
string ::= "\"" char* "\""
char ::= [^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4})
ws ::= [ \t\n]*
'''

GRAMMAR_THOUGHT_OUTPUT = r'''
root ::= thought
thought ::= "{" ws "\"next\"" ws ":" ws next_val ws "," ws "\"reason\"" ws ":" ws string ws "}"
next_val ::= "\"fetch_data\"" | "\"analyze\"" | "\"visualize\"" | "\"report\"" | "\"done\""
string ::= "\"" char* "\""
char ::= [^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4})
ws ::= [ \t\n]*
'''


def _sanitize_gbnf(g: str) -> str:
    """
    Sanitize GBNF grammar for cross-platform compatibility.

    - Converts CRLF to LF (Windows line endings break the parser)
    - Removes empty-string literals ("" causes parse errors)
    - Removes hex escape sequences
    """
    g = g.replace('\r\n', '\n').replace('\r', '\n')
    lines = [l.strip() for l in g.split('\n') if l.strip()]
    g = '\n'.join(lines)
    g = g.replace('| ""', '').replace('""', '')
    g = re.sub(r'\\x[0-9a-fA-F]{2}', '', g)
    return g.rstrip() + '\n'


def _sanitize_json_string(raw: str) -> str:
    """
    Sanitize JSON string by escaping unescaped quotes and newlines inside string values.

    This fixes common LLM output issues where double quotes inside JSON string values
    are not properly escaped, causing json.loads() to fail.

    Algorithm:
    1. Strip markdown fences
    2. Find outermost { ... } bounds
    3. Use state machine to track string context
    4. Escape literal newlines and unescaped quotes inside strings
    """
    # Strip markdown fences
    raw = re.sub(r'^```json\s*', '', raw.strip(), flags=re.IGNORECASE | re.MULTILINE)
    raw = re.sub(r'^```\s*', '', raw, flags=re.MULTILINE)
    raw = raw.strip()

    # Find JSON object bounds
    start = raw.find('{')
    end = raw.rfind('}')
    if start == -1 or end == -1 or end <= start:
        return raw

    blob = raw[start:end+1]
    result = []
    in_string = False
    escape = False
    i = 0

    # JSON only permits a backslash before these characters. Small local
    # models frequently emit backslash-apostrophe (\') or backslash-backtick
    # (\`) when generating embedded code (Python uses \' in some contexts;
    # backticks show up in ad-hoc markdown-style quoting). Those are not
    # legal JSON escapes and make json.loads() fail with "Invalid \escape".
    # The model didn't mean an escape sequence there — it meant the literal
    # character — so drop the stray backslash and keep the character.
    VALID_ESCAPE_CHARS = set('"\\/bfnrtu')

    while i < len(blob):
        ch = blob[i]

        if escape:
            result.append(ch)
            escape = False
            i += 1
            continue

        if ch == '\\':
            nxt = blob[i + 1] if i + 1 < len(blob) else ''
            if nxt in VALID_ESCAPE_CHARS:
                result.append(ch)
                escape = True
            # else: invalid escape — drop the backslash; the next loop
            # iteration handles `nxt` as a normal (non-escaped) character.
            i += 1
            continue

        if ch == '"':
            if not in_string:
                # Opening delimiter
                in_string = True
                result.append(ch)
                i += 1
                continue

            # In string — check if this is a closing delimiter or unescaped inner quote
            # Look ahead: skip spaces, check if next char is structural
            j = i + 1
            while j < len(blob) and blob[j] in ' \t\r\n':
                j += 1

            if j < len(blob) and blob[j] in ':,}]':
                # Closing delimiter
                in_string = False
                result.append(ch)
            else:
                # Unescaped inner quote — escape it
                result.append('\\"')

            i += 1
            continue

        # Escape literal newlines inside strings
        if ch in '\r\n' and in_string:
            result.append('\\n')
            i += 1
            continue

        result.append(ch)
        i += 1

    return ''.join(result)


def _parse_json_with_newline_fix(raw_text: str) -> Dict[str, Any]:
    """
    Parse JSON from raw text that may contain literal newlines inside string values.

    The model often returns multi-line strings inside JSON values like:
        {"code": "-- SQL here\nSELECT ...", "language": "sql"}

    This function escapes literal newlines inside JSON string delimiters before parsing.

    Algorithm:
    1. First try direct parse (fast path for valid JSON)
    2. If that fails, escape newlines inside JSON string values
    3. Parse the cleaned JSON
    """
    # First try direct parse (fast path)
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    # Escape newlines inside JSON string values
    # We need to find content between quotes and escape newlines there
    result = []
    in_string = False
    i = 0
    while i < len(raw_text):
        char = raw_text[i]

        if char == '"' and (i == 0 or raw_text[i-1] != '\\'):
            # Toggle string state
            in_string = not in_string
            result.append(char)
        elif char == '\n' and in_string:
            # Escape newline inside string
            result.append('\\n')
        elif char == '\r' and in_string:
            # Skip carriage return inside string (already handled by \n)
            pass
        else:
            result.append(char)

        i += 1

    cleaned_text = ''.join(result)
    logger.debug("Cleaned JSON text (escaped newlines): %s", cleaned_text[:200])

    # Try parsing the cleaned text
    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError as e:
        logger.warning("JSON parse failed after newline escaping: %s", e)
        raise


def _extract_json_fields_regex(raw_text: str, field_names: List[str]) -> Optional[Dict[str, str]]:
    """
    Extract JSON fields directly from raw text using regex fallback.

    Used when json.loads() fails but we can still extract key-value pairs.

    Args:
        raw_text: Raw model output
        field_names: List of field names to extract (e.g., ['code', 'language'] or ['next', 'reason'])

    Returns:
        Dict with extracted fields, or None if extraction fails
    """
    result = {}
    for field in field_names:
        # Try multiple patterns for robustness

        # Pattern 1: Standard quoted value - handles escaped quotes inside
        pattern1 = rf'"{re.escape(field)}"\s*:\s*"((?:[^"\\]|\\.)*)"'
        match = re.search(pattern1, raw_text, re.DOTALL | re.IGNORECASE)
        if match:
            value = match.group(1)
            # Unescape common escape sequences
            value = value.replace('\\n', '\n').replace('\\t', '\t').replace('\\\\', '\\').replace('\\"', '"')
            result[field] = value
            logger.debug("Regex (pattern1) extracted %s: %s", field, value[:50])
            continue

        # Pattern 2: Multi-line value until next field or closing brace
        # This catches cases where newlines break the JSON structure
        pattern2 = rf'"{re.escape(field)}"\s*:\s*"([\s\S]*?)"\s*(?:,|\}})'
        match = re.search(pattern2, raw_text, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            result[field] = value
            logger.debug("Regex (pattern2) extracted %s: %s", field, value[:50])
            continue

    return result if result else None


def _format_ministral_prompt(messages: List[Dict[str, str]]) -> str:
    """
    Format messages using Ministral/Mistral chat template.

    This is a minimal, reliable implementation that avoids calling
    create_chat_completion() which crashes on Windows with grammars.

    Template format:
    [INST] <<SYS>>
    {system}
    <</SYS>>

    {user} [/INST]
    {assistant}</s><s>[INST]
    """
    formatted = ""
    system_content = ""

    for msg in messages:
        role = msg.get('role', 'user')
        content = msg.get('content', '')

        if role == 'system':
            system_content = content
        elif role == 'user':
            if system_content:
                formatted += f"[INST] <<SYS>>\n{system_content}\n<</SYS>>\n\n{content} [/INST]"
                system_content = ""  # Only apply system once
            else:
                formatted += f"[INST] {content} [/INST]"
        elif role == 'assistant':
            formatted += f"{content}</s><s>[INST] "

    return formatted


class ModelClient:
    """
    Singleton wrapper for llama-cpp-python model.

    Provides two distinct inference paths:
    1. Conversational (generate, generate_response): create_chat_completion() without grammars
    2. Structured (generate_structured, supervisor_step, etc.): create_completion() with grammars

    This split avoids Windows-native crashes while preserving grammar-constrained reliability.
    """

    _instance: Optional['ModelClient'] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_path: Optional[str] = None):
        if self._initialized:
            return

        self.model_path = Path(model_path or MODEL_PATH)
        self.model = None
        self._load_model()
        self._initialized = True

    def _load_model(self):
        """Load the GGUF model with resource-efficient settings."""
        logger.info("Loading model from %s", self.model_path)
        try:
            from llama_cpp import Llama
        except ImportError:
            logger.error("llama-cpp-python not installed")
            raise ImportError(
                "llama-cpp-python not installed. Run: pip install llama-cpp-python"
            )

        if not self.model_path.exists():
            logger.error("Model file not found: %s", self.model_path)
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # Load model with GPU offloading and limited context
        try:
            self.model = Llama(
                model_path=str(self.model_path),
                n_ctx=MODEL_N_CTX,
                n_gpu_layers=MODEL_N_GPU_LAYERS,
                n_threads=None,  # Auto-detect
                verbose=False,
                use_mmap=True,
                use_mlock=False,
            )
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error("Failed to load model: %s", e)
            raise

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        """
        Generate free-form conversational text (NO grammar constraints).

        Uses create_chat_completion() for proper chat template handling.
        DO NOT use for structured outputs—use generate_structured() instead.

        Args:
            messages: Chat messages list.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            stop_sequences: Optional stop strings.

        Returns:
            Generated text string.
        """
        if self.model is None:
            logger.error("Model not loaded")
            raise RuntimeError("Model not loaded")

        try:
            # Conversational path: create_chat_completion WITHOUT grammar
            logger.debug("Calling create_chat_completion with %d messages", len(messages))
            response = self.model.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop_sequences,
            )

            content = response['choices'][0]['message']['content']
            logger.info("Generated conversational response (first 200 chars): %s", content[:200])
            logger.debug("Full conversational response: %s", content)
            return content
        except Exception as e:
            logger.error("create_chat_completion failed: %s", e)
            raise

    def generate_structured(
        self,
        messages: List[Dict[str, str]],
        output_schema: str,
        grammar: str,
        max_tokens: int = 256,
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        """
        Generate structured JSON with grammar-constrained decoding.

        Uses create_completion() with manual prompt formatting to avoid
        Windows-native crash in create_chat_completion(grammar=...).

        On Windows: skips grammar parameter to avoid segfaults.
        On Linux/Mac: uses grammar-constrained decoding for reliability.

        Args:
            messages: Chat messages list.
            output_schema: Description of expected JSON structure.
            grammar: GBNF grammar string for constrained decoding.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (low for determinism).

        Returns:
            Parsed JSON dict.
        """
        if self.model is None:
            logger.error("Model not loaded")
            raise RuntimeError("Model not loaded")

        try:
            # Augment last user message with schema instruction
            augmented_messages = messages.copy()
            last_msg = augmented_messages[-1]
            if last_msg['role'] == 'user':
                augmented_messages[-1] = {
                    'role': 'user',
                    'content': f"{last_msg['content']}\n\nRespond in this exact JSON format:\n{output_schema}"
                }

            # Build prompt using manual template (avoids create_chat_completion crash)
            prompt = _format_ministral_prompt(augmented_messages)
            logger.debug("Generated prompt length: %d chars", len(prompt))

            # Token-budget check: nothing previously measured the actual
            # prompt against n_ctx, so a prompt that ran over the limit
            # (e.g. a large schema_summary plus retry context) would be
            # silently truncated by llama.cpp -- most likely from the
            # front, which is exactly where the system prompt (rulebase
            # instructions) lives. Make that visible instead of invisible.
            try:
                n_ctx = self.model.n_ctx()
                prompt_tokens = self.model.tokenize(prompt.encode('utf-8'), add_bos=True)
                prompt_len = len(prompt_tokens)
                if prompt_len + max_tokens > n_ctx:
                    logger.warning(
                        "Prompt (%d tokens) + max_tokens (%d) exceeds n_ctx (%d) by "
                        "%d tokens -- content will be truncated, most likely from the "
                        "start of the system prompt (the rulebase). Shrink schema_summary "
                        "or the retry context, or raise MODEL_N_CTX.",
                        prompt_len, max_tokens, n_ctx, prompt_len + max_tokens - n_ctx,
                    )
                else:
                    logger.debug("Prompt token budget OK: %d + %d <= %d (n_ctx)",
                                 prompt_len, max_tokens, n_ctx)
            except Exception as e:
                logger.debug("Could not check prompt token budget: %s", e)

            # Platform-specific grammar handling
            from llama_cpp import LlamaGrammar

            if sys.platform == "win32":
                # Windows: skip grammar to avoid DLL segfault
                logger.debug("Windows detected: skipping grammar-constrained decoding")
                response = self.model.create_completion(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=["</s>", "[INST]"],
                )
            else:
                # Linux/Mac: use grammar-constrained decoding
                cleaned_grammar = _sanitize_gbnf(grammar)
                grammar_obj = LlamaGrammar.from_string(cleaned_grammar)
                logger.debug("Calling create_completion with grammar")
                response = self.model.create_completion(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    grammar=grammar_obj,
                    stop=["</s>", "[INST]"],
                )

            raw_response = response['choices'][0]['text'].strip()
            # Log raw model output
            logger.info("Raw structured response (first 500 chars): %s", raw_response[:500])
            logger.debug("Full raw structured response: %s", raw_response)

            # Pre-process raw response: strip markdown fences
            raw_response = re.sub(r'^```json\s*', '', raw_response, flags=re.IGNORECASE)
            raw_response = re.sub(r'^```\s*', '', raw_response, flags=re.IGNORECASE)
            raw_response = re.sub(r'\s*```$', '', raw_response, flags=re.IGNORECASE)
            raw_response = raw_response.strip()

            # Sanitize JSON string (escape unescaped quotes and newlines)
            cleaned_response = _sanitize_json_string(raw_response)

            # Parse JSON from response
            try:
                result = json.loads(cleaned_response)
                logger.info("Parsed JSON successfully: %s", result)
                return result
            except json.JSONDecodeError as e:
                logger.error("Failed to parse JSON: %s, raw: %s", e, raw_response)
                # Fallback: extract fields via regex
                fallback_result = _extract_json_fields_regex(raw_response, ['code', 'language', 'next', 'reason', 'action', 'feedback', 'sql', 'python'])
                if fallback_result:
                    logger.info("Regex fallback succeeded, extracted: %s", fallback_result)
                    return fallback_result
                return {'error': f'Failed to parse JSON: {e}', 'raw': raw_response}
        except Exception as e:
            logger.error("generate_structured failed: %s", e)
            return {'error': f'Generation failed: {e}'}

    def thinker_decide(self, system_prompt: str, context: str) -> Dict[str, Any]:
        """
        Generate Thinker agent decision.
        FIX A: Returns ONLY 'action' and 'reason'. No 'feedback'.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': context}
        ]

        result = self.generate_structured(
            messages=messages,
            output_schema='{"action": "extract|analyze|visualize|report|done", "reason": "one sentence"}',
            grammar=GRAMMAR_THINKER,
            temperature=0.1,
        )
        logger.info("Thinker decision: action=%s, reason=%s", result.get('action'), result.get('reason'))
        return result

    def analyst_generate(self, system_prompt: str, context: str) -> Dict[str, Any]:
        """
        Generate Analyst agent output.
        FIX E: SQL is forced to single line by grammar.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': context}
        ]

        result = self.generate_structured(
            messages=messages,
            output_schema='{"python": "pandas code", "language": "python"}',
            grammar=GRAMMAR_ANALYST,
            max_tokens=1024,   # was defaulting to 256
            temperature=0.2,
        )
        logger.info("Analyst generated: sql=%s, python=%s", result.get('sql', '')[:100], result.get('python', '')[:100])
        return result

    def generate_code(
        self,
        system_prompt: str,
        task_description: str,
        data_context: str,
    ) -> Dict[str, str]:
        """
        Generate code for execution (Analyst/Viz nodes).

        Returns dict with 'code' and 'language' keys.
        Logs raw model response for debugging.
        """
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': f"Data context:\n{data_context}\n\nTask: {task_description}\n\nGenerate code to accomplish this."}
        ]

        result = self.generate_structured(
            messages=messages,
            output_schema='{"code": "string", "language": "python|sql"}',
            grammar=GRAMMAR_CODE_BLOCK,
        )

        # Log the raw model response for debugging
        logger.info("generate_code result: code length=%d, language=%s",
                    len(result.get('code', '')), result.get('language'))
        logger.debug("generate_code full result: %s", result)
        return result

    def generate_response(
        self,
        system_prompt: str,
        query: str,
        findings: str,
        interpretation: str,
    ) -> str:
        """
        Generate final response to user (Report node).
        FIX: Removed instruction to ask follow-up questions.
        """
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': f"Original query: {query}\n\nInterpretation:\n{interpretation}\n\nGenerate a clear, concise response summarizing the findings above. Do not invent new analysis steps."}
        ]

        return self.generate(
            messages=messages,
            max_tokens=400,
            temperature=0.5,
        )