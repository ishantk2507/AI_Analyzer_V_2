"""
LLM client using llama-cpp-python for in-process model inference.

Architecture:
- Conversational generation (free-form text): Uses create_chat_completion() WITHOUT grammars
- Structured generation (JSON/code): Uses create_completion() WITH grammars + manual prompt formatting

This split avoids the Windows-native crash in create_chat_completion(grammar=...) while preserving
grammar-constrained decoding reliability for structured outputs.
"""

import json
import logging
import sys
import threading
import re
from typing import Optional, Dict, Any, List
from pathlib import Path

from app.config import (
    MODEL_PATH,
    MODEL_N_CTX,
    MODEL_N_GPU_LAYERS,
)

logger = logging.getLogger(__name__)


# GBNF grammars for constrained decoding
# Supervisor grammar: only routing decisions
GRAMMAR_SUPERVISOR_ACTION = r"""
root ::= action
action ::= "fetch_data" | "analyze" | "visualize" | "report" | "done"
"""

GRAMMAR_SUPERVISOR_OUTPUT = r"""
root ::= thought
thought ::= "{" ws "\"next\"" ws ":" ws next_value ws "," ws "\"reason\"" ws ":" ws string ws "}"
next_value ::= "\"fetch_data\"" | "\"analyze\"" | "\"visualize\"" | "\"report\"" | "\"done\""
string ::= "\"" (char)* "\""
char ::= [^"\\] | "\\" ["\\/bfnrt] | "\\u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]
ws ::= [ \t\n]*
"""

# Code generation grammar (unchanged)
GRAMMAR_CODE_BLOCK = r"""
root ::= code
code ::= "{" ws "\"code\"" ws ":" ws string ws "," ws "\"language\"" ws ":" ws language ws "}"
string ::= "\"" (char)* "\""
language ::= "\"python\"" | "\"sql\""
char ::= [^"\\\n] | "\\" ["\\/bfnrt] | "\\n" | "\\u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]
ws ::= [ \t\n]*
"""


def _clean_gbnf(grammar: str) -> str:
    """
    Clean GBNF grammar for cross-platform compatibility.
    
    - Converts CRLF to LF (Windows line endings break the parser)
    - Removes empty-string literals ("" causes parse errors)
    """
    grammar = grammar.replace('\r\n', '\n').replace('\r', '\n')
    grammar = re.sub(r'::=\s*""', '::= ""', grammar)  # Normalize empty strings
    return grammar


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
            logger.debug("Generated %d tokens", len(content.split()))
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
                cleaned_grammar = _clean_gbnf(grammar)
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
            logger.debug("Raw response: %s", raw_response[:200])

            # Parse JSON from response
            try:
                json_match = re.search(r'\{[^{}]*\}', raw_response, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    logger.debug("Parsed JSON successfully")
                    return result
                return json.loads(raw_response)
            except json.JSONDecodeError as e:
                logger.error("Failed to parse JSON: %s, raw: %s", e, raw_response)
                return {'error': f'Failed to parse JSON: {e}', 'raw': raw_response}
        except Exception as e:
            logger.error("generate_structured failed: %s", e)
            return {'error': f'Generation failed: {e}'}

    def supervisor_step(
        self,
        system_prompt: str,
        context: str,
        user_query: str,
    ) -> Dict[str, str]:
        """
        Supervisor decision: choose next worker node.

        Returns dict with 'next' and 'reason' keys.
        next ∈ {fetch_data, analyze, visualize, report, done}
        """
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': f"Context:\n{context}\n\nQuery: {user_query}\n\nWhich specialist should run next?"}
        ]

        return self.generate_structured(
            messages=messages,
            output_schema='{"next": "fetch_data|analyze|visualize|report|done", "reason": "string"}',
            grammar=GRAMMAR_SUPERVISOR_OUTPUT,
        )

    def generate_code(
        self,
        system_prompt: str,
        task_description: str,
        data_context: str,
    ) -> Dict[str, str]:
        """
        Generate code for execution (Analyst/Viz nodes).

        Returns dict with 'code' and 'language' keys.
        """
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': f"Data context:\n{data_context}\n\nTask: {task_description}\n\nGenerate code to accomplish this."}
        ]

        return self.generate_structured(
            messages=messages,
            output_schema='{"code": "string", "language": "python|sql"}',
            grammar=GRAMMAR_CODE_BLOCK,
        )

    def generate_response(
        self,
        system_prompt: str,
        query: str,
        findings: str,
        interpretation: str,
    ) -> str:
        """
        Generate final response to user (Report node).
        """
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': f"Original query: {query}\n\nFindings:\n{findings}\n\nBusiness interpretation:\n{interpretation}\n\nGenerate a clear, concise response with follow-up questions."}
        ]

        return self.generate(
            messages=messages,
            max_tokens=400,
            temperature=0.5,
        )