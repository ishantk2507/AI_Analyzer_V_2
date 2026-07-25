"""
LLM client using llama-cpp-python for in-process model inference.

Loads the GGUF model once per session, uses grammar-constrained decoding
for structured outputs (action selection, code generation, verification).
"""

import json
import threading
from typing import Optional, Dict, Any, List
from pathlib import Path

from app.config import (
    MODEL_PATH,
    MODEL_N_CTX,
    MODEL_N_GPU_LAYERS,
)


# GBNF grammars for constrained decoding
GRAMMAR_ACTION_CHOICE = r"""
root ::= action
action ::= "explore" | "think" | "act" | "verify" | "visualize" | "respond"
"""

GRAMMAR_THOUGHT_OUTPUT = r"""
root ::= thought
thought ::= "{" ws "\"action\"" ws ":" ws action_value ws "," ws "\"reason\"" ws ":" ws string ws "}"
action_value ::= "\"" ("explore" | "think" | "act" | "verify" | "visualize" | "respond") "\""
string ::= "\"" (char)* "\""
char ::= [^"\\] | "\\" ["\\/bfnrt] | "\\u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]
ws ::= [ \t\n]*
"""

GRAMMAR_CODE_BLOCK = r"""
root ::= code
code ::= "{" ws "\"code\"" ws ":" ws string ws "," ws "\"language\"" ws ":" ws ("\"python\"" | "\"sql\"") ws "}"
string ::= "\"" (char)* "\""
char ::= [^"\\\n] | "\\" ["\\/bfnrt] | "\\n" | "\\u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]
ws ::= [ \t\n]*
"""

GRAMMAR_VERIFY_RESULT = r"""
root ::= result
result ::= "{" ws "\"pass\"" ws ":" ws boolean ws "," ws "\"issues\"" ws ":" ws array ws "}"
boolean ::= "true" | "false"
array ::= "[" ws "]" | "[" ws string (ws "," ws string)* ws "]"
string ::= "\"" (char)* "\""
char ::= [^"\\] | "\\" ["\\/bfnrt] | "\\u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]
ws ::= [ \t\n]*
"""

GRAMMAR_VISUALIZATION_DECISION = r"""
root ::= decision
decision ::= "{" ws "\"should_visualize\"" ws ":" ws boolean ws "," ws "\"chart_type\"" ws ":" ws (chart_type | "null") ws "," ws "\"rationale\"" ws ":" ws string ws "}"
boolean ::= "true" | "false"
chart_type ::= "\"bar\"" | "\"line\"" | "\"scatter\"" | "\"histogram\"" | "\"box\""
string ::= "\"" (char)* "\""
char ::= [^"\\] | "\\" ["\\/bfnrt] | "\\u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]
ws ::= [ \t\n]*
"""


class ModelClient:
    """
    Singleton wrapper for llama-cpp-python model.
    
    Loads the model once and provides methods for grammar-constrained
    generation with the chat template from the GGUF metadata.
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
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python not installed. Run: pip install llama-cpp-python"
            )
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Load model with GPU offloading and limited context
        self.model = Llama(
            model_path=str(self.model_path),
            n_ctx=MODEL_N_CTX,
            n_gpu_layers=MODEL_N_GPU_LAYERS,
            n_threads=None,  # Auto-detect
            verbose=False,
            use_mmap=True,
            use_mlock=False,
        )
    
    def _apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        """
        Apply the chat template from GGUF metadata.
        
        Args:
            messages: List of dicts with 'role' and 'content' keys.
                     Roles: 'system', 'user', 'assistant'.
        
        Returns:
            Formatted prompt string.
        """
        # Use llama-cpp's built-in template application
        try:
            return self.model.create_chat_completion(
                messages=messages,
                max_tokens=1,
                temperature=0,
            )['choices'][0]['message']['content']
        except Exception:
            # Fallback: manual template for Mistral/Ministral format
            formatted = ""
            for msg in messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                if role == 'system':
                    formatted += f"[INST] <<SYS>>\n{content}\n<</SYS>>\n\n"
                elif role == 'user':
                    formatted += f"{content} [/INST]"
                elif role == 'assistant':
                    formatted += f"{content}</s><s>[INST] "
            return formatted
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        grammar: Optional[str] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        """
        Generate a response with optional grammar constraints.
        
        Args:
            messages: Chat messages list.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0 for deterministic).
            grammar: Optional GBNF grammar string for constrained decoding.
            stop_sequences: Optional list of strings to stop generation at.
        
        Returns:
            Generated text string.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Build grammar object if provided
        grammar_obj = None
        if grammar:
            from llama_cpp import LlamaGrammar
            grammar_obj = LlamaGrammar.from_string(grammar)
        
        # Generate using chat completion
        response = self.model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            grammar=grammar_obj,
            stop=stop_sequences,
        )
        
        return response['choices'][0]['message']['content']
    
    def generate_structured(
        self,
        messages: List[Dict[str, str]],
        output_schema: str,
        grammar: str,
        max_tokens: int = 256,
    ) -> Dict[str, Any]:
        """
        Generate a structured JSON response with grammar constraints.
        
        Args:
            messages: Chat messages list.
            output_schema: Description of expected output structure.
            grammar: GBNF grammar for the expected JSON shape.
            max_tokens: Maximum tokens to generate.
        
        Returns:
            Parsed JSON dict.
        """
        # Append schema instruction to last user message
        augmented_messages = messages.copy()
        last_msg = augmented_messages[-1]
        if last_msg['role'] == 'user':
            augmented_messages[-1] = {
                'role': 'user',
                'content': f"{last_msg['content']}\n\nRespond in this exact JSON format:\n{output_schema}"
            }
        
        raw_response = self.generate(
            messages=augmented_messages,
            max_tokens=max_tokens,
            temperature=0.2,  # Low temp for structured output
            grammar=grammar,
        )
        
        # Parse JSON response
        try:
            # Extract JSON from response (handle markdown code blocks)
            import re
            json_match = re.search(r'\{[^{}]*\}', raw_response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return json.loads(raw_response)
        except json.JSONDecodeError as e:
            return {'error': f'Failed to parse JSON: {e}', 'raw': raw_response}
    
    def think_step(
        self,
        system_prompt: str,
        context: str,
        user_query: str,
    ) -> Dict[str, str]:
        """
        Generate the next action decision (Think node).
        
        Returns dict with 'action' and 'reason' keys.
        """
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': f"Context:\n{context}\n\nQuery: {user_query}\n\nWhat is the next action?"}
        ]
        
        return self.generate_structured(
            messages=messages,
            output_schema='{"action": "explore|think|act|verify|visualize|respond", "reason": "string"}',
            grammar=GRAMMAR_THOUGHT_OUTPUT,
        )
    
    def generate_code(
        self,
        system_prompt: str,
        task_description: str,
        data_context: str,
    ) -> Dict[str, str]:
        """
        Generate code for execution (Act node).
        
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
    
    def verify_result(
        self,
        system_prompt: str,
        task: str,
        code: str,
        result: str,
    ) -> Dict[str, Any]:
        """
        Verify execution results (Verify node).
        
        Returns dict with 'pass' (bool) and 'issues' (list of strings).
        """
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': f"Task: {task}\n\nCode executed:\n{code}\n\nResult:\n{result}\n\nDoes this correctly answer the query? Identify any issues."}
        ]
        
        return self.generate_structured(
            messages=messages,
            output_schema='{"pass": true|false, "issues": ["string"]}',
            grammar=GRAMMAR_VERIFY_RESULT,
        )
    
    def decide_visualization(
        self,
        system_prompt: str,
        findings: str,
        query: str,
    ) -> Dict[str, Any]:
        """
        Decide if visualization would help (VizDecision node).
        
        Returns dict with 'should_visualize', 'chart_type', 'rationale'.
        """
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': f"Query: {query}\n\nFindings:\n{findings}\n\nWould a chart help communicate these findings?"}
        ]
        
        return self.generate_structured(
            messages=messages,
            output_schema='{"should_visualize": true|false, "chart_type": "bar|line|scatter|histogram|box|null", "rationale": "string"}',
            grammar=GRAMMAR_VISUALIZATION_DECISION,
        )
    
    def generate_response(
        self,
        system_prompt: str,
        query: str,
        findings: str,
        interpretation: str,
    ) -> str:
        """
        Generate final response to user (Respond node).
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
