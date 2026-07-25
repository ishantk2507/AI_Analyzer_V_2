"""
Sandbox client for Docker-based code execution.

Manages a persistent container with --network none, communicates via
docker exec -i using JSON-lines protocol over stdin/stdout.
"""

import subprocess
import json
import threading
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List
import atexit

from app.config import (
    SANDBOX_IMAGE_NAME,
    SANDBOX_MEMORY,
    SANDBOX_MEMORY_SWAP,
    SANDBOX_CPUS,
    SANDBOX_TIMEOUT_DEFAULT,
    SANDBOX_TIMEOUT_MAX,
    DOCKER_DIR,
)


class SandboxClient:
    """
    Client for the isolated code execution sandbox.
    
    Starts a single container per session and maintains a persistent
    docker exec process for low-overhead communication.
    """
    
    def __init__(self, data_dir: str, scratch_dir: Optional[str] = None):
        """
        Initialize and start the sandbox container.
        
        Args:
            data_dir: Host path to mount as read-only /data.
            scratch_dir: Host path to mount as read-write /scratch.
                        Created if doesn't exist.
        """
        self.data_dir = Path(data_dir).resolve()
        self.scratch_dir = Path(scratch_dir or tempfile.mkdtemp(prefix='sandbox_scratch_')).resolve()
        self.scratch_dir.mkdir(parents=True, exist_ok=True)
        
        self.container_id: Optional[str] = None
        self.exec_process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._cleanup_registered = False
        
        # Start the container
        self._start_container()
        
        # Register cleanup on exit
        atexit.register(self.stop)
    
    def _start_container(self):
        """Start the sandbox container with resource limits."""
        cmd = [
            'docker', 'run', '-d',
            '--network', 'none',
            '--memory', SANDBOX_MEMORY,
            '--memory-swap', SANDBOX_MEMORY_SWAP,
            '--cpus', str(SANDBOX_CPUS),
            '-v', f'{self.data_dir}:/data:ro',
            '-v', f'{self.scratch_dir}:/scratch:rw',
            '--name', f'sandbox-{id(self)}',  # Unique name per instance
            SANDBOX_IMAGE_NAME,
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        self.container_id = result.stdout.strip()
        
        # Start the persistent REPL process
        self._start_repl()
    
    def _start_repl(self):
        """Start the persistent REPL inside the container."""
        if not self.container_id:
            raise RuntimeError("Container not started")
        
        cmd = [
            'docker', 'exec', '-i',
            self.container_id,
            'python', '/persistent_repl.py'
        ]
        
        # Start process with pipes for stdin/stdout
        self.exec_process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
    
    def execute(self, code: str, timeout: int = SANDBOX_TIMEOUT_DEFAULT) -> Dict[str, Any]:
        """
        Execute Python or SQL code in the sandbox.
        
        Args:
            code: Python code string or SQL query.
            timeout: Maximum execution time in seconds.
        
        Returns:
            Dict with keys: stdout, stderr, result_repr, error, artifacts
        """
        if timeout > SANDBOX_TIMEOUT_MAX:
            timeout = SANDBOX_TIMEOUT_MAX
        if timeout < 1:
            timeout = 1
        
        request = {
            'code': code,
            'timeout': timeout
        }
        
        with self._lock:
            if not self.exec_process or self.exec_process.poll() is not None:
                # REPL died, restart it
                self._start_repl()
            
            # Write request to stdin
            request_line = json.dumps(request) + '\n'
            try:
                self.exec_process.stdin.write(request_line)
                self.exec_process.stdin.flush()
            except BrokenPipeError:
                # Restart REPL and retry
                self._start_repl()
                self.exec_process.stdin.write(request_line)
                self.exec_process.stdin.flush()
            
            # Read response from stdout
            response_line = self.exec_process.stdout.readline()
            if not response_line:
                return {
                    'stdout': '',
                    'stderr': '',
                    'result_repr': '',
                    'error': 'Sandbox process terminated unexpectedly',
                    'artifacts': []
                }
            
            try:
                response = json.loads(response_line)
            except json.JSONDecodeError:
                response = {
                    'stdout': response_line,
                    'stderr': '',
                    'result_repr': '',
                    'error': 'Invalid JSON response from sandbox',
                    'artifacts': []
                }
            
            return response
    
    def get_artifact(self, artifact_path: str) -> Optional[bytes]:
        """
        Retrieve a generated artifact (e.g., chart PNG) from the sandbox.
        
        Args:
            artifact_path: Path inside container (e.g., /scratch/chart.png).
        
        Returns:
            File contents as bytes, or None if file doesn't exist.
        """
        if not self.container_id:
            return None
        
        # Extract file from container
        cmd = [
            'docker', 'cp',
            f'{self.container_id}:{artifact_path}',
            '-'
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, check=True)
            return result.stdout
        except subprocess.CalledProcessError:
            return None
    
    def list_artifacts(self) -> List[str]:
        """List all PNG artifacts in /scratch."""
        if not self.container_id:
            return []
        
        cmd = [
            'docker', 'exec',
            self.container_id,
            'find', '/scratch', '-name', '*.png', '-type', 'f'
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return [line.strip() for line in result.stdout.split('\n') if line.strip()]
        except subprocess.CalledProcessError:
            return []
    
    def stop(self):
        """Stop and remove the sandbox container."""
        if self.exec_process:
            try:
                self.exec_process.terminate()
                self.exec_process.wait(timeout=5)
            except Exception:
                pass
            self.exec_process = None
        
        if self.container_id:
            try:
                subprocess.run(
                    ['docker', 'stop', '-t', '5', self.container_id],
                    capture_output=True,
                    timeout=30
                )
                subprocess.run(
                    ['docker', 'rm', '-f', self.container_id],
                    capture_output=True,
                    timeout=30
                )
            except Exception:
                pass
            self.container_id = None
        
        # Clean up scratch directory
        if self.scratch_dir.exists():
            try:
                shutil.rmtree(self.scratch_dir)
            except Exception:
                pass
