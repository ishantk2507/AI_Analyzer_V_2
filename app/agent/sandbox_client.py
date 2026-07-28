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
import logging
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

logger = logging.getLogger(__name__)


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
        logger.info("Initializing SandboxClient with data_dir=%s", data_dir)
        self.data_dir = Path(data_dir).resolve()
        self.scratch_dir = Path(scratch_dir or tempfile.mkdtemp(prefix='sandbox_scratch_')).resolve()
        self.scratch_dir.mkdir(parents=True, exist_ok=True)

        self.container_id: Optional[str] = None
        self.exec_process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._cleanup_registered = False

        # Start the container
        try:
            self._start_container()
            logger.info("Sandbox container started successfully")
        except Exception as e:
            logger.error("Failed to start sandbox container: %s", e)
            raise

        # Register cleanup on exit
        atexit.register(self.stop)

    def _start_container(self):
        """Start the sandbox container with resource limits."""
        logger.info("Starting sandbox container")
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

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            self.container_id = result.stdout.strip()
            logger.info("Container started with id=%s", self.container_id[:12])
        except subprocess.CalledProcessError as e:
            logger.error("Docker run failed: %s, stderr=%s", e, e.stderr)
            raise
        except FileNotFoundError:
            logger.error("Docker command not found. Is Docker installed and in PATH?")
            raise

        # Start the persistent REPL process
        self._start_repl()

    def _start_repl(self):
        """Start the persistent REPL inside the container."""
        if not self.container_id:
            logger.error("Cannot start REPL: container not started")
            raise RuntimeError("Container not started")

        cmd = [
            'docker', 'exec', '-i',
            self.container_id,
            'python', '/persistent_repl.py'
        ]

        logger.debug("Starting REPL process")
        # Start process with pipes for stdin/stdout
        self.exec_process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        logger.info("REPL process started")

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

        # Log the request (code) for debugging
        logger.info("Executing code (first 300 chars): %s", code[:300])
        logger.debug("Full code being executed:\n%s", code)

        request = {
            'code': code,
            'timeout': timeout
        }

        with self._lock:
            if not self.exec_process or self.exec_process.poll() is not None:
                # REPL died, restart it
                logger.warning("REPL process dead, restarting")
                self._start_repl()

            # Write request to stdin
            request_line = json.dumps(request) + '\n'
            try:
                self.exec_process.stdin.write(request_line)
                self.exec_process.stdin.flush()
            except BrokenPipeError:
                logger.warning("BrokenPipeError writing to REPL, restarting")
                # Restart REPL and retry
                self._start_repl()
                self.exec_process.stdin.write(request_line)
                self.exec_process.stdin.flush()
            except Exception as e:
                logger.error("Error writing to REPL: %s", e)
                return {
                    'stdout': '',
                    'stderr': '',
                    'result_repr': '',
                    'error': f'Failed to write to sandbox: {e}',
                    'artifacts': []
                }

            # Read response from stdout
            try:
                response_line = self.exec_process.stdout.readline()
            except Exception as e:
                logger.error("Error reading from REPL: %s", e)
                return {
                    'stdout': '',
                    'stderr': '',
                    'result_repr': '',
                    'error': f'Failed to read from sandbox: {e}',
                    'artifacts': []
                }

            if not response_line:
                logger.error("Sandbox process terminated unexpectedly")
                return {
                    'stdout': '',
                    'stderr': '',
                    'result_repr': '',
                    'error': 'Sandbox process terminated unexpectedly',
                    'artifacts': []
                }

            try:
                response = json.loads(response_line)
            except json.JSONDecodeError as e:
                logger.error("Invalid JSON from sandbox: %s, line=%s", e, response_line[:200])
                response = {
                    'stdout': response_line,
                    'stderr': '',
                    'result_repr': '',
                    'error': 'Invalid JSON response from sandbox',
                    'artifacts': []
                }

            # Log the response summary
            if response.get('error'):
                logger.warning("Execution failed: %s", response['error'])
                logger.debug("Full error response: %s", response)
            else:
                logger.info("Execution succeeded. stdout length=%d, result_repr=%s",
                           len(response.get('stdout', '')),
                           response.get('result_repr', '')[:100])
                logger.debug("Full execution response: %s", response)
            print(f"Execution response: {response}")
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
            logger.warning("Cannot get artifact: container not running")
            return None

        logger.info("Retrieving artifact: %s", artifact_path)

        # Extract file from container
        cmd = [
            'docker', 'cp',
            f'{self.container_id}:{artifact_path}',
            '-'
        ]

        try:
            logger.debug("Copying artifact %s", artifact_path)
            result = subprocess.run(cmd, capture_output=True, check=True)
            logger.info("Artifact retrieved successfully, size=%d bytes", len(result.stdout))
            return result.stdout
        except subprocess.CalledProcessError as e:
            logger.error("Failed to copy artifact %s: %s", artifact_path, e)
            return None
        except FileNotFoundError:
            logger.error("Docker command not found")
            return None

    def list_artifacts(self) -> List[str]:
        """List all PNG artifacts in /scratch."""
        if not self.container_id:
            logger.warning("Cannot list artifacts: container not running")
            return []

        cmd = [
            'docker', 'exec',
            self.container_id,
            'find', '/scratch', '-name', '*.png', '-type', 'f'
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            artifacts = [line.strip() for line in result.stdout.split('\n') if line.strip()]
            logger.info("Found %d artifacts: %s", len(artifacts), artifacts)
            return artifacts
        except subprocess.CalledProcessError as e:
            logger.error("Failed to list artifacts: %s", e)
            return []
        except FileNotFoundError:
            logger.error("Docker command not found")
            return []

    def stop(self):
        """Stop and remove the sandbox container."""
        logger.info("Stopping sandbox")
        if self.exec_process:
            try:
                self.exec_process.terminate()
                self.exec_process.wait(timeout=5)
                logger.info("REPL process terminated")
            except Exception as e:
                logger.warning("Failed to terminate REPL: %s", e)
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
                logger.info("Container %s stopped and removed", self.container_id[:12])
            except Exception as e:
                logger.warning("Failed to stop/remove container: %s", e)
            self.container_id = None

        # Clean up scratch directory
        if self.scratch_dir.exists():
            try:
                shutil.rmtree(self.scratch_dir)
                logger.info("Scratch directory cleaned up")
            except Exception as e:
                logger.warning("Failed to clean scratch dir: %s", e)