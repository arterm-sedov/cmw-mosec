"""Process management for Mosec servers."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import requests

from .server_config import MosecModelConfig, ServerStatus

logger = logging.getLogger(__name__)

PID_DIR = Path.home() / ".cmw-mosec"


def _pid_file_key(model_key: str) -> str:
    """Filesystem-safe key for PID file (slashes not allowed on Windows)."""
    return model_key.replace("/", "-")


def _get_pid_file(model_key: str) -> Path:
    """Get path to PID file for a model."""
    PID_DIR.mkdir(parents=True, exist_ok=True)
    return PID_DIR / f"{_pid_file_key(model_key)}.pid"


def _get_actual_device(pid: int) -> str:
    """Detect actual device (cuda/cpu) by checking GPU usage."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            gpu_pids = [int(line.strip()) for line in result.stdout.strip().split("\n") if line.strip()]
            if pid in gpu_pids:
                return "cuda"
        return "cpu"
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        return "cpu"


def _save_pid(model_key: str, pid: int, config: MosecModelConfig, actual_device: str | None = None) -> None:
    """Save process info to PID file."""
    pid_file = _get_pid_file(model_key)
    data = {
        "pid": pid,
        "model_key": model_key,
        "model_id": config.model_id,
        "model_type": config.model_type,
        "port": config.port,
        "device": config.device,
        "actual_device": actual_device,
        "started_at": time.time(),
    }
    pid_file.write_text(json.dumps(data))


def _load_pid_info(model_key: str) -> dict[str, Any] | None:
    """Load process info from PID file."""
    pid_file = _get_pid_file(model_key)
    if not pid_file.exists():
        return None
    try:
        return json.loads(pid_file.read_text())
    except OSError:
        return None


def _remove_pid_file(model_key: str) -> None:
    """Remove PID file."""
    pid_file = _get_pid_file(model_key)
    if pid_file.exists():
        pid_file.unlink()


def _is_process_running(pid: int) -> bool:
    """Check if a process is running."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _check_server_health(port: int, timeout: float = 2.0) -> bool:
    """Check if Mosec server is responding."""
    try:
        response = requests.get(f"http://localhost:{port}/health", timeout=timeout)
        return response.status_code == 200
    except requests.RequestException:
        return False


def _generate_mosec_script(config: MosecModelConfig) -> str:
    """Generate Mosec server script based on model type."""
    if config.model_type == "embedding":
        return _generate_embedding_script(config)
    elif config.model_type == "reranker":
        return _generate_reranker_script(config)
    else:
        return _generate_guard_script(config)


def _generate_embedding_script(config: MosecModelConfig) -> str:
    """Generate Mosec embedding server script."""
    return f'''
import os
from typing import List, Union

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from llmspec import EmbeddingData, EmbeddingRequest, EmbeddingResponse, TokenUsage
from mosec import ClientError, Server, Worker

os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_NAME = "{config.model_id}"
DEVICE = "{config.device}" if "{config.device}" != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
WORKERS = {config.workers}
PORT = {config.port}
DTYPE = "{config.dtype}"


class Embedding(Worker):
    def __init__(self):
        self.model_name = MODEL_NAME
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)
        self.model = transformers.AutoModel.from_pretrained(self.model_name)
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.model.eval()
        if DTYPE == "float16" and self.device != "cpu":
            self.model = self.model.half()
        elif DTYPE == "int8":
            self.model = self.model.quantized = True

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def get_embeddings(self, sentences: Union[str, List[Union[str, List[int]]]]]):
        encoded_input = self.tokenizer(
            sentences, padding=True, truncation=True, return_tensors="pt"
        )
        inputs = encoded_input.to(self.device)
        with torch.no_grad():
            model_output = self.model(**inputs)
        sentence_embeddings = self.mean_pooling(model_output, inputs["attention_mask"])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        token_count = inputs["attention_mask"].sum(dim=1).tolist()[0]
        return token_count, sentence_embeddings

    def deserialize(self, data: bytes) -> EmbeddingRequest:
        return EmbeddingRequest.from_bytes(data)

    def serialize(self, data: EmbeddingResponse) -> bytes:
        return data.to_json()

    def forward(self, data: EmbeddingRequest) -> EmbeddingResponse:
        if data.model != self.model_name:
            raise ClientError(
                f"the requested model {{data.model}} is not supported by "
                f"this worker {{self.model_name}}"
            )
        token_count, embeddings = self.get_embeddings(data.input)
        embeddings = embeddings.detach()
        if self.device != "cpu":
            embeddings = embeddings.cpu()
        embeddings = embeddings.numpy()
        if data.encoding_format == "base64":
            embeddings = [
                base64.b64encode(emb.astype(np.float32).tobytes()).decode("utf-8")
                for emb in embeddings
            ]
        else:
            embeddings = [emb.tolist() for emb in embeddings]

        return EmbeddingResponse(
            data=[EmbeddingData(embedding=emb, index=i) for i, emb in enumerate(embeddings)],
            model=self.model_name,
            usage=TokenUsage(
                prompt_tokens=token_count,
                completion_tokens=0,
                total_tokens=token_count,
            ),
        )


if __name__ == "__main__":
    server = Server()
    from mosec import Runtime
    emb = Runtime(Embedding)
    server.register_runtime({{"/v1/embeddings": [emb], "/embeddings": [emb]}})
    server.run(host="0.0.0.0", port=PORT, workers=WORKERS)
'''


def _generate_reranker_script(config: MosecModelConfig) -> str:
    """Generate Mosec reranker server script."""
    return f'''
import os
from typing import List

from msgspec import Struct
from sentence_transformers import CrossEncoder
from mosec import Server, Worker
from mosec.mixin import TypedMsgPackMixin

os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_NAME = "{config.model_id}"
WORKERS = {config.workers}
PORT = {config.port}
DTYPE = "{config.dtype}"


class Request(Struct, kw_only=True):
    query: str
    docs: List[str]


class Response(Struct, kw_only=True):
    scores: List[float]


class Reranker(TypedMsgPackMixin, Worker):
    def __init__(self):
        self.model_name = MODEL_NAME
        self.model = CrossEncoder(self.model_name)

    def forward(self, data: Request) -> Response:
        scores = self.model.predict([[data.query, doc] for doc in data.docs])
        return Response(scores=scores.tolist())


if __name__ == "__main__":
    server = Server()
    server.append_worker(Reranker, num=WORKERS)
    server.run(host="0.0.0.0", port=PORT)
'''


def _generate_guard_script(config: MosecModelConfig) -> str:
    """Generate Mosec guard server script.

    Implements content safety moderation with three-tier classification:
    - Safe: Content is safe
    - Controversial: Content may be context-dependent
    - Unsafe: Content is harmful

    Output format:
        Safety: Safe|Controversial|Unsafe
        Categories: <list of categories>
        Refusal: Yes|No (for response moderation)
    """
    max_new_tokens = config.max_new_tokens or 128

    return f'''
import os
import re
from typing import List, Optional

import torch
import transformers
from mosec import Server, Worker
from mosec.mixin import TypedMsgPackMixin
from msgspec import Struct

os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_NAME = "{config.model_id}"
WORKERS = {config.workers}
PORT = {config.port}
DTYPE = "{config.dtype}"
MAX_NEW_TOKENS = {max_new_tokens}

SAFETY_CATEGORIES = [
    "Violent",
    "Non-violent Illegal Acts",
    "Sexual Content or Sexual Acts",
    "PII",
    "Suicide & Self-Harm",
    "Unethical Acts",
    "Politically Sensitive Topics",
    "Copyright Violation",
    "Jailbreak",
]


class GuardRequest(Struct, kw_only=True):
    content: str
    context: Optional[str] = None
    moderation_type: str = "prompt"


class GuardResponse(Struct, kw_only=True):
    safety_level: str
    categories: List[str]
    refusal: Optional[str] = None
    is_safe: bool
    raw_output: str
    model: str = MODEL_NAME


class GuardWorker(TypedMsgPackMixin, Worker):
    """MOSEC Worker for Qwen3Guard content safety moderation.

    Supports both prompt moderation (user input only) and response moderation
    (user query + assistant response) with three-tier classification.
    """

    def __init__(self):
        self.model_name = MODEL_NAME
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if DTYPE == "float16" else torch.bfloat16,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        self.model.eval()

        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for output parsing."""
        self.safety_pattern = re.compile(
            r"Safety:\\s*(Safe|Controversial|Unsafe)",
            re.IGNORECASE
        )
        category_list = "|".join(re.escape(cat) for cat in SAFETY_CATEGORIES)  # noqa: F821
        self.category_pattern = re.compile(f"({category_list})", re.IGNORECASE)  # noqa: F821
        self.refusal_pattern = re.compile(r"Refusal:\\s*(Yes|No)", re.IGNORECASE)

    def _format_prompt_moderation(self, content: str) -> str:
        """Format prompt for user input moderation."""
        messages = [{{"role": "user", "content": content}}]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    def _format_response_moderation(self, user_prompt: str, assistant_response: str) -> str:
        """Format prompt for assistant response moderation."""
        messages = [
            {{"role": "user", "content": user_prompt}},
            {{"role": "assistant", "content": assistant_response}}
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    def _parse_output(self, output: str) -> dict:
        """Parse model output into structured result."""
        result = {{
            "safety_level": "Unknown",
            "categories": [],
            "refusal": None,
            "raw_output": output,
        }}

        safety_match = self.safety_pattern.search(output)
        if safety_match:
            result["safety_level"] = safety_match.group(1).capitalize()

        category_matches = self.category_pattern.findall(output)
        if category_matches:
            normalized = []
            for match in category_matches:
                for cat in SAFETY_CATEGORIES:
                    if cat.lower() == match.lower():
                        normalized.append(cat)
                        break
                else:
                    normalized.append(match)
            result["categories"] = normalized
        else:
            result["categories"] = ["None"]

        refusal_match = self.refusal_pattern.search(output)
        if refusal_match:
            result["refusal"] = refusal_match.group(1).capitalize()

        return result

    def forward(self, data: GuardRequest) -> GuardResponse:
        if data.moderation_type == "response" and data.context:
            prompt = self._format_response_moderation(data.context, data.content)
        else:
            prompt = self._format_prompt_moderation(data.content)

        model_inputs = self.tokenizer(
            [prompt],
            return_tensors="pt",
            truncation=True,
            max_length=32768
        )

        if torch.cuda.is_available():
            model_inputs = {{k: v.cuda() for k, v in model_inputs.items()}}

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        output_ids = generated_ids[0][len(model_inputs["input_ids"][0]):].tolist()
        output_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)

        parsed = self._parse_output(output_text)

        return GuardResponse(
            safety_level=parsed["safety_level"],
            categories=parsed["categories"],
            refusal=parsed.get("refusal"),
            is_safe=parsed["safety_level"] == "Safe",
            raw_output=parsed["raw_output"],
        )


if __name__ == "__main__":
    server = Server()
    server.append_worker(GuardWorker, num=WORKERS)
    server.run(host="0.0.0.0", port=PORT)
'''


class MosecServerManager:
    """Manages Mosec server processes."""

    def __init__(self):
        self.pid_dir = PID_DIR

    def start(
        self,
        model_key: str,
        config: MosecModelConfig,
        background: bool = True,
    ) -> bool:
        """Start a Mosec server.

        Args:
            model_key: Model identifier
            config: Server configuration
            background: Whether to run in background

        Returns:
            True if started successfully
        """
        status = self.get_status(model_key, config)
        if status.is_running:
            logger.info(f"Server for {model_key} already running on port {config.port}")
            return True

        import sys

        server_script = _generate_mosec_script(config)
        cmd = [sys.executable, "-c", server_script]
        logger.info(f"Starting Mosec server for {config.model_id} on port {config.port}")

        try:
            if background:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
            else:
                process = subprocess.Popen(cmd)

            _save_pid(model_key, process.pid, config)

            if background:
                logger.info(f"Waiting for server to start on port {config.port}...")
                for _ in range(60):
                    if _check_server_health(config.port):
                        logger.info(f"Server {model_key} is ready!")
                        actual_device = _get_actual_device(process.pid)
                        _save_pid(model_key, process.pid, config, actual_device)
                        logger.info(f"Server {model_key} running on device: {actual_device}")
                        return True
                    time.sleep(1)
                    if process.poll() is not None:
                        logger.error(f"Server process exited with code {process.returncode}")
                        _remove_pid_file(model_key)
                        return False

                logger.warning(f"Server may still be starting... (port {config.port})")
                return True
            else:
                process.wait()
                return process.returncode == 0

        except FileNotFoundError:
            logger.error("Python not found")
            return False
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            return False

    def stop(self, model_key: str) -> bool:
        """Stop a Mosec server."""
        pid_info = _load_pid_info(model_key)
        if not pid_info:
            logger.info(f"No PID file found for {model_key}")
            return True

        pid = pid_info.get("pid")
        if not pid:
            _remove_pid_file(model_key)
            return True

        if not _is_process_running(pid):
            logger.info(f"Server {model_key} (PID {pid}) is not running")
            _remove_pid_file(model_key)
            return True

        logger.info(f"Stopping server {model_key} (PID {pid})...")

        try:
            os.kill(pid, signal.SIGTERM)

            for _ in range(10):
                if not _is_process_running(pid):
                    logger.info(f"Server {model_key} stopped gracefully")
                    _remove_pid_file(model_key)
                    return True
                time.sleep(1)

            logger.warning(f"Force killing server {model_key} (PID {pid})...")
            kill_signal = signal.SIGTERM if sys.platform == "win32" else signal.SIGKILL
            with contextlib.suppress(OSError, ProcessLookupError):
                os.kill(pid, kill_signal)
            time.sleep(1)

            _remove_pid_file(model_key)
            return True

        except (OSError, ProcessLookupError) as e:
            logger.warning(f"Error stopping process: {e}")
            _remove_pid_file(model_key)
            return True

    def get_status(self, model_key: str, config: MosecModelConfig) -> ServerStatus:
        """Get status of a server."""
        pid_info = _load_pid_info(model_key)
        pid = pid_info.get("pid") if pid_info else None

        is_running = False
        uptime = None

        if pid and _is_process_running(pid) and _check_server_health(config.port):
            is_running = True
            if pid_info and "started_at" in pid_info:
                uptime = time.time() - pid_info["started_at"]

        device = config.device
        if pid_info and "actual_device" in pid_info and pid_info["actual_device"]:
            device = pid_info["actual_device"]

        return ServerStatus(
            model_key=model_key,
            model_id=config.model_id,
            model_type=config.model_type,
            port=config.port,
            device=device,
            pid=pid,
            is_running=is_running,
            uptime_seconds=uptime,
        )

    def list_running(self) -> list[ServerStatus]:
        """List all running servers."""
        from .server_config import ModelRegistry

        registry = ModelRegistry()
        statuses = []
        for slug in registry.list_embeddings() + registry.list_rerankers() + registry.list_guards():
            config = registry.get_config(slug)
            status = self.get_status(slug, config)
            if status.pid:
                statuses.append(status)

        return statuses

    def stop_all(self) -> bool:
        """Stop all running servers."""
        running = self.list_running()
        if not running:
            logger.info("No servers are running")
            return True

        success = True
        for status in running:
            if not self.stop(status.model_key):
                success = False

        return success
