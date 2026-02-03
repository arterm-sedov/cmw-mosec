"""Process management for Mosec servers.

Single combined server with dynamic model loading/unloading.
"""

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

from .server_config import (
    ModelRegistry,
    ServerSettings,
    ServerStatus,
    load_active_models,
    load_server_settings,
)

logger = logging.getLogger(__name__)

PID_DIR = Path.home() / ".cmw-mosec"
SERVER_PID_FILE = PID_DIR / "server.pid"


def _ensure_pid_dir() -> None:
    """Ensure PID directory exists."""
    PID_DIR.mkdir(parents=True, exist_ok=True)


def _save_server_pid(pid: int, port: int) -> None:
    """Save server PID info."""
    _ensure_pid_dir()
    data = {
        "pid": pid,
        "port": port,
        "started_at": time.time(),
    }
    SERVER_PID_FILE.write_text(json.dumps(data))


def _load_server_pid() -> dict[str, Any] | None:
    """Load server PID info."""
    if not SERVER_PID_FILE.exists():
        return None
    try:
        return json.loads(SERVER_PID_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _remove_server_pid() -> None:
    """Remove server PID file."""
    if SERVER_PID_FILE.exists():
        SERVER_PID_FILE.unlink()


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


def _generate_server_script(
    settings: ServerSettings,
    embedding_model: str | None,
    reranker_model: str | None,
    guard_model: str | None,
) -> str:
    """Generate the combined Mosec server script."""
    embedder_code = ""
    reranker_code = ""
    guard_code = ""

    # Embedding worker
    if embedding_model:
        embedder_code = f'''
import os
from typing import List, Union

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from llmspec import EmbeddingData, EmbeddingRequest, EmbeddingResponse, TokenUsage
from mosec import ClientError, Worker

os.environ["TOKENIZERS_PARALLELISM"] = "false"

EMBEDDING_MODEL = "{embedding_model}"
DTYPE = "{settings.dtype}"


class EmbeddingWorker(Worker):
    def __init__(self):
        self.model_name = EMBEDDING_MODEL
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

    def cls_pooling(self, model_output):
        """CLS pooling - use first token (CLS) as sentence representation.

        Recommended for FRIDA according to HuggingFace docs.
        """
        return model_output[0][:, 0, :]

    def get_embeddings(self, sentences: Union[str, List[Union[str, List[int]]]]]):
        encoded_input = self.tokenizer(
            sentences, padding=True, truncation=True, return_tensors="pt"
        )
        inputs = encoded_input.to(self.device)
        with torch.no_grad():
            model_output = self.model(**inputs)
        sentence_embeddings = self.cls_pooling(model_output)
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
'''

    # Reranker worker
    if reranker_model:
        reranker_code = f'''
import os
from typing import List

from msgspec import Struct
from sentence_transformers import CrossEncoder
from mosec import Worker
from mosec.mixin import TypedMsgPackMixin

os.environ["TOKENIZERS_PARALLELISM"] = "false"

RERANKER_MODEL = "{reranker_model}"


class RerankRequest(Struct, kw_only=True):
    query: str
    docs: List[str]


class RerankResponse(Struct, kw_only=True):
    scores: List[float]


class RerankerWorker(TypedMsgPackMixin, Worker):
    def __init__(self):
        self.model_name = RERANKER_MODEL
        self.model = CrossEncoder(self.model_name)

    def forward(self, data: RerankRequest) -> RerankResponse:
        scores = self.model.predict([[data.query, doc] for doc in data.docs])
        return RerankResponse(scores=scores.tolist())
'''

    # Guard worker
    if guard_model:
        guard_code = f'''
import os
import re
from typing import List, Optional

import torch
import transformers
from mosec import Worker
from mosec.mixin import TypedMsgPackMixin
from msgspec import Struct

os.environ["TOKENIZERS_PARALLELISM"] = "false"

GUARD_MODEL = "{guard_model}"
DTYPE = "{settings.dtype}"
MAX_NEW_TOKENS = {settings.idle_timeout}

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

SAFETY_CATEGORY_PATTERN = "|".join(re.escape(cat) for cat in SAFETY_CATEGORIES)


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
    model: str = GUARD_MODEL


class GuardWorker(TypedMsgPackMixin, Worker):
    def __init__(self):
        self.model_name = GUARD_MODEL
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
        self.safety_pattern = re.compile(
            r"Safety:\\s*(Safe|Controversial|Unsafe)",
            re.IGNORECASE
        )
        category_pattern = "|".join(re.escape(cat) for cat in SAFETY_CATEGORIES)
        # ruff: noqa: F821
        self.category_pattern = re.compile(f"({category_pattern})", re.IGNORECASE)
        self.refusal_pattern = re.compile(r"Refusal:\\s*(Yes|No)", re.IGNORECASE)

    def _format_prompt(self, content: str, context: Optional[str], moderation_type: str) -> str:
        if moderation_type == "response" and context:
            messages = [
                {{"role": "user", "content": context}},
                {{"role": "assistant", "content": content}}
            ]
        else:
            messages = [{{"role": "user", "content": content}}]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    def _parse_output(self, output: str) -> dict:
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
        prompt = self._format_prompt(data.content, data.context, data.moderation_type)

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
'''

    return f'''
import os
import sys

{embedder_code}
{reranker_code}
{guard_code}

from mosec import Server, Runtime

PORT = {settings.server_port}

if __name__ == "__main__":
    server = Server()

    # Register embedding endpoint
    if "EmbeddingWorker" in globals():
        from mosec import Runtime
        emb = Runtime(EmbeddingWorker)
        server.register_runtime({{"/v1/embeddings": [emb], "/embeddings": [emb]}})

    # Register reranker endpoint
    if "RerankerWorker" in globals():
        server.register_runtime({{"/v1/rerank": [Runtime(RerankerWorker)], "/rerank": [Runtime(RerankerWorker)]}})

    # Register guard endpoint
    if "GuardWorker" in globals():
        server.register_runtime({{"/v1/moderate": [Runtime(GuardWorker)], "/moderate": [Runtime(GuardWorker)]}})

    server.run(host="0.0.0.0", port=PORT)
'''


class MosecServerManager:
    """Manages the combined Mosec server with dynamic model loading."""

    def __init__(self):
        self.pid_dir = PID_DIR

    def get_status(self) -> ServerStatus:
        """Get status of the combined server."""
        pid_info = _load_server_pid()
        if not pid_info:
            return ServerStatus(
                model_key="combined",
                model_id="combined",
                model_type="combined",
                port=0,
                device="unknown",
                pid=None,
                is_running=False,
                uptime_seconds=None,
            )

        pid = pid_info.get("pid")
        port = pid_info.get("port", 0)

        is_running = False
        uptime = None

        if pid and _is_process_running(pid) and _check_server_health(port):
            is_running = True
            if "started_at" in pid_info:
                uptime = time.time() - pid_info["started_at"]

        try:
            settings = load_server_settings()
            device = settings.device
        except Exception:
            device = "unknown"

        return ServerStatus(
            model_key="combined",
            model_id="combined",
            model_type="combined",
            port=port,
            device=device,
            pid=pid,
            is_running=is_running,
            uptime_seconds=uptime,
        )

    def is_running(self) -> bool:
        """Check if server is running."""
        status = self.get_status()
        return status.is_running

    def start(
        self,
        embedding_model: str | None = None,
        reranker_model: str | None = None,
        guard_model: str | None = None,
        background: bool = True,
    ) -> tuple[bool, list[str]]:
        """Start the combined server.

        Args:
            embedding_model: Embedding model slug to load
            reranker_model: Reranker model slug to load
            guard_model: Guard model slug to load
            background: Whether to run in background

        Returns:
            Tuple of (success, list of failed models)
        """
        if self.is_running():
            logger.info("Server already running")
            return True, []

        try:
            settings = load_server_settings()
        except Exception as e:
            logger.error(f"Failed to load settings: {e}")
            return False, ["settings"]

        # Use active models from .env if not specified
        active = load_active_models()
        emb_model = embedding_model or active["embedding"]
        rer_model = reranker_model or active["reranker"]
        guard_m = guard_model or active["guard"]

        failed_models = []

        # Validate models exist
        from .server_config import ModelRegistry
        registry = ModelRegistry()

        if emb_model:
            try:
                registry.get_config(emb_model)
            except ValueError as e:
                logger.error(f"Embedding model error: {e}")
                failed_models.append(f"embedding: {emb_model}")
                emb_model = None

        if rer_model:
            try:
                registry.get_config(rer_model)
            except ValueError as e:
                logger.error(f"Reranker model error: {e}")
                failed_models.append(f"reranker: {rer_model}")
                rer_model = None

        if guard_m:
            try:
                registry.get_config(guard_m)
            except ValueError as e:
                logger.error(f"Guard model error: {e}")
                failed_models.append(f"guard: {guard_m}")
                guard_m = None

        if not emb_model and not rer_model and not guard_m:
            logger.error("No valid models to load")
            return False, failed_models

        logger.info(f"Starting server with models: emb={emb_model}, rer={rer_model}, guard={guard_m}")

        server_script = _generate_server_script(settings, emb_model, rer_model, guard_m)
        cmd = [sys.executable, "-c", server_script]

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

            _save_server_pid(process.pid, settings.server_port)

            if background:
                logger.info(f"Waiting for server on port {settings.server_port}...")
                for _ in range(60):
                    if _check_server_health(settings.server_port):
                        logger.info("Server is ready!")
                        return True, failed_models
                    time.sleep(1)
                    if process.poll() is not None:
                        logger.error(f"Server process exited with code {process.returncode}")
                        _remove_server_pid()
                        return False, failed_models

                logger.warning("Server may still be starting...")
                return True, failed_models
            else:
                process.wait()
                return process.returncode == 0, failed_models

        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            return False, failed_models

    def stop(self) -> bool:
        """Stop the server."""
        pid_info = _load_server_pid()
        if not pid_info:
            logger.info("No server running")
            return True

        pid = pid_info.get("pid")
        if not pid:
            _remove_server_pid()
            return True

        if not _is_process_running(pid):
            logger.info("Server not running")
            _remove_server_pid()
            return True

        logger.info(f"Stopping server (PID {pid})...")

        try:
            os.kill(pid, signal.SIGTERM)

            for _ in range(10):
                if not _is_process_running(pid):
                    logger.info("Server stopped gracefully")
                    _remove_server_pid()
                    return True
                time.sleep(1)

            logger.warning(f"Force killing server (PID {pid})...")
            kill_signal = signal.SIGTERM if sys.platform == "win32" else signal.SIGKILL
            with contextlib.suppress(OSError, ProcessLookupError):
                os.kill(pid, kill_signal)
            time.sleep(1)

            _remove_server_pid()
            return True

        except (OSError, ProcessLookupError) as e:
            logger.warning(f"Error stopping process: {e}")
            _remove_server_pid()
            return True

    def load_model(self, model_slug: str) -> bool:
        """Load a model into the running server.

        Note: This requires hot-reload support in Mosec which may not be available.
        For now, this is a placeholder - full implementation would require
        Mosec hot-reload or model swapping functionality.

        Args:
            model_slug: Model to load

        Returns:
            True if successful
        """
        logger.info(f"Model loading requested: {model_slug}")
        logger.warning("Dynamic model loading requires Mosec hot-reload support")

        try:
            ModelRegistry().get_config(model_slug)
            logger.info(f"Model {model_slug} config loaded (server restart required to apply)")
            return True
        except ValueError as e:
            logger.error(f"Unknown model: {e}")
            return False

    def unload_model(self, model_slug: str) -> bool:
        """Unload a model from the running server.

        Note: See load_model() - this is a placeholder.

        Args:
            model_slug: Model to unload

        Returns:
            True if successful
        """
        logger.info(f"Model unloading requested: {model_slug}")
        logger.warning("Dynamic model unloading requires Mosec hot-reload support")
        return True

    def list_loaded_models(self) -> dict[str, str | None]:
        """List currently loaded models.

        Returns:
            Dict with embedding, reranker, guard -> model slug or None
        """
        active = load_active_models()
        return {
            "embedding": active["embedding"],
            "reranker": active["reranker"],
            "guard": active["guard"],
        }

    def stop_all(self) -> bool:
        """Stop all servers (just stops the combined server)."""
        return self.stop()
