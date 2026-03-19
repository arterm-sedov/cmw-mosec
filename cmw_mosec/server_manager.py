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


def _save_server_pid(pid: int, port: int, models: dict[str, str | None] | None = None) -> None:
    """Save server PID info."""
    _ensure_pid_dir()
    data = {
        "pid": pid,
        "port": port,
        "started_at": time.time(),
        "models": models or {},
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
    """Check if Mosec server is responding (using metrics endpoint)."""
    try:
        response = requests.get(f"http://localhost:{port}/metrics", timeout=timeout)
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
    from .server_config import ModelRegistry

    embedder_code = ""
    reranker_code = ""
    guard_code = ""

    guard_max_new_tokens = 128
    if guard_model:
        try:
            guard_config = ModelRegistry().get_guard_config(guard_model)
            guard_max_new_tokens = guard_config.max_new_tokens or 128
        except ValueError:
            guard_max_new_tokens = 128

    # Get pooling config for embedding model
    pooling_method = "mean"  # default
    embed_dtype = "float16"  # default
    model_class = "AutoModel"  # default
    if embedding_model:
        try:
            from .server_config import ModelRegistry

            registry = ModelRegistry()
            config_dict = registry._embeddings.get(embedding_model.lower(), {})
            pooling_method = config_dict.get("pooling", "mean")
            embed_dtype = config_dict.get("dtype", "float16")
            model_class = config_dict.get("model_class", "AutoModel")
        except Exception:
            pooling_method = "mean"
            embed_dtype = "float16"
            model_class = "AutoModel"

    # Embedding worker
    if embedding_model:
        embedder_code = f'''
import base64
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
DTYPE = "{embed_dtype}"
POOLING = "{pooling_method}"
MODEL_CLASS = "{model_class}"


class EmbeddingWorker(Worker):
    def __init__(self):
        self.model_name = EMBEDDING_MODEL
        self.pooling = POOLING
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)

        # Load model class from config (e.g., T5EncoderModel for FRIDA, AutoModel for others)
        if MODEL_CLASS == "T5EncoderModel":
            self.model = transformers.T5EncoderModel.from_pretrained(self.model_name)
        else:
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

    def last_token_pool(self, model_output, attention_mask):
        """Last token pooling - use final token as sentence representation.

        Required for Qwen3 embedding models (causal LM architecture).
        """
        # Get the last non-padding token for each sequence
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = model_output[0].shape[0]
        # Gather last token embeddings
        last_token_embeddings = model_output[0][torch.arange(batch_size, device=model_output[0].device), sequence_lengths]
        return last_token_embeddings

    def get_embeddings(self, texts: Union[str, List[Union[str, List[int]]]]):
        # Handle both single string and list of strings
        if isinstance(texts, str):
            texts = [texts]

        encoded_input = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        )
        inputs = encoded_input.to(self.device)
        with torch.no_grad():
            model_output = self.model(**inputs)

        # Select pooling method based on config or auto-detect for T5
        if hasattr(self.model, 'encoder') and self.pooling == "cls":
            # T5-based models with explicit CLS pooling config (FRIDA)
            sentence_embeddings = self.cls_pooling(model_output)
        elif self.pooling == "last_token":
            # Last token pooling (Qwen3 embedding models)
            sentence_embeddings = self.last_token_pool(model_output, inputs["attention_mask"])
        elif self.pooling == "cls":
            # CLS pooling for non-T5 models
            sentence_embeddings = self.cls_pooling(model_output)
        else:
            # Default: mean pooling
            sentence_embeddings = self.mean_pooling(model_output, inputs["attention_mask"])

        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        token_count = inputs["attention_mask"].sum(dim=1).tolist()[0]
        return token_count, sentence_embeddings

    def deserialize(self, data: bytes) -> EmbeddingRequest:
        # llmspec expects JSON format
        return EmbeddingRequest.from_bytes(data)

    def serialize(self, data: EmbeddingResponse) -> bytes:
        # llmspec's to_json() already returns bytes
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
import json
import os
import torch
from typing import Any, List

from mosec import Worker

os.environ["TOKENIZERS_PARALLELISM"] = "false"

RERANKER_MODEL = "{reranker_model}"


class RerankerWorker(Worker):
    def __init__(self):
        self.model_name = RERANKER_MODEL
        # Determine if this is a Qwen3 model (needs special handling) or standard CrossEncoder
        self.is_qwen3 = "Qwen" in self.model_name and "Reranker" in self.model_name
        
        if self.is_qwen3:
            # Use AutoModelForCausalLM for Qwen3 reranker models
            from transformers import AutoTokenizer, AutoModelForCausalLM
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side='left')
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.model.eval()
            
            # Set pad token if not set (Qwen3 tokenizer may not have pad token initially)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            # Get token IDs for yes/no
            self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
            self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
            
            # Prefix and suffix tokens from the model documentation
            self.prefix = "system\\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \\\"yes\\\" or \\\"no\\\".\\nuser\\n"
            self.suffix = "\\nassistant\\n\\n\\n\\n"
            self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
            self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)
        else:
            # Use sentence_transformers CrossEncoder for standard models (DiTy, BGE, etc.)
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name)
            # Fix for padding token issue
            if self.model.tokenizer.pad_token is None:
                self.model.tokenizer.pad_token = self.model.tokenizer.eos_token

    def deserialize(self, data: bytes) -> dict[str, Any]:
        return json.loads(data.decode("utf-8"))

    def serialize(self, data: dict[str, Any]) -> bytes:
        return json.dumps(data).encode("utf-8")

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        query = data["query"]
        # Accept both "docs" and "documents" field names
        docs = data.get("docs") or data.get("documents")
        
        if self.is_qwen3:
            # Qwen3-specific processing - use client instruction or empty string if none provided
            # This ensures the server is agnostic and doesn't impose semantic bias
            if "instruction" in data:
                instruction = data["instruction"]
                # Handle null/explicitly None instruction
                if instruction is None:
                    instruction = ""
            else:
                # No instruction field provided - use empty string
                instruction = ""
                
            pairs = []
            for doc in docs:
                # Format according to Qwen3 documentation
                output = f"<Instruct>: {{instruction}}\\n<Query>: {{query}}\\n<Document>: {{doc}}".format(instruction=instruction, query=query, doc=doc)
                pairs.append(output)
            
            # Tokenize inputs
            inputs = self.tokenizer(
                pairs, padding=False, truncation='longest_first',
                return_attention_mask=False, max_length=8192
            )
            # Add prefix and suffix tokens
            for i, ele in enumerate(inputs['input_ids']):
                inputs['input_ids'][i] = self.prefix_tokens + ele + self.suffix_tokens
            inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=8192+len(self.prefix_tokens)+len(self.suffix_tokens))
            
            # Move to same device as model
            inputs = {{k: v.to(self.model.device) for k, v in inputs.items()}}
            
            # Get predictions
            with torch.no_grad():
                batch_scores = self.model(**inputs).logits[:, -1, :]
                true_vector = batch_scores[:, self.token_true_id]
                false_vector = batch_scores[:, self.token_false_id]
                batch_scores = torch.stack([false_vector, true_vector], dim=1)
                batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
                scores = batch_scores[:, 1].exp().tolist()
            
            return {{"scores": scores}}
        else:
            # Standard sentence_transformers approach
            # top_k is optional - return all scores, client can slice
            scores = self.model.predict([[query, doc] for doc in docs])
            return {{"scores": scores.tolist()}}
'''

    # Guard worker
    if guard_model:
        guard_code = f'''
import json
import os
import re
from typing import List, Optional

import torch
import transformers
from mosec import Worker

os.environ["TOKENIZERS_PARALLELISM"] = "false"

GUARD_MODEL = "{guard_model}"
DTYPE = "{settings.dtype}"
MAX_NEW_TOKENS = {guard_max_new_tokens}

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


class GuardWorker(Worker):
    def __init__(self):
        self.model_name = GUARD_MODEL
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.float16 if DTYPE == "float16" else torch.bfloat16,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        self.model.eval()
        self._compile_patterns()

    def deserialize(self, data: bytes) -> dict:
        return json.loads(data.decode('utf-8'))

    def serialize(self, data: dict) -> bytes:
        return json.dumps(data).encode('utf-8')

    def _compile_patterns(self):
        self.safety_pattern = re.compile(
            r"Safety:\\s*(Safe|Controversial|Unsafe)",
            re.IGNORECASE
        )
        category_pattern = "|".join(re.escape(cat) for cat in SAFETY_CATEGORIES)
        # ruff: noqa: F821
        self.category_pattern = re.compile(f"({{category_pattern}})", re.IGNORECASE)
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

    def forward(self, data: dict) -> dict:
        content = data.get("content") or data.get("input", "")
        prompt = self._format_prompt(content, data.get("context"), data.get("moderation_type", "prompt"))

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

        return {{
            "safety_level": parsed["safety_level"],
            "categories": parsed["categories"],
            "refusal": parsed.get("refusal"),
            "is_safe": parsed["safety_level"] == "Safe",
            "raw_output": parsed["raw_output"],
            "model": self.model_name,
        }}
'''

    return f"""
import os
import sys

{embedder_code}
{reranker_code}
{guard_code}

from mosec import Server, Runtime

PORT = {settings.server_port}

if __name__ == "__main__":
    server = Server()

    routes = {{}}

    # Register embedding endpoint
    if "EmbeddingWorker" in globals():
        emb = Runtime(EmbeddingWorker)
        routes["/v1/embeddings"] = [emb]
        routes["/embeddings"] = [emb]

    # Register reranker endpoint
    if "RerankerWorker" in globals():
        routes["/v1/rerank"] = [Runtime(RerankerWorker)]
        routes["/rerank"] = [Runtime(RerankerWorker)]

    # Register guard endpoint
    if "GuardWorker" in globals():
        routes["/v1/moderate"] = [Runtime(GuardWorker)]
        routes["/moderate"] = [Runtime(GuardWorker)]

    server.register_runtime(routes)

    server.run()
"""


class MosecServerManager:
    """Manages the combined Mosec server with dynamic model loading."""

    def __init__(self):
        self.pid_dir = PID_DIR
        self._script_dir = PID_DIR / "scripts"
        self._script_dir.mkdir(parents=True, exist_ok=True)

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

        # cli.py passes None when a model is not specified
        # If None, don't load that model (no fallback to .env)
        emb_model = embedding_model
        rer_model = reranker_model
        guard_m = guard_model

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

        logger.info(
            f"Starting server with models: emb={emb_model}, rer={rer_model}, guard={guard_m}"
        )

        server_script = _generate_server_script(settings, emb_model, rer_model, guard_m)
        script_path = self._script_dir / "mosec_server.py"
        with open(script_path, "w") as f:
            f.write(server_script)
        cmd = [sys.executable, str(script_path), "--port", str(settings.server_port)]
        env = os.environ.copy()

        # Pass HF_TOKEN to subprocess if set
        if settings.hf_token:
            env["HF_TOKEN"] = settings.hf_token

        try:
            if background:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                    env=env,
                )
            else:
                process = subprocess.Popen(cmd, env=env)

            loaded_models = {
                "embedding": emb_model,
                "reranker": rer_model,
                "guard": guard_m,
            }
            _save_server_pid(process.pid, settings.server_port, loaded_models)

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
        pid_info = _load_server_pid()
        if pid_info and "models" in pid_info:
            return pid_info["models"]

        # Fallback to .env for compatibility
        active = load_active_models()
        return {
            "embedding": active["embedding"],
            "reranker": active["reranker"],
            "guard": active["guard"],
        }

    def stop_all(self) -> bool:
        """Stop all servers (just stops the combined server)."""
        return self.stop()
