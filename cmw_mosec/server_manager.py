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

    guard_max_new_tokens = None
    guard_max_length = None
    if guard_model:
        try:
            guard_config = ModelRegistry().get_guard_config(guard_model)
            guard_max_new_tokens = guard_config.max_new_tokens
            guard_max_length = guard_config.max_length
        except ValueError:
            pass
        if guard_max_new_tokens is None:
            raise ValueError(f"max_new_tokens not configured for guard model {guard_model}")
        if guard_max_length is None:
            raise ValueError(f"max_length not configured for guard model {guard_model}")

    # Get pooling config for embedding model
    pooling_method = None
    embed_dtype = None
    model_class = None
    embed_dimensions = None
    embed_max_length = None
    if embedding_model:
        try:
            from .server_config import ModelRegistry

            registry = ModelRegistry()
            config_dict = registry._embeddings.get(embedding_model.lower(), {})
            pooling_method = config_dict.get("pooling")
            embed_dtype = config_dict.get("dtype")
            model_class = config_dict.get("model_class")
            embed_dimensions = config_dict.get("dimensions")
            embed_max_length = config_dict.get("max_length")
        except Exception:
            pass
        if pooling_method is None:
            raise ValueError(f"pooling not configured for embedding model {embedding_model}")
        if embed_dtype is None:
            raise ValueError(f"dtype not configured for embedding model {embedding_model}")
        if embed_max_length is None:
            raise ValueError(f"max_length not configured for embedding model {embedding_model}")
        if model_class is None:
            model_class = "AutoModel"  # sensible default

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
DIMENSIONS = {embed_dimensions}  # Native dimension, None means use model's full dimension
MAX_LENGTH = {embed_max_length}  # Max context length for tokenization


class EmbeddingWorker(Worker):
    def __init__(self):
        self.model_name = EMBEDDING_MODEL
        self.pooling = POOLING
        self.dimensions = DIMENSIONS
        self.max_length = MAX_LENGTH

        # LLM-based embedders (Qwen3) need left padding for last_token pooling
        # Encoder-based (FRIDA) use default right padding
        if self.pooling == "last_token":
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.model_name,
                padding_side='left'
            )
        else:
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

    def get_embeddings(self, texts: Union[str, List[Union[str, List[int]]]], max_length: int = None):
        # Handle both single string and list of strings
        if isinstance(texts, str):
            texts = [texts]

        effective_max_length = max_length or MAX_LENGTH
        encoded_input = self.tokenizer(
            texts, padding=True, truncation=True, max_length=effective_max_length, return_tensors="pt"
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
        # Extract client-controllable params from raw JSON before llmspec loses them
        # llmspec's EmbeddingRequest doesn't have dimensions or max_length fields
        import json
        raw = json.loads(data)
        dimensions = raw.get('dimensions')  # OpenAI API parameter for MRL
        max_length = raw.get('max_length')  # Optional: override config max_length
        req = EmbeddingRequest.from_bytes(data)
        # Store as dynamic attributes
        req.dimensions = dimensions
        req.max_length = max_length
        return req

    def serialize(self, data: EmbeddingResponse) -> bytes:
        # llmspec's to_json() already returns bytes
        return data.to_json()

    def forward(self, data: EmbeddingRequest) -> EmbeddingResponse:
        if data.model != self.model_name:
            raise ClientError(
                f"the requested model {{data.model}} is not supported by "
                f"this worker {{self.model_name}}"
            )

        token_count, embeddings = self.get_embeddings(data.input, max_length=getattr(data, 'max_length', None))
        embeddings = embeddings.detach()
        if self.device != "cpu":
            embeddings = embeddings.cpu()

        # MRL dimension truncation (Matryoshka Representation Learning)
        # If dimensions parameter is provided, truncate to that dimension
        requested_dim = getattr(data, 'dimensions', None)
        if requested_dim is not None:
            if requested_dim < 1:
                raise ClientError(f"dimensions must be >= 1, got {{requested_dim}}")
            if self.dimensions is not None and requested_dim > self.dimensions:
                raise ClientError(
                    f"dimensions {{requested_dim}} exceeds model's max dimension {{self.dimensions}}"
                )
            # Truncate to requested dimension
            embeddings = embeddings[:, :requested_dim]

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
        # Get config values at script generation time
        from .server_config import ModelRegistry

        registry = ModelRegistry()
        config = registry._rerankers.get(reranker_model.lower(), {})

        reranker_max_length = config.get("max_length")
        if reranker_max_length is None:
            raise ValueError(f"max_length not configured for reranker model {reranker_model}")

        reranker_type = config.get("reranker_type", "cross_encoder")
        scoring_method = config.get("scoring_method")
        scoring_tokens = config.get("scoring_tokens", {})

        # Format scoring_method for embedding in generated code
        scoring_method_str = f'"{scoring_method}"' if scoring_method else "None"

        # Format scoring_tokens for embedding in generated code
        scoring_tokens_str = "None"
        if scoring_tokens:
            scoring_tokens_str = str(scoring_tokens)

        reranker_code = f'''
import json
import os
import torch
from typing import Any, List

from mosec import Worker

os.environ["TOKENIZERS_PARALLELISM"] = "false"

RERANKER_MODEL = "{reranker_model}"
MAX_LENGTH = {reranker_max_length}
RERANKER_TYPE = "{reranker_type}"
SCORING_METHOD = {scoring_method_str}
SCORING_TOKENS = {scoring_tokens_str}


class RerankerWorker(Worker):
    def __init__(self):
        self.model_name = RERANKER_MODEL
        self.max_length = MAX_LENGTH
        self.reranker_type = RERANKER_TYPE
        self.scoring_method = SCORING_METHOD
        self.scoring_tokens = SCORING_TOKENS

        if self.reranker_type == "llm_reranker":
            # Use AutoModelForCausalLM for LLM-based rerankers
            # Models like Qwen3-Reranker, BGE-Gemma that use language models
            from transformers import AutoTokenizer, AutoModelForCausalLM
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side='left')
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto" if torch.cuda.is_available() else None
            )
            self.model.eval()

            # Set pad token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Get scoring token IDs from config
            # For softmax: scoring_tokens has 'true' and 'false' keys
            # For raw_logit: scoring_tokens has only 'true' key
            if self.scoring_tokens:
                self.token_true_id = self.tokenizer.convert_tokens_to_ids(self.scoring_tokens.get("true", "yes"))
                if "false" in self.scoring_tokens:
                    self.token_false_id = self.tokenizer.convert_tokens_to_ids(self.scoring_tokens["false"])
                else:
                    self.token_false_id = None
            else:
                # Default to yes/no for backward compatibility
                self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
                self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        else:
            # cross_encoder: Use sentence_transformers CrossEncoder
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name)
            if self.model.tokenizer.pad_token is None:
                self.model.tokenizer.pad_token = self.model.tokenizer.eos_token

    def _compute_scores(self, query: str, docs: list, max_length: int) -> list:
        """Compute relevance scores for query-document pairs."""
        if self.reranker_type == "llm_reranker":
            # LLM reranker: client sends pre-formatted query and documents
            # Server pairs them: query + doc for each document
            pairs = [query + doc for doc in docs]

            inputs = self.tokenizer(
                pairs, padding=True, truncation=True,
                return_tensors="pt", max_length=max_length
            )
            inputs = {{k: v.to(self.model.device) for k, v in inputs.items()}}

            with torch.no_grad():
                batch_scores = self.model(**inputs).logits[:, -1, :]

                if self.scoring_method == "softmax":
                    # Softmax over [false_token, true_token], return probability of true
                    true_vector = batch_scores[:, self.token_true_id]
                    false_vector = batch_scores[:, self.token_false_id]
                    stacked = torch.stack([false_vector, true_vector], dim=1)
                    log_probs = torch.nn.functional.log_softmax(stacked, dim=1)
                    scores = log_probs[:, 1].exp().tolist()
                else:
                    # raw_logit: return raw logit for true token
                    scores = batch_scores[:, self.token_true_id].tolist()

            return scores
        else:
            # cross_encoder: standard sentence_transformers approach
            original_max_length = self.model.tokenizer.model_max_length
            if max_length != original_max_length:
                self.model.tokenizer.model_max_length = max_length
            try:
                scores = self.model.predict([[query, doc] for doc in docs])
                return scores.tolist()
            finally:
                self.model.tokenizer.model_max_length = original_max_length

    def deserialize(self, data: bytes) -> dict[str, Any]:
        return json.loads(data.decode("utf-8"))

    def serialize(self, data: dict[str, Any]) -> bytes:
        return json.dumps(data).encode("utf-8")

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        """Forward pass for reranker endpoint.

        Supports two formats detected by request parameters:
        - vLLM score format: "queries" parameter (returns data array)
        - Cohere format: "query" + "docs" (returns results array)

        Single model instance handles both output formats.
        """
        docs = data.get("docs") or data.get("documents")
        effective_max_length = data.get("max_length") or self.max_length

        # vLLM score format: uses "queries" parameter
        if "queries" in data:
            query = data["queries"]
            if isinstance(query, list):
                query = query[0] if query else ""
            scores = self._compute_scores(query, docs, effective_max_length)
            return {{
                "data": [
                    {{"index": i, "object": "score", "score": float(s)}}
                    for i, s in enumerate(scores)
                ]
            }}

        # Cohere format: uses "query" parameter
        query = data.get("query", "")
        scores = self._compute_scores(query, docs, effective_max_length)

        # Cohere format (standard rerank request)
        top_n = data.get("top_n")
        indexed_scores = list(enumerate(zip(docs, scores)))
        indexed_scores.sort(key=lambda x: x[1][1], reverse=True)

        if top_n is not None:
            indexed_scores = indexed_scores[:top_n]

        return {{"results": [
            {{"index": i, "document": {{"text": doc}}, "relevance_score": float(score)}}
            for i, (doc, score) in indexed_scores
        ]}}


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
MAX_LENGTH = {guard_max_length}

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
        effective_max_length = data.get("max_length") or MAX_LENGTH

        model_inputs = self.tokenizer(
            [prompt],
            return_tensors="pt",
            truncation=True,
            max_length=effective_max_length
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
        routes["/v1/embeddings"] = [Runtime(EmbeddingWorker)]

    # Register reranker endpoints (single worker, single memory allocation)
    # RerankerWorker.forward() handles both vLLM score and Cohere formats
    if "RerankerWorker" in globals():
        reranker = Runtime(RerankerWorker)
        routes["/v1/rerank"] = [reranker]
        routes["/v1/score"] = [reranker]

    # Register guard endpoint
    if "GuardWorker" in globals():
        routes["/v1/moderate"] = [Runtime(GuardWorker)]

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
