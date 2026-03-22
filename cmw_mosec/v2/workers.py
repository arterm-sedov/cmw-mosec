"""Worker classes with dynamic configuration lookup.

Workers fetch their model configuration at runtime from cmw-mosec's
ModelRegistry, following Mosec best practices.
"""

import base64
import json
import os
import re
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import transformers
from llmspec import EmbeddingData, EmbeddingRequest, EmbeddingResponse, TokenUsage
from mosec import ClientError, Worker

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _get_embedding_config():
    """Fetch embedding config from ModelRegistry at runtime."""
    from cmw_mosec.server_config import ModelRegistry

    model_slug = os.getenv("ACTIVE_EMBEDDING_MODEL", "")
    if not model_slug:
        raise ValueError("No embedding model configured (ACTIVE_EMBEDDING_MODEL not set)")

    registry = ModelRegistry()
    return registry.get_embedding_config(model_slug.lower())


def _get_reranker_config():
    """Fetch reranker config from ModelRegistry at runtime."""
    from cmw_mosec.server_config import ModelRegistry

    model_slug = os.getenv("ACTIVE_RERANKER_MODEL", "")
    if not model_slug:
        raise ValueError("No reranker model configured (ACTIVE_RERANKER_MODEL not set)")

    registry = ModelRegistry()
    return registry.get_reranker_config(model_slug.lower())


def _get_guard_config():
    """Fetch guard config from ModelRegistry at runtime."""
    from cmw_mosec.server_config import ModelRegistry

    model_slug = os.getenv("ACTIVE_GUARD_MODEL", "")
    if not model_slug:
        raise ValueError("No guard model configured (ACTIVE_GUARD_MODEL not set)")

    registry = ModelRegistry()
    return registry.get_guard_config(model_slug.lower())


class EmbeddingWorkerV2(Worker):
    """Embedding worker with dynamic configuration."""

    def __init__(self):
        model_slug = os.getenv("ACTIVE_EMBEDDING_MODEL", "")
        if not model_slug:
            return

        config = _get_embedding_config()

        self.model_name = config.model_id
        self.pooling = config.pooling
        self.dimensions = config.dimensions
        self.max_length = config.max_length
        self.model_class = config.model_class or "AutoModel"

        if self.pooling == "last_token":
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.model_name,
                padding_side="left",
            )
        else:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)

        if self.model_class == "T5EncoderModel":
            self.model = transformers.T5EncoderModel.from_pretrained(self.model_name)
        else:
            self.model = transformers.AutoModel.from_pretrained(self.model_name)

        self.device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.model.eval()
        self.dtype = config.dtype
        if self.dtype == "float16" and self.device != "cpu":
            self.model = self.model.half()
        elif self.dtype == "int8":
            self.model.quantized = True

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def cls_pooling(self, model_output):
        return model_output[0][:, 0, :]

    def last_token_pool(self, model_output, attention_mask):
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = model_output[0].shape[0]
        last_token_embeddings = model_output[0][
            torch.arange(batch_size, device=model_output[0].device), sequence_lengths
        ]
        return last_token_embeddings

    def get_embeddings(self, texts: str | list[str | list[int]], max_length: int = None):
        if isinstance(texts, str):
            texts = [texts]

        effective_max_length = max_length or self.max_length
        encoded_input = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=effective_max_length,
            return_tensors="pt",
        )
        inputs = encoded_input.to(self.device)
        with torch.no_grad():
            model_output = self.model(**inputs)

        if hasattr(self.model, "encoder") and self.pooling == "cls":
            sentence_embeddings = self.cls_pooling(model_output)
        elif self.pooling == "last_token":
            sentence_embeddings = self.last_token_pool(model_output, inputs["attention_mask"])
        elif self.pooling == "cls":
            sentence_embeddings = self.cls_pooling(model_output)
        else:
            sentence_embeddings = self.mean_pooling(model_output, inputs["attention_mask"])

        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        token_count = inputs["attention_mask"].sum(dim=1).tolist()[0]
        return token_count, sentence_embeddings

    def deserialize(self, data: bytes) -> EmbeddingRequest:
        raw = json.loads(data)
        dimensions = raw.get("dimensions")
        max_length = raw.get("max_length")
        req = EmbeddingRequest.from_bytes(data)
        req.dimensions = dimensions
        req.max_length = max_length
        return req

    def serialize(self, data: EmbeddingResponse) -> bytes:
        return data.to_json()

    def forward(self, data: EmbeddingRequest) -> EmbeddingResponse:
        if data.model != self.model_name:
            raise ClientError(
                f"the requested model {data.model} is not supported by "
                f"this worker {self.model_name}"
            )

        token_count, embeddings = self.get_embeddings(
            data.input, max_length=getattr(data, "max_length", None)
        )
        embeddings = embeddings.detach()
        if self.device != "cpu":
            embeddings = embeddings.cpu()

        requested_dim = getattr(data, "dimensions", None)
        if requested_dim is not None:
            if requested_dim < 1:
                raise ClientError(f"dimensions must be >= 1, got {requested_dim}")
            if self.dimensions is not None and requested_dim > self.dimensions:
                raise ClientError(
                    f"dimensions {requested_dim} exceeds model's max dimension {self.dimensions}"
                )
            embeddings = embeddings[:, :requested_dim]

        embeddings = embeddings.float().numpy()
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


class RerankerWorkerV2(Worker):
    """Base reranker worker with dynamic configuration."""

    def __init__(self):
        model_slug = os.getenv("ACTIVE_RERANKER_MODEL", "")
        if not model_slug:
            return

        config = _get_reranker_config()

        self.model_name = config.model_id
        self.max_length = config.max_length
        self.reranker_type = config.reranker_type
        self.scoring_method = config.scoring_method
        self.scoring_tokens = config.scoring_tokens or {}
        self.inference_batch_size = config.inference_batch_size

        if self.reranker_type == "llm_reranker":
            from transformers import AutoModelForCausalLM

            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.model_name, padding_side="left"
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            self.model.eval()

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            if self.scoring_tokens:
                self.token_true_id = self.tokenizer.convert_tokens_to_ids(
                    self.scoring_tokens.get("true", "yes")
                )
                if "false" in self.scoring_tokens:
                    self.token_false_id = self.tokenizer.convert_tokens_to_ids(
                        self.scoring_tokens["false"]
                    )
                else:
                    self.token_false_id = None
            else:
                self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
                self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        else:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(self.model_name)
            if self.model.tokenizer.pad_token is None:
                self.model.tokenizer.pad_token = self.model.tokenizer.eos_token

    def _compute_scores(self, query: str, docs: list, max_length: int) -> list:
        if self.reranker_type == "llm_reranker":
            batch_size = self.inference_batch_size
            all_scores = []

            for i in range(0, len(docs), batch_size):
                batch_docs = docs[i : i + batch_size]
                pairs = [query + doc for doc in batch_docs]

                inputs = self.tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=max_length,
                )
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

                with torch.no_grad():
                    batch_scores = self.model(**inputs).logits[:, -1, :]

                    if self.scoring_method == "softmax":
                        true_vector = batch_scores[:, self.token_true_id]
                        false_vector = batch_scores[:, self.token_false_id]
                        stacked = torch.stack([false_vector, true_vector], dim=1)
                        log_probs = torch.nn.functional.log_softmax(stacked, dim=1)
                        scores = log_probs[:, 1].exp().tolist()
                    else:
                        scores = batch_scores[:, self.token_true_id].tolist()

                all_scores.extend(scores)

            return all_scores
        else:
            scores = self.model.predict([[query, doc] for doc in docs])
            return scores.tolist()

    def deserialize(self, data: bytes) -> dict[str, Any]:
        return json.loads(data.decode("utf-8"))

    def serialize(self, data: dict[str, Any]) -> bytes:
        return json.dumps(data).encode("utf-8")

    def _get_query_and_docs(self, data: dict[str, Any]) -> tuple[str, list, int]:
        docs = data.get("docs") or data.get("documents")
        effective_max_length = data.get("max_length") or self.max_length
        query = data.get("query", "")
        return query, docs, effective_max_length


class ScoreWorkerV2(RerankerWorkerV2):
    """Worker for /v2/score endpoint - vLLM format."""

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        query = data.get("queries", "")
        if isinstance(query, list):
            query = query[0] if query else ""
        docs = data.get("documents", [])
        max_length = data.get("max_length") or self.max_length

        scores = self._compute_scores(query, docs, max_length)

        return {
            "data": [
                {"index": i, "object": "score", "score": float(s)} for i, s in enumerate(scores)
            ]
        }


class RerankWorkerV2(RerankerWorkerV2):
    """Worker for /v2/rerank endpoint - Cohere format."""

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        query, docs, max_length = self._get_query_and_docs(data)
        scores = self._compute_scores(query, docs, max_length)

        top_n = data.get("top_n")
        indexed_scores = list(enumerate(zip(docs, scores, strict=False)))
        indexed_scores.sort(key=lambda x: x[1][1], reverse=True)

        if top_n is not None:
            indexed_scores = indexed_scores[:top_n]

        return {
            "results": [
                {
                    "index": i,
                    "document": {"text": doc},
                    "relevance_score": float(score),
                }
                for i, (doc, score) in indexed_scores
            ]
        }


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


class GuardWorkerV2(Worker):
    """Guard worker with dynamic configuration."""

    def __init__(self):
        model_slug = os.getenv("ACTIVE_GUARD_MODEL", "")
        if not model_slug:
            return

        config = _get_guard_config()

        self.model_name = config.model_id
        self.max_new_tokens = config.max_new_tokens
        self.max_length = config.max_length
        self.dtype = config.dtype

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.float16 if self.dtype == "float16" else torch.bfloat16,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        self.model.eval()
        self._compile_patterns()

    def deserialize(self, data: bytes) -> dict:
        return json.loads(data.decode("utf-8"))

    def serialize(self, data: dict) -> bytes:
        return json.dumps(data).encode("utf-8")

    def _compile_patterns(self):
        self.safety_pattern = re.compile(
            r"Safety:\s*(Safe|Controversial|Unsafe)",
            re.IGNORECASE,
        )
        category_pattern = "|".join(re.escape(cat) for cat in SAFETY_CATEGORIES)
        self.category_pattern = re.compile(f"({category_pattern})", re.IGNORECASE)
        self.refusal_pattern = re.compile(r"Refusal:\s*(Yes|No)", re.IGNORECASE)

    def _format_prompt(self, content: str, context: str | None, moderation_type: str) -> str:
        if moderation_type == "response" and context:
            messages = [
                {"role": "user", "content": context},
                {"role": "assistant", "content": content},
            ]
        else:
            messages = [{"role": "user", "content": content}]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _parse_output(self, output: str) -> dict:
        result = {
            "safety_level": "Unknown",
            "categories": [],
            "refusal": None,
            "raw_output": output,
        }

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
        prompt = self._format_prompt(
            content, data.get("context"), data.get("moderation_type", "prompt")
        )
        effective_max_length = data.get("max_length") or self.max_length

        model_inputs = self.tokenizer(
            [prompt],
            return_tensors="pt",
            truncation=True,
            max_length=effective_max_length,
        )

        if torch.cuda.is_available():
            model_inputs = {k: v.cuda() for k, v in model_inputs.items()}

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        output_ids = generated_ids[0][len(model_inputs["input_ids"][0]) :].tolist()
        output_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)

        parsed = self._parse_output(output_text)

        return {
            "safety_level": parsed["safety_level"],
            "categories": parsed["categories"],
            "refusal": parsed.get("refusal"),
            "is_safe": parsed["safety_level"] == "Safe",
            "raw_output": parsed["raw_output"],
            "model": self.model_name,
        }
