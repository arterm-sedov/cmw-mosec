# Qwen3-Reranker-0.6B MOSEC Server Implementation Plan

## Overview
High-performance MOSEC server for Qwen3-Reranker-0.6B, a state-of-the-art 0.6B parameter instruction-aware reranker model supporting 100+ languages and 32K context length.

## Model Characteristics

### Qwen3-Reranker-0.6B Specifications
- **Architecture**: CausalLM (not traditional cross-encoder)
- **Parameters**: 0.6B (278M active)
- **Context Length**: 32K tokens
- **Input Format**: Instruction-aware chat format
- **Output Format**: Binary yes/no logits (requires sigmoid for probabilities)
- **Special Requirements**: transformers >= 4.51.0

### Prompt Format
```
<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".
<|im_start|>user
<Instruct>: {instruction}
<Query>: {query}
<Document>: {doc}
<|im_start|>assistant
The answer is:
```

## Project Structure

```
cmw-qwen3-reranker/
├── pyproject.toml                  # Dependencies and project metadata
├── README.md                       # Comprehensive documentation
├── LICENSE                         # MIT license
├── Dockerfile                      # Production deployment
├── docker-compose.yml              # Multi-service orchestration
├── .gitignore                      # Exclude model cache
├── .env-example                    # Environment variable template
├── cmw_qwen3_reranker/             # Python package
│   ├── __init__.py
│   ├── worker.py                   # MOSEC Worker with Qwen3
│   ├── server.py                   # Server configuration
│   ├── config.py                   # Settings and validation
│   ├── formatter.py                # Prompt formatting utilities
│   ├── cli.py                      # Click CLI interface
│   └── models/                     # Model loading utilities
├── tests/                          # Unit and integration tests
│   ├── test_worker.py
│   ├── test_formatter.py
│   └── test_integration.py
├── scripts/                        # Utility scripts
│   ├── download_model.py           # Model download script
│   ├── benchmark.py                # Performance benchmarking
│   └── health_check.py             # Health monitoring
└── docs/                           # Additional documentation
    ├── API.md
    └── DEPLOYMENT.md
```

## Implementation Phases

### Phase 1: Core Implementation (Priority: High)
**Timeline: 3-4 hours**

#### 1.1 Qwen3 Reranker Worker

```python
# cmw_qwen3_reranker/worker.py
import os
import torch
from typing import List, Dict, Any
from mosec import Server, Worker, Runtime
from mosec.mixin import MsgpackMixin, TypedMsgPackMixin
from transformers import AutoModelForCausalLM, AutoTokenizer
from msgspec import Struct
import logging

logger = logging.getLogger(__name__)

# Qwen3 prompt templates
SYSTEM_PROMPT = """Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."""

DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"

class RerankRequest(Struct, kw_only=True):
    """Request structure for reranking."""
    query: str
    documents: List[str]
    instruction: str = DEFAULT_INSTRUCTION
    top_k: int = 10

class RerankResponse(Struct, kw_only=True):
    """Response structure with scores."""
    scores: List[float]
    model: str = "Qwen/Qwen3-Reranker-0.6B"

class Qwen3RerankerWorker(MsgpackMixin, Worker):
    """MOSEC Worker for Qwen3-Reranker-0.6B.
    
    Implements the instruction-aware reranking logic with dynamic batching
    for optimal throughput.
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen3-Reranker-0.6B"):
        """Initialize Qwen3 reranker with model and tokenizer.
        
        Args:
            model_name: HuggingFace model identifier
        """
        self.model_name = model_name
        
        # Load tokenizer with left padding for generation
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="left",
            trust_remote_code=True
        )
        
        # Load model with optimal settings
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        # Enable flash attention if available
        if hasattr(self.model.config, "attn_implementation"):
            self.model.config.attn_implementation = "flash_attention_2"
        
        self.model.eval()
        
        # Token IDs for yes/no classification
        self.token_yes_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.token_no_id = self.tokenizer.convert_tokens_to_ids("no")
        
        # Build prompt templates
        self._build_prompt_templates()
        
        logger.info(f"Loaded Qwen3-Reranker model: {model_name}")
    
    def _build_prompt_templates(self):
        """Build chat template components."""
        self.system_message = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
    
    def format_prompt(self, instruction: str, query: str, document: str) -> str:
        """Format instruction-aware prompt for Qwen3.
        
        Args:
            instruction: Task-specific instruction
            query: User query
            document: Document to evaluate
            
        Returns:
            Formatted prompt string
        """
        user_content = f"""<Instruct>: {instruction}
<Query>: {query}
<Document>: {document}"""
        
        messages = self.system_message + [
            {"role": "user", "content": user_content}
        ]
        
        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Add assistant prefix for scoring
        prompt += "The answer is:"
        
        return prompt
    
    def forward(self, data: List[Dict[str, Any]]) -> List[float]:
        """Process batch of reranking requests.
        
        MOSEC aggregates individual requests into a list for dynamic batching.
        Each item should contain: query, document, instruction (optional).
        
        Args:
            data: List of request dictionaries
            
        Returns:
            List of relevance scores (0-1 probabilities)
        """
        # Format prompts for all query-document pairs
        prompts = []
        for item in data:
            instruction = item.get("instruction", DEFAULT_INSTRUCTION)
            query = item["query"]
            document = item["document"]
            
            prompt = self.format_prompt(instruction, query, document)
            prompts.append(prompt)
        
        # Tokenize with truncation
        max_length = min(8192, self.model.config.max_position_embeddings)
        
        inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # Move to device
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # Extract logits for last token
            last_token_logits = outputs.logits[:, -1, :]
            
            # Get yes/no logits
            yes_logits = last_token_logits[:, self.token_yes_id]
            no_logits = last_token_logits[:, self.token_no_id]
            
            # Compute probabilities using softmax
            probs = torch.softmax(
                torch.stack([no_logits, yes_logits], dim=-1),
                dim=-1
            )
            
            # Return yes probability (relevance score)
            relevance_scores = probs[:, 1].tolist()
        
        return relevance_scores
```

#### 1.2 Server Configuration

```python
# cmw_qwen3_reranker/server.py
from mosec import Server, Runtime
from .worker import Qwen3RerankerWorker
from .config import ServerConfig
import logging

logger = logging.getLogger(__name__)

def create_server(config: ServerConfig) -> Server:
    """Create MOSEC server optimized for Qwen3 reranker.
    
    Configures dynamic batching and worker processes for optimal
    GPU/CPU utilization.
    
    Args:
        config: Server configuration
        
    Returns:
        Configured MOSEC server
    """
    server = Server(
        # Enable Prometheus metrics
        enable_metrics=True,
        # Configure timeouts
        timeout=config.timeout,
        # HTTP/2 support
        http2=True
    )
    
    # Create worker runtime
    reranker_runtime = Runtime(Qwen3RerankerWorker)
    
    # Register endpoint
    server.register_runtime({
        "/rerank": [reranker_runtime],
        "/v1/rerank": [reranker_runtime],  # OpenAI-compatible
    })
    
    # Configure worker with dynamic batching
    server.append_worker(
        Qwen3RerankerWorker,
        num=config.workers,
        max_batch_size=config.max_batch_size,
        max_wait_time=config.max_wait_time
    )
    
    logger.info(
        f"Qwen3 Reranker Server configured: "
        f"workers={config.workers}, "
        f"batch_size={config.max_batch_size}, "
        f"wait_time={config.max_wait_time}ms"
    )
    
    return server
```

#### 1.3 Prompt Formatter Utilities

```python
# cmw_qwen3_reranker/formatter.py
from typing import List, Dict, Optional

class Qwen3PromptFormatter:
    """Utility class for formatting Qwen3 reranker prompts."""
    
    DEFAULT_INSTRUCTION = (
        "Given a web search query, retrieve relevant passages that answer the query"
    )
    
    def __init__(self, tokenizer):
        """Initialize formatter with tokenizer.
        
        Args:
            tokenizer: HuggingFace tokenizer instance
        """
        self.tokenizer = tokenizer
        self.system_prompt = """Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."""
    
    def format_single(
        self,
        query: str,
        document: str,
        instruction: Optional[str] = None
    ) -> str:
        """Format a single query-document pair.
        
        Args:
            query: Search query
            document: Document to evaluate
            instruction: Task instruction (optional)
            
        Returns:
            Formatted prompt string
        """
        instruction = instruction or self.DEFAULT_INSTRUCTION
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": f"""<Instruct>: {instruction}
<Query>: {query}
<Document>: {document}"""
            }
        ]
        
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        return prompt + "The answer is:"
    
    def format_batch(
        self,
        query: str,
        documents: List[str],
        instruction: Optional[str] = None
    ) -> List[str]:
        """Format multiple documents for a single query.
        
        Args:
            query: Search query
            documents: List of documents to evaluate
            instruction: Task instruction (optional)
            
        Returns:
            List of formatted prompts
        """
        return [
            self.format_single(query, doc, instruction)
            for doc in documents
        ]
    
    def validate_instruction(self, instruction: str) -> str:
        """Validate and clean instruction text.
        
        Args:
            instruction: Raw instruction string
            
        Returns:
            Cleaned instruction string
        """
        # Remove excessive whitespace
        cleaned = " ".join(instruction.split())
        
        # Ensure instruction is not too long (model limit)
        max_chars = 500
        if len(cleaned) > max_chars:
            cleaned = cleaned[:max_chars] + "..."
        
        return cleaned
```

#### 1.4 CLI Interface

```python
# cmw_qwen3_reranker/cli.py
import click
import os
from pathlib import Path
from .server import create_server
from .config import ServerConfig

@click.group()
@click.version_option(version="0.1.0")
def cli():
    """CMW Qwen3 Reranker - High-performance reranker server for Qwen3 models.
    
    Serves Qwen3-Reranker-0.6B with MOSEC dynamic batching and instruction-aware
    prompting for optimal multilingual reranking performance.
    """
    pass

@cli.command()
@click.option(
    "--model", "-m",
    default="Qwen/Qwen3-Reranker-0.6B",
    help="HuggingFace model identifier (default: Qwen/Qwen3-Reranker-0.6B)"
)
@click.option(
    "--port", "-p",
    default=8080,
    help="Server port (default: 8080)"
)
@click.option(
    "--workers", "-w",
    default=1,
    help="Number of worker processes (default: 1 for GPU, 4 for CPU)"
)
@click.option(
    "--batch-size", "-b",
    default=8,
    help="Max batch size for dynamic batching (default: 8)"
)
@click.option(
    "--max-wait-time",
    default=50,
    help="Max wait time for batching in milliseconds (default: 50)"
)
@click.option(
    "--timeout",
    default=30000,
    help="Request timeout in milliseconds (default: 30000)"
)
@click.option(
    "--log-level",
    type=click.Choice(["debug", "info", "warning", "error"]),
    default="info",
    help="Logging level (default: info)"
)
@click.option(
    "--cache-dir",
    type=click.Path(),
    help="Directory to cache downloaded models"
)
def serve(
    model: str,
    port: int,
    workers: int,
    batch_size: int,
    max_wait_time: int,
    timeout: int,
    log_level: str,
    cache_dir: str
):
    """Start Qwen3 reranker server."""
    # Set environment variables
    os.environ["LOG_LEVEL"] = log_level.upper()
    if cache_dir:
        os.environ["HF_HOME"] = str(cache_dir)
    
    # Auto-adjust workers for CPU vs GPU
    import torch
    if not torch.cuda.is_available() and workers == 1:
        workers = 4
        click.echo(f"Auto-adjusted workers to {workers} for CPU inference")
    
    config = ServerConfig(
        model_name=model,
        port=port,
        workers=workers,
        max_batch_size=batch_size,
        max_wait_time=max_wait_time,
        timeout=timeout
    )
    
    server = create_server(config)
    server.run()

@cli.command()
@click.option(
    "--model",
    default="Qwen/Qwen3-Reranker-0.6B",
    help="Model to download"
)
@click.option(
    "--cache-dir",
    type=click.Path(),
    required=True,
    help="Directory to save model"
)
def download(model: str, cache_dir: str):
    """Download Qwen3 reranker model for offline use."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    click.echo(f"Downloading {model} to {cache_dir}...")
    
    os.environ["HF_HOME"] = cache_dir
    
    # Download tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    model_obj = AutoModelForCausalLM.from_pretrained(
        model,
        trust_remote_code=True
    )
    
    click.echo(f"✓ Model downloaded successfully to {cache_dir}")

@cli.command()
@click.option(
    "--endpoint",
    default="http://localhost:8080",
    help="Server endpoint URL"
)
@click.option(
    "--concurrent",
    "-c",
    default=10,
    help="Number of concurrent clients"
)
@click.option(
    "--duration",
    "-d",
    default=30,
    help="Test duration in seconds"
)
def benchmark(endpoint: str, concurrent: int, duration: int):
    """Run performance benchmark against server."""
    import subprocess
    
    cmd = [
        "python", "-m", "scripts.benchmark",
        "--endpoint", endpoint,
        "--concurrent", str(concurrent),
        "--duration", str(duration)
    ]
    
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    cli()
```

### Phase 2: Configuration and Dependencies

#### 2.1 pyproject.toml

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "cmw-qwen3-reranker"
version = "0.1.0"
description = "High-performance MOSEC server for Qwen3-Reranker-0.6B with instruction-aware reranking"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.10"
authors = [
    {name = "CMW Team"},
]
keywords = [
    "mosec", "qwen3", "reranker", "multilingual", 
    "text-ranking", "instruction-aware", "transformers", "causal-lm"
]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Software Development :: Libraries :: Python Modules",
]

dependencies = [
    # Core MOSEC
    "mosec>=0.8.0",
    
    # Qwen3 model dependencies (critical: >= 4.51.0)
    "transformers>=4.51.0",
    "torch>=2.0.0",
    "accelerate>=0.20.0",
    
    # Optional optimizations
    "flash-attn>=2.0.0; sys_platform != 'darwin'",  # CUDA only
    "bitsandbytes>=0.41.0",  # Quantization support
    
    # API and utilities
    "fastapi>=0.104.0",
    "uvicorn[standard]>=0.24.0",
    "pydantic>=2.0",
    "click>=8.0",
    "requests>=2.30",
    "msgspec>=0.18.0",  # Fast serialization
    "numpy>=1.24.0",
    
    # Monitoring
    "prometheus-client>=0.17.0",
    "structlog>=23.0.0",  # Structured logging
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.0.0",
    "ruff>=0.1.0",
    "black>=23.0.0",
    "mypy>=1.5.0",
    "httpx>=0.25.0",  # For testing
]

[project.scripts]
cmw-qwen3-reranker = "cmw_qwen3_reranker.cli:cli"

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP", "B", "C4", "SIM"]
ignore = ["E501"]

[tool.ruff.lint.pydocstyle]
convention = "google"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v --tb=short"
asyncio_mode = "auto"
```

#### 2.2 Docker Configuration

```dockerfile
# Multi-stage build for production
FROM nvidia/cuda:12.1-devel-ubuntu22.04 as base

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set Python3 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Production stage
FROM nvidia/cuda:12.1-runtime-ubuntu22.04 as production

# Install Python
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Copy Python packages from base
COPY --from=base /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages

WORKDIR /app

# Install dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e "."

# Copy application code
COPY cmw_qwen3_reranker/ ./cmw_qwen3_reranker/
COPY scripts/ ./scripts/

# Create model cache directory
RUN mkdir -p /app/models

# Environment configuration
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/models
ENV TRANSFORMERS_CACHE=/app/models
ENV CUDA_VISIBLE_DEVICES=0

# Expose ports
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/metrics')" || exit 1

# Start server
CMD ["cmw-qwen3-reranker", "serve", \
     "--model", "Qwen/Qwen3-Reranker-0.6B", \
     "--port", "8080", \
     "--workers", "1", \
     "--batch-size", "8"]
```

### Phase 3: Testing Protocol

#### 3.1 cURL Testing Commands

```bash
# Test basic reranking (default instruction)
curl -X POST http://localhost:8080/rerank \
     -H "Content-Type: application/json" \
     -d '{
       "query": "What is machine learning?",
       "documents": [
         "Machine learning is a subset of artificial intelligence.",
         "The weather today is sunny and warm.",
         "Deep learning uses neural networks for pattern recognition."
       ],
       "top_k": 2
     }'

# Test with custom instruction
curl -X POST http://localhost:8080/rerank \
     -H "Content-Type: application/json" \
     -d '{
       "query": "Как приготовить борщ?",
       "documents": [
         "Борщ — это традиционный свекольный суп.",
         "Вчера была хорошая погода для прогулки.",
         "Машинное обучение использует нейронные сети."
       ],
       "instruction": "Найти рецепты приготовления блюд",
       "top_k": 1
     }'

# Check health endpoint
curl http://localhost:8080/metrics

# Test with msgpack (for production)
curl -X POST http://localhost:8080/rerank \
     -H "Content-Type: application/msgpack" \
     --data-binary @request.msgpack
```

#### 3.2 Integration Test Script

```python
# tests/test_integration.py
import pytest
import requests
import time
from typing import List

# Test configuration
ENDPOINT = "http://localhost:8080"
TIMEOUT = 30

def test_basic_reranking():
    """Test basic reranking functionality."""
    response = requests.post(
        f"{ENDPOINT}/rerank",
        json={
            "query": "machine learning",
            "documents": [
                "Machine learning is a field of AI.",
                "Cooking recipes for beginners.",
                "Neural networks and deep learning."
            ],
            "top_k": 2
        },
        timeout=TIMEOUT
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert "scores" in data
    assert len(data["scores"]) == 3
    assert all(0 <= s <= 1 for s in data["scores"])
    
    # ML-related docs should score higher
    assert data["scores"][0] > data["scores"][1]
    assert data["scores"][2] > data["scores"][1]

def test_custom_instruction():
    """Test reranking with custom instruction."""
    response = requests.post(
        f"{ENDPOINT}/rerank",
        json={
            "query": "Python programming",
            "documents": [
                "Python is a programming language.",
                "Snakes are reptiles found in tropical regions."
            ],
            "instruction": "Find programming tutorials and documentation",
            "top_k": 1
        },
        timeout=TIMEOUT
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Programming doc should score higher
    assert data["scores"][0] > data["scores"][1]

def test_multilingual():
    """Test multilingual reranking."""
    test_cases = [
        {
            "query": "машинное обучение",
            "documents": [
                "Машинное обучение и искусственный интеллект.",
                "Приготовление кофе и рецепты выпечки."
            ],
            "lang": "russian"
        },
        {
            "query": "machine learning",
            "documents": [
                "Machine learning is a subset of AI.",
                "Weather forecast for tomorrow."
            ],
            "lang": "english"
        }
    ]
    
    for case in test_cases:
        response = requests.post(
            f"{ENDPOINT}/rerank",
            json={
                "query": case["query"],
                "documents": case["documents"],
                "top_k": 1
            },
            timeout=TIMEOUT
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # First document should be more relevant
        assert data["scores"][0] > data["scores"][1]

def test_performance():
    """Test performance under load."""
    import concurrent.futures
    
    def make_request(i):
        start = time.time()
        response = requests.post(
            f"{ENDPOINT}/rerank",
            json={
                "query": f"test query {i}",
                "documents": ["doc1", "doc2", "doc3"],
                "top_k": 2
            },
            timeout=TIMEOUT
        )
        latency = time.time() - start
        return response.status_code == 200, latency
    
    # Run 10 concurrent requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(make_request, i) for i in range(10)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    
    success_rate = sum(1 for success, _ in results if success) / len(results)
    avg_latency = sum(lat for _, lat in results) / len(results)
    
    assert success_rate >= 0.9, f"Success rate too low: {success_rate}"
    assert avg_latency < 2.0, f"Average latency too high: {avg_latency}s"
```

### Phase 4: Performance Benchmarks

#### 4.1 Expected Performance Metrics

| Metric | Qwen3-0.6B (GPU) | Qwen3-0.6B (CPU) | DiTy (CPU) |
|---------|-----------------|------------------|------------|
| **Throughput** | 25-30 req/s | 5-8 req/s | 15 req/s |
| **Latency (p95)** | 50ms | 200ms | 85ms |
| **Memory** | 2.5GB | 1.2GB | 800MB |
| **Batch Efficiency** | High | Medium | High |

#### 4.2 Optimization Strategies

```python
# cmw_qwen3_reranker/optimizations.py
import torch
from typing import Optional

class OptimizationConfig:
    """Configuration for model optimizations."""
    
    @staticmethod
    def apply_flash_attention(model):
        """Enable Flash Attention 2 for faster inference."""
        if hasattr(model.config, "attn_implementation"):
            model.config.attn_implementation = "flash_attention_2"
        return model
    
    @staticmethod
    def quantize_model(model, bits: int = 8):
        """Apply quantization for reduced memory."""
        from transformers import BitsAndBytesConfig
        
        if bits == 8:
            config = BitsAndBytesConfig(load_in_8bit=True)
        elif bits == 4:
            config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
        else:
            return model
        
        # Note: Requires model reload with config
        return model
    
    @staticmethod
    def optimize_torch_settings():
        """Apply PyTorch optimizations."""
        # Enable TF32 for better performance on Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Enable cudnn benchmarking
        torch.backends.cudnn.benchmark = True
```

### Phase 5: Deployment Strategy

#### 5.1 Docker Compose Configuration

```yaml
# docker-compose.yml
version: '3.8'

services:
  qwen3-reranker:
    build: .
    ports:
      - "8080:8080"
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - HF_HOME=/app/models
      - LOG_LEVEL=info
    volumes:
      - ./models:/app/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "import requests; requests.get('http://localhost:8080/metrics')"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    depends_on:
      - qwen3-reranker

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-storage:/var/lib/grafana
    depends_on:
      - prometheus

volumes:
  grafana-storage:
```

#### 5.2 Kubernetes Deployment

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: qwen3-reranker
  labels:
    app: qwen3-reranker
spec:
  replicas: 2
  selector:
    matchLabels:
      app: qwen3-reranker
  template:
    metadata:
      labels:
        app: qwen3-reranker
    spec:
      containers:
      - name: reranker
        image: cmw-qwen3-reranker:latest
        ports:
        - containerPort: 8080
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "4Gi"
          requests:
            memory: "2Gi"
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
        - name: WORKERS
          value: "1"
        - name: BATCH_SIZE
          value: "8"
        livenessProbe:
          httpGet:
            path: /metrics
            port: 8080
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /metrics
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: qwen3-reranker-service
spec:
  selector:
    app: qwen3-reranker
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
  type: LoadBalancer
```

### Phase 6: RAG Integration

#### 6.1 Integration with Existing RAG System

```python
# Update rag_engine/retrieval/reranker.py

class Qwen3RerankerAdapter(HTTPClientMixin):
    """Adapter for Qwen3 MOSEC reranker."""
    
    def __init__(self, endpoint: str, default_instruction: Optional[str] = None):
        super().__init__(endpoint=endpoint, timeout=60.0, max_retries=3)
        self.default_instruction = default_instruction or (
            "Given a web search query, retrieve relevant passages that answer the query"
        )
    
    def rerank(
        self,
        query: str,
        candidates: Sequence[tuple[Any, float]],
        top_k: int,
        metadata_boost_weights: Optional[dict[str, float]] = None,
        instruction: Optional[str] = None,
    ) -> list[tuple[Any, float]]:
        """Rerank using Qwen3 with instruction-aware prompting."""
        
        documents = [
            doc.page_content if hasattr(doc, "page_content") else str(doc)
            for doc, _ in candidates
        ]
        
        # Use custom instruction or default
        task_instruction = instruction or self.default_instruction
        
        response = self._post(
            "/rerank",
            {
                "query": query,
                "documents": documents,
                "instruction": task_instruction,
                "top_k": top_k
            }
        )
        
        scores = response["scores"]
        
        # Apply metadata boosts and sort
        scored = []
        for (doc, _), score in zip(candidates, scores):
            boost = self._calculate_boost(doc, metadata_boost_weights)
            final_score = float(score) * (1.0 + boost)
            scored.append((doc, final_score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
    
    def _calculate_boost(self, doc, weights):
        """Calculate metadata-based score boost."""
        if not weights or not hasattr(doc, "metadata"):
            return 0.0
        
        boost = 0.0
        meta = getattr(doc, "metadata", {})
        
        if meta.get("has_code") and weights.get("code_presence"):
            boost += weights["code_presence"]
        if meta.get("tags") and weights.get("tag_match"):
            boost += weights["tag_match"]
        if meta.get("section_heading") and weights.get("section_match"):
            boost += weights["section_match"]
        
        return boost
```

#### 6.2 Environment Configuration

```bash
# .env updates for Qwen3 Reranker
RERANKER_PROVIDER_TYPE=cmw_qwen3
CMW_QWEN3_RERANKER_ENDPOINT=http://localhost:8080
CMW_QWEN3_RERANKER_MODEL=Qwen/Qwen3-Reranker-0.6B
CMW_QWEN3_DEFAULT_INSTRUCTION="Given a web search query, retrieve relevant passages that answer the query"

# Performance tuning
CMW_QWEN3_BATCH_SIZE=8
CMW_QWEN3_WORKERS=1  # Set to 4 for CPU, 1 for GPU
CMW_QWEN3_MAX_WAIT_TIME=50

# GPU/CPU settings
CUDA_VISIBLE_DEVICES=0  # Set to "" for CPU-only
OMP_NUM_THREADS=4       # For CPU inference
```

## Comparison with DiTy MOSEC Server

| Feature | DiTy Cross-Encoder | Qwen3 CausalLM |
|---------|-------------------|----------------|
| **Architecture** | Traditional cross-encoder | CausalLM with yes/no classification |
| **Parameters** | ~110M | 0.6B (278M active) |
| **Context Length** | 512 tokens | 32K tokens |
| **Multilingual** | Russian-focused | 100+ languages |
| **Instruction Aware** | ❌ No | ✅ Yes (+1-5% accuracy) |
| **Performance** | 15 req/s (CPU) | 25 req/s (GPU), 5-8 req/s (CPU) |
| **Memory** | 800MB | 2.5GB (GPU), 1.2GB (CPU) |
| **Dependencies** | ONNX Runtime | Transformers >= 4.51.0 |
| **Prompt Format** | Simple concatenation | Chat template with system/user |

## Migration Path from DiTy to Qwen3

### Step 1: Parallel Deployment
```bash
# Run both servers
# DiTy on port 7998 (existing)
# Qwen3 on port 8080 (new)

# Test Qwen3 thoroughly before switchover
curl http://localhost:8080/rerank ...
```

### Step 2: A/B Testing
```python
# RAG system can route traffic between both
if use_qwen3:
    reranker = Qwen3RerankerAdapter("http://localhost:8080")
else:
    reranker = CMWMosecReranker("http://localhost:7998")
```

### Step 3: Gradual Migration
- Start with 10% traffic to Qwen3
- Monitor accuracy and latency
- Scale up based on results

### Step 4: Full Cutover
- Update primary endpoint
- Keep DiTy as fallback
- Decommission after stability confirmed

## Success Criteria

### Technical Requirements
- [ ] Server starts without errors
- [ ] Handles 32K context length
- [ ] Instruction-aware prompts work correctly
- [ ] Dynamic batching functional
- [ ] Health/metrics endpoints responding
- [ ] GPU memory stays under 4GB
- [ ] CPU version functional (fallback)

### Performance Targets
- [ ] GPU: 25+ req/s with batch_size=8
- [ ] CPU: 5+ req/s with batch_size=8
- [ ] P95 latency < 100ms (GPU), < 300ms (CPU)
- [ ] No memory leaks over 24h
- [ ] Graceful degradation under load

### Integration Success
- [ ] RAG system integration tested
- [ ] Multilingual accuracy validated
- [ ] Custom instructions improve results
- [ ] Backward compatibility maintained
- [ ] Monitoring and alerting working

## Risk Mitigation

### Technical Risks
1. **Transformers Version**: Requires >= 4.51.0
   - *Mitigation*: Pin version in requirements, test compatibility

2. **Memory Usage**: Higher than DiTy
   - *Mitigation*: Quantization options, batch size tuning

3. **GPU Availability**: Requires CUDA for optimal performance
   - *Mitigation*: CPU fallback mode implemented

### Operational Risks
1. **Instruction Tuning**: Wrong instructions degrade performance
   - *Mitigation*: Provide default instructions per use case

2. **Context Length**: 32K may be overkill, wastes memory
   - *Mitigation*: Configurable truncation, smart chunking

3. **Deployment Complexity**: More complex than DiTy
   - *Mitigation*: Docker-first deployment, detailed docs

## Implementation Timeline

| Phase | Duration | Deliverables |
|-------|----------|-------------|
| **Day 1** | 4 hours | Core MOSEC server, worker implementation |
| **Day 2** | 3 hours | CLI, configuration, Docker setup |
| **Day 3** | 3 hours | Testing suite, integration tests |
| **Day 4** | 2 hours | RAG integration, documentation |
| **Day 5** | 2 hours | Deployment, monitoring, benchmarking |

## Next Steps

1. **Confirm Requirements**: GPU availability, traffic patterns
2. **Model Download**: Pre-download Qwen3 model (2.5GB)
3. **Environment Setup**: Transformers 4.51.0+ installation
4. **Development**: Begin Phase 1 implementation
5. **Testing**: Validate with existing RAG test suite

**Ready to begin implementation?**