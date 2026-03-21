# Implementation Fixes Report - March 19, 2026

## Summary
Today's work focused on fixing the Qwen3-Reranker-0.6B model implementation in the cmw-mosec server to resolve "inference internal error" issues and ensure proper handling of instruction-aware reranking models while maintaining backward compatibility with standard CrossEncoder models like DiTy and BGE.

## Issues Fixed

### 1. Padding Token Issue in SentenceTransformer Models
**Problem**: Standard CrossEncoder models (DiTy, BGE) were failing with "ValueError: Cannot handle batch sizes > 1 if no padding token is defined."
**Solution**: Added padding token initialization in the RerankerWorker:
```python
if self.model.tokenizer.pad_token is None:
    self.model.tokenizer.pad_token = self.model.tokenizer.eos_token
```

### 2. Qwen3-Reranker-Specific Implementation
**Problem**: Qwen3-Reranker models require special handling as they are instruction-aware CausalLM models, not standard CrossEncoders.
**Solution**: Implemented dual-model support in RerankerWorker:
- Model type classification via `reranker_type` config field:
  - `cross_encoder`: Standard models (DiTy, BGE) using sentence-transformers
  - `causal_lm`: Qwen3 instruction-aware models using AutoModelForCausalLM
- Separate code paths based on `self.is_qwen3 = RERANKER_TYPE == "causal_lm"`
- Proper Qwen3 input formatting with prefix/suffix tokens per model documentation
- Correct yes/no token extraction for scoring

### 3. Instruction Handling Policy
**Decision**: Server remains instruction-agnostic - does not provide default instructions
**Rationale**:
- Maintains traceability - all instructions come from client
- Follows same pattern as FRIDA prefixes (client-provided via config)
- Server's role is to process what client sends, not impose semantic bias
- Testing/benchmarking utilities can provide their own instructions as needed

### 4. Backward Compatibility
**Achieved**:
- DiTy/cross-encoder-russian-msmarco continues to work unchanged
- BAAI/bge-reranker-v2-m3 works with same interface
- Existing client code requires no modifications
- Standard sentence-transformers scoring path preserved when no instruction provided

### 5. Max Length Configuration
**Implementation**:
- `max_length` required in `config/models.yaml` for all reranker models
- Read from config at server script generation time
- Embedded as `MAX_LENGTH` constant in generated worker script
- No runtime config lookup (avoids subprocess import issues)
- Client can override via request parameter: `"max_length": <value>`
- Worker uses: `effective_max_length = data.get("max_length") or self.max_length`

**For CrossEncoder models (DiTy, BGE)**:
- Set `tokenizer.model_max_length = max_length` at initialization
- For client override: temporarily set `model_max_length`, then restore after predict
- CrossEncoder does NOT accept `max_length` parameter in `predict()` method

**For Qwen3 models**:
- Use `effective_max_length` in tokenizer truncation
- Account for prefix/suffix token overhead in padding

**Config validation**:
```python
reranker_max_length = config.get("max_length")
if reranker_max_length is None:
    raise ValueError(f"max_length not configured for reranker model {model}")
```

## Files Modified
- `cmw_mosec/server_manager.py`: max_length config lookup, embedded constant, effective_max_length handling
- `config/models.yaml`: Added `max_length` to all reranker models

## Testing Performed
1. **DiTy Model**: Verified existing functionality preserved
   - Test query "машина" → "Автомобиль для перевозки грузов" ranks highest (score: 0.136)
   - Test query "artificial intelligence" → "AI and deep learning..." ranks highest (score: 0.019)

2. **BGE Model**: Verified compatibility maintained
   - Test query "машина" → "Автомобиль для перевозки грузов" ranks highest (score: 0.929)

3. **Qwen3 Model**: Verified new functionality works
   - Test query "машина" → "Автомобиль для перевозки грузов" ranks highest (score: 0.096)
   - Works without instruction (falls back to empty string)
   - Works with client-provided instructions (proper Qwen3 formatting)
   - Handles null/empty instructions correctly

4. **End-to-End Testing**:
   - `cmw-mosec check-rerank` command works for all models
   - Direct HTTP endpoint testing successful
   - Comparative ranking analysis completed

## Key Technical Details
- Qwen3 models use `<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc>` format
- ChatML format from HuggingFace model card:
  - Fixed: `<|im_start|>system\n`, `<|im_end|>\n`, `<|im_start|>user\n`, `<|im_end|>\n<|im_start|>assistant\n`
  - Fixed: `"Note that the answer can only be \"yes\" or \"no\"."` (binary classification)
  - Configurable: `system_prompt` from models.yaml
  - Client-provided: `instruction` in request (server uses empty string if not provided)
- Scoring: Uses softmax over "yes"/"no" token logits at final position
- Standard models: Use sentence-transformers CrossEncoder.predict() with model_max_length for truncation
- Max length: Config-driven defaults (max_length in models.yaml), client-controllable override
- Reranker type: Config-driven (reranker_type: cross_encoder or causal_lm)

## Impact
- Enables production use of Qwen3-Reranker-0.6B/4B/8B models
- Maintains zero-breaking-change guarantee for existing deployments
- Provides clear path for instruction-aware reranking when beneficial
- Server remains truly agnostic to instruction semantics
- All configuration traceable from `config/models.yaml`