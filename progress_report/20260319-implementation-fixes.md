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
- Detection of Qwen3 models via `"Qwen" in model_name and "Reranker" in model_name`
- Separate code paths for Qwen3 (AutoModelForCausalLM) vs standard models (SentenceTransformer CrossEncoder)
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

## Files Modified
- `cmw_mosec/server_manager.py`: Major rewrite of RerankerWorker class
- `cmw_mosec/config/models.yaml`: Added documentation for model types

## Testing Performed
1. **DiTy Model**: Verified existing functionality preserved
2. **Qwen3 Model**: 
   - Works without instruction (falls back to standard scoring)
   - Works with client-provided instructions (proper Qwen3 formatting)
   - Handles null/empty instructions correctly
3. **BGE Model**: Verified compatibility maintained
4. **End-to-End Testing**: 
   - `cmw-mosec check-rerank` command works for all models
   - Direct HTTP endpoint testing successful
   - Comparative ranking analysis completed

## Key Technical Details
- Qwen3 models use `<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}` format
- Prefix: `"system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".\nuser\n"`
- Suffix: `"\nassistant\n\n\n\n"`
- Scoring: Uses softmax over "yes"/"no" token logits at final position
- Standard models: Use sentence-transformers CrossEncoder.predict() with padding token fix

## Impact
- Enables production use of Qwen3-Reranker-0.6B/4B/8B models
- Maintains zero-breaking-change guarantee for existing deployments
- Provides clear path for instruction-aware reranking when beneficial
- Server remains truly agnostic to instruction semantics