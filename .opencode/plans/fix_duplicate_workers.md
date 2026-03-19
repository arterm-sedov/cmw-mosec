# Fix Duplicate GPU Workers - Reduce Memory Usage by ~50%

## Problem Summary

The Mosec server is running **4 worker processes instead of 2**, consuming ~50% more GPU memory than necessary.

**Current VRAM Usage:**
- Total: 49,140 MiB
- Used: 41,066 MiB
- Free: 7,445 MiB

**Current Process Count:**
- RerankerWorker: 2 workers (should be 1)
- GuardWorker: 2 workers (should be 1)
- Total: 4 workers (should be 2)

**Expected Memory Savings:**
- Reduce from ~5,950 MiB to ~2,975 MiB per model
- Save ~3,000 MiB (~50% of worker memory)
- Free up GPU memory for other processes

---

## Root Cause Analysis

### IssueLocation

**File:** `cmw_mosec/server_manager.py` lines 574-580

```python
# REGISTERING EMBEDDING ENDPOINT (CORRECT PATTERN)
if "EmbeddingWorker" in globals():
    emb = Runtime(EmbeddingWorker)  # Create ONCE
    routes["/v1/embeddings"] = [emb]  # Reuse instance
    routes["/embeddings"] = [emb]      # Reuse instance

# REGISTERING RERANKER ENDPOINT (WRONG PATTERN)
if "RerankerWorker" in globals():
    routes["/v1/rerank"] = [Runtime(RerankerWorker)]  # Creates worker #1
    routes["/rerank"] = [Runtime(RerankerWorker)]     # Creates worker #2

# REGISTERING GUARD ENDPOINT (WRONG PATTERN)
if "GuardWorker" in globals():
    routes["/v1/moderate"] = [Runtime(GuardWorker)]  # Creates worker #3
    routes["/moderate"] = [Runtime(GuardWorker)]     # Creates worker #4
```

### Why This Happens

**Mosec Runtime Behavior:**
- Each `Runtime(WorkerClass)` instantiation spawns a **new worker process**
- Worker processes load the model into GPU memory
- Multiple `Runtime()` calls = multiple workers = multiple GPU memory allocations

**The Duplicate Routes:**
- `/v1/rerank` - OpenAI-compatible endpoint
- `/rerank` - Convenience alias (UNUSED)
- `/v1/moderate` - OpenAI-style endpoint
- `/moderate` - Convenience alias (UNUSED)

### Codebase Usage Analysis

**Scanned repositories:** `cmw-mosec`, `cmw-rag`

**Non-v1 endpoint usage:** **ZERO occurrences**

All code uses `/v1/*` endpoints exclusively:
- `/v1/embeddings` used in: tests, examples, cmw-rag integration
- `/v1/rerank` used in: tests, examples, cmw-rag integration  
- `/v1/moderate` used in: tests, examples, cmw-rag integration

**Dead code:** `/embeddings`, `/rerank`, `/moderate` routes are completely unused

---

## Solution Design

### Principles Applied

- **DRY:** Remove duplicate code
- **Minimal:** Smallest change to fix the problem
- **Non-breaking:** Preserve `/v1/*` endpoints (used everywhere)
- **SDD:** Simple solution - remove dead code
- **Clean:** Remove technical debt

### Proposed Fix

**Option A: Remove unused endpoints (RECOMMENDED)**

Remove the non-v1 routes entirely. They're dead code that wastes resources.

**Option B: Reuse Runtime instances**

Keep both routes but share the same worker (like EmbeddingWorker).

**Recommendation: Option A** because:
1. Non-v1 routes have zero usage
2. Simpler codebase
3. Saves unnecessary route maintenance
4. Prevents future confusion

---

## Implementation Plan

### Phase 1: Code Changes

#### File: `cmw_mosec/server_manager.py`

**Lines 574-580 - Current:**
```python
# Register reranker endpoint
if "RerankerWorker" in globals():
    routes["/v1/rerank"] = [Runtime(RerankerWorker)]
    routes["/rerank"] = [Runtime(RerankerWorker)]

# Register guard endpoint
if "GuardWorker" in globals():
    routes["/v1/moderate"] = [Runtime(GuardWorker)]
    routes["/moderate"] = [Runtime(GuardWorker)]
```

**Lines 574-580 - Fixed:**
```python
# Register reranker endpoint
if "RerankerWorker" in globals():
    routes["/v1/rerank"] = [Runtime(RerankerWorker)]

# Register guard endpoint
if "GuardWorker" in globals():
    routes["/v1/moderate"] = [Runtime(GuardWorker)]
```

**Lines 568-570 - Optional cleanup:**
```python
# Register embedding endpoint
if "EmbeddingWorker" in globals():emb = Runtime(EmbeddingWorker)
    routes["/v1/embeddings"] = [emb]
    routes["/embeddings"] = [emb]
```

If we remove non-v1 routes, also simplify this to:
```python
# Register embedding endpoint
if "EmbeddingWorker" in globals():
    routes["/v1/embeddings"] = [Runtime(EmbeddingWorker)]
```

#### File: `cmw_mosec/cli.py`

**Lines 131-135 - Current:**
```python
click.echo("  POST /v1/embeddings - Embeddings API")
click.echo("  POST /v1/rerank - Reranking API")
click.echo("  POST /v1/moderate - Content moderation API")
```

**No changes needed** - already shows only `/v1/*` endpoints

---

### Phase 2: Testing Strategy (TDD)

#### Test 1: Verify worker count reduction

**Before fix:**
```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
# Should show 4 workers for cmw-mosec
```

**After fix:**
```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
# Should show 2 workers for cmw-mosec
```

#### Test 2: Verify endpoints still work

```bash
# Test reranker
curl -X POST http://localhost:7998/v1/rerank \
  -H "Content-Type: application/json"\
  -d '{"query": "test", "docs": ["doc1", "doc2"]}'

# Test guard
curl -X POST http://localhost:7998/v1/moderate \
  -H "Content-Type: application/json" \
  -d '{"content": "test"}'

# Test embedding (if running)
curl -X POST http://localhost:7998/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "test", "input": "test"}'
```

#### Test 3: Verify memory reduction

**Before:**
```bash
nvidia-smi --query-gpu=memory.used --format=csv,noheader
# Expected: ~41,000 MiB
```

**After:**
```bash
nvidia-smi --query-gpu=memory.used --format=csv,noheader
# Expected: ~38,000 MiB (or lower)
```

#### Test 4: Existing test suite

```bash
cd /home/asedov/cmw-mosec
pytest tests/test_cpu_integration.py -v
```

All existing tests should pass (they use `/v1/*` endpoints)

---

### Phase 3: Verification Checklist

- [ ] Code changes applied to `server_manager.py`
- [ ] No changes needed to `cli.py` (already correct)
- [ ] Server restarted with new code
- [ ] Worker count reduced from 4 to 2
- [ ] GPU memory usage reduced by ~3,000 MiB
- [ ] All `/v1/*` endpoints functional
- [ ] Existing test suite passes
- [ ] No regression in API responses
- [ ] Documentation updated (if any)

---

## Impact Analysis

### Positive Impacts

1. **Memory Savings:** ~3,000 MiB freed (~50% of worker memory)
2. **Resource Efficiency:** Better GPU utilization
3. **Code Simplicity:** Remove dead code
4. **Maintenance:** Fewer routes to maintain
5. **Clarity:** Consistent API surface (`/v1/*` only)

### Negative Impacts

1. **Breaking Change:** Non-v1 endpoints removed
   - Mitigation: None in use, verified across codebase
   - Impact: Zero users affected

### Risk Assessment

**Risk Level: LOW**

- Unused endpoints being removed
- No production usage found
- Tests use `/v1/*` endpoints exclusively
- Can rollback quickly if needed

---

## Alternative Solutions Considered

### Alternative 1: Keep both routes, reuse Runtime

```python
reranker = Runtime(RerankerWorker)
routes["/v1/rerank"] = [reranker]
routes["/rerank"] = [reranker]

guard = Runtime(GuardWorker)
routes["/v1/moderate"] = [guard]
routes["/moderate"] = [guard]
```

**Pros:**
- Maintains backward compatibility
- Both routes available

**Cons:**
- Keeps dead code
- Route proliferation
- Future confusion about which route to use

**Decision:** Rejected - adds complexity for no benefit

### Alternative 2: Migrate all to non-v1 routes

**Pros:**
- Simpler URLs

**Cons:**
- Breaking change for all users
- Against OpenAI API convention
- Extensive refactoring needed

**Decision:** Rejected - would break all existing clients

---

## Rollback Plan

If issues arise:

1. **Immediate:** Restart with previous code version
   ```bash
   git checkout HEAD~1 cmw_mosec/server_manager.py
   ```

2. **Alternative:** Use Option B (reuse Runtime)
   ```python
   reranker = Runtime(RerankerWorker)
   routes["/v1/rerank"] = [reranker]
   routes["/rerank"] = [reranker]
   ```

---

## Success Metrics

1. **Worker Count:** Reduced from 4 to 2
2. **Memory Usage:** Reduced by ~3,000 MiB
3. **Test Suite:** All tests pass
4. **API Response:** No change in functionality
5. **Code Quality:** Removed dead code

---

## Timeline

- **Preparation:** 5 minutes (verify current state)
- **Implementation:** 2 minutes (code changes)
- **Testing:** 5 minutes (run tests)
- **Verification:** 3 minutes (check memory)
- **Total:** ~15 minutes

---

## References

### Files Affected
- `cmw_mosec/server_manager.py` (lines 264-292 in mosec_server.py template)
- `cmw_mosec/cli.py` (documentation only, no changes)

### Related Documentation
- Mosec Runtime documentation
- OpenAI API conventions

### Related Issues
- GPU memory optimization
- Resource efficiency
- Dead code removal

---

## Appendix: Process State

### Current Process List

```
PID      COMMAND                                     MEMORY
3227323  python3                                     34,086 MiB (other process)
3318116  mosec worker (RerankerWorker route1)        ~1,824 MiB
3318117  mosec worker (RerankerWorker route 2)        ~1,120 MiB
3318118  mosec worker (GuardWorker route 1)          ~1,964 MiB
3318127  mosec worker (GuardWorker route 2)          ~2,042 MiB
```

### Expected Process List (After Fix)

```
PID      COMMAND                                     MEMORY
3227323  python3                                     34,086 MiB (other process)
3318XXX  mosec worker (RerankerWorker)               ~1,824 MiB
3318YYY  mosec worker (GuardWorker)                  ~1,964 MiB
```

---

## Execution Command

When ready to implement:

```bash
# 1. Stop the server
pkill -f mosec_server.py

# 2. Apply fixes to server_manager.py

# 3. Restart the server
cmw-mosec start <model>

# 4. Verify worker count
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

# 5. Run tests
pytest tests/
```

---

## Notes

- This is a **low-risk, high-impact** optimization
- No user-facing changes expected
- Immediate memory benefit
- Cleaner codebase as bonus
- Follows minimal change principle