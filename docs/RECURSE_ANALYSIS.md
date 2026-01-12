# Recurse Analysis: High-Value Capabilities for RLM-Claude-Code

Deep analysis of the [recurse](https://github.com/rand/recurse) project to identify capabilities worth porting to rlm-claude-code.

---

## Executive Summary

Recurse extends the RLM paradigm with three transformational innovations:

| Priority | Capability | Value | Effort |
|----------|-----------|-------|--------|
| **P0** | Persistent Hypergraph Memory | Transforms RLM from stateless to learning | 2 weeks |
| **P1** | Advanced REPL Functions | Enables new processing strategies | 3-4 days |
| **P2** | Deciduous Reasoning Traces | Explainability + git integration | 1 week |
| **P3** | Enhanced Budget Tracking | Granular cost control | 2-3 days |

**Recommendation**: Implement P0 (Memory) first as it provides the foundation for all other capabilities. P1 (REPL Functions) can be done in parallel as a quick win.

---

## Current State Comparison

### What rlm-claude-code Has

| Component | Implementation | Location |
|-----------|---------------|----------|
| Secure REPL Sandbox | RestrictedPython, allowlist | `repl_environment.py` |
| Complexity Classification | Pattern-based, threshold configurable | `complexity_classifier.py` |
| Strategy Cache | Feature-based similarity, JSON storage | `strategy_cache.py` |
| Cost Tracking | Model costs, session budget, alerts | `cost_tracker.py` |
| Trajectory System | Event streaming, visualization | `trajectory.py` |
| Intelligent Orchestrator | Haiku-powered meta-controller | `intelligent_orchestrator.py` |
| Multi-Provider Client | Anthropic + OpenAI | `api_client.py` |
| REPL Helpers | peek, search, summarize, llm, llm_batch | `repl_environment.py` |

### What rlm-claude-code Lacks (from Recurse)

| Capability | Recurse Implementation | Gap Severity |
|------------|----------------------|--------------|
| **Persistent Memory** | SQLite hypergraph, 3 tiers | **CRITICAL** |
| **Memory Evolution** | Consolidation, promotion, decay | **HIGH** |
| **Memory REPL Functions** | memory_query, memory_add_fact, etc. | **HIGH** |
| **map_reduce()** | Parallel map + reduce synthesis | **MEDIUM** |
| **find_relevant()** | Keyword + LLM relevance scoring | **MEDIUM** |
| **extract_functions()** | Multi-language function parser | **LOW** |
| **Decision Nodes** | Goal→Decision→Option→Action→Outcome | **MEDIUM** |
| **Git Integration** | Commits linked to decisions | **LOW** |
| **Trace Events Schema** | Persistent trace storage | **MEDIUM** |

---

## Priority 0: Persistent Hypergraph Memory

### Why Critical

The recurse memory system enables capabilities impossible with stateless RLM:

1. **Cross-Session Learning**: Knowledge persists and improves across conversations
2. **Codebase Familiarity**: System remembers file structures, patterns, decisions
3. **Reduced Re-Discovery**: Don't re-analyze the same code repeatedly
4. **Contextual Recall**: Retrieve relevant past experiences for current tasks
5. **User Preference Learning**: Remember how user likes explanations, code style, etc.

### Architecture from Recurse

```
┌─────────────────────────────────────────────────────────────────┐
│                    TIERED HYPERGRAPH MEMORY                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  TASK TIER (Working Memory)         Lifecycle: Single task       │
│  ├─ Current context, immediate facts                             │
│  ├─ Aggressive consolidation                                     │
│  └─ Fast access, high churn                                      │
│                                                                  │
│  SESSION TIER (Accumulated)         Lifecycle: Claude Code run   │
│  ├─ Consolidated from task tier                                  │
│  ├─ Persists across tasks within session                         │
│  └─ Merged facts, deduplicated                                   │
│                                                                  │
│  LONG-TERM TIER (Persistent)        Lifecycle: Indefinite        │
│  ├─ Promoted from session tier                                   │
│  ├─ Cross-session knowledge (survives restarts)                  │
│  └─ Curated, high-confidence facts with decay                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Node Types (from Recurse SPEC)

| Type | Purpose | Example | Typical Confidence |
|------|---------|---------|-------------------|
| **entity** | Code elements | "Function `authenticate` in auth.py" | 0.95+ |
| **fact** | Extracted knowledge | "auth.py uses bcrypt for password hashing" | 0.7-0.9 |
| **experience** | Interaction patterns | "User prefers detailed explanations" | 0.6-0.8 |
| **decision** | Reasoning traces | "Chose SQLAlchemy over raw SQL" | 0.8-0.95 |
| **snippet** | Verbatim content with provenance | Code blocks, error messages | 1.0 |

### Hyperedge Types

| Type | Purpose | Roles | Example |
|------|---------|-------|---------|
| **relation** | Entity connections | subject, object | "auth.py imports db.py" |
| **composition** | Part-of | container, part | "User class contains email field" |
| **causation** | Cause-effect | cause, effect | "Missing validation caused XSS bug" |
| **context** | Situational grouping | members | "These files relate to auth" |

### SQLite Schema (from Recurse)

```sql
-- Core hypergraph structure
CREATE TABLE nodes (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL CHECK(type IN ('entity', 'fact', 'experience', 'decision', 'snippet')),
    subtype TEXT,  -- e.g., 'function', 'class', 'file' for entities
    content TEXT NOT NULL,
    embedding BLOB,  -- vector for similarity search (Voyage-3)
    tier TEXT DEFAULT 'task' CHECK(tier IN ('task', 'session', 'longterm', 'archive')),
    confidence REAL DEFAULT 0.5 CHECK(confidence >= 0.0 AND confidence <= 1.0),
    provenance TEXT,  -- source of the node (file path, commit, etc.)
    created_at INTEGER DEFAULT (strftime('%s', 'now') * 1000),
    updated_at INTEGER DEFAULT (strftime('%s', 'now') * 1000),
    last_accessed INTEGER DEFAULT (strftime('%s', 'now') * 1000),
    access_count INTEGER DEFAULT 0,
    metadata JSON DEFAULT '{}'
);

CREATE TABLE hyperedges (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL CHECK(type IN ('relation', 'composition', 'causation', 'context')),
    label TEXT,  -- human-readable label
    weight REAL DEFAULT 1.0 CHECK(weight >= 0.0)
);

CREATE TABLE membership (
    hyperedge_id TEXT NOT NULL REFERENCES hyperedges(id) ON DELETE CASCADE,
    node_id TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    role TEXT NOT NULL,  -- subject, object, context, participant, cause, effect
    position INTEGER DEFAULT 0,  -- ordering within hyperedge
    PRIMARY KEY (hyperedge_id, node_id, role)
);

-- Decisions (reasoning trace integration)
CREATE TABLE decisions (
    node_id TEXT PRIMARY KEY REFERENCES nodes(id) ON DELETE CASCADE,
    decision_type TEXT NOT NULL CHECK(decision_type IN ('goal', 'decision', 'option', 'action', 'outcome', 'observation')),
    confidence REAL DEFAULT 0.5,
    prompt TEXT,  -- original prompt that spawned this
    files JSON,  -- related files
    branch TEXT,  -- git branch
    commit_hash TEXT,  -- linked commit
    parent_id TEXT REFERENCES decisions(node_id)
);

-- Evolution audit log
CREATE TABLE evolution_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp INTEGER DEFAULT (strftime('%s', 'now') * 1000),
    operation TEXT NOT NULL CHECK(operation IN ('create', 'consolidate', 'promote', 'decay', 'prune', 'archive')),
    node_ids JSON NOT NULL,  -- affected nodes
    from_tier TEXT,
    to_tier TEXT,
    reasoning TEXT  -- why this operation was performed
);

-- Indexes for efficient queries
CREATE INDEX idx_nodes_tier ON nodes(tier);
CREATE INDEX idx_nodes_type ON nodes(type);
CREATE INDEX idx_nodes_confidence ON nodes(confidence);
CREATE INDEX idx_nodes_last_accessed ON nodes(last_accessed);
CREATE INDEX idx_membership_node ON membership(node_id);
CREATE INDEX idx_membership_edge ON membership(hyperedge_id);
CREATE INDEX idx_decisions_type ON decisions(decision_type);
CREATE INDEX idx_decisions_parent ON decisions(parent_id);

-- Automatic timestamp updates
CREATE TRIGGER update_node_timestamp
AFTER UPDATE ON nodes
BEGIN
    UPDATE nodes SET updated_at = strftime('%s', 'now') * 1000 WHERE id = NEW.id;
END;

CREATE TRIGGER update_access_timestamp
AFTER UPDATE OF access_count ON nodes
BEGIN
    UPDATE nodes SET last_accessed = strftime('%s', 'now') * 1000 WHERE id = NEW.id;
END;
```

### Memory Evolution Operations

```python
class MemoryEvolution:
    """Memory lifecycle management from Recurse."""

    def consolidate(self, task_id: str) -> list[str]:
        """
        Consolidate task tier → session tier.

        1. Find related facts (similar embeddings, shared hyperedges)
        2. Merge redundant facts (keep highest confidence)
        3. Strengthen frequently-accessed edges
        4. Preserve detail with summary links
        """

    def promote(self, session_id: str, threshold: float = 0.8) -> list[str]:
        """
        Promote session tier → long-term tier.

        1. Select nodes with confidence >= threshold
        2. Select nodes with high access_count
        3. Apply Ebbinghaus decay to existing long-term nodes
        4. Create crystallized (summary) nodes for complex subgraphs
        """

    def decay(self, factor: float = 0.95, min_confidence: float = 0.3) -> list[str]:
        """
        Apply temporal decay to long-term memory.

        Formula: new_confidence = base_confidence * (factor ^ days_since_access)
        Amplified by: access_count / (1 + log(access_count))

        Archive nodes that fall below min_confidence (never delete).
        """
```

### REPL Memory Functions (from Recurse bootstrap.py)

```python
# Already in recurse - need to add to rlm-claude-code

def memory_query(query: str, limit: int = 10) -> list[MemoryNode]:
    """Search persistent memory by semantic similarity."""

def memory_add_fact(content: str, confidence: float = 0.8) -> str:
    """Store extracted knowledge with confidence level."""

def memory_add_experience(content: str, outcome: str, success: bool) -> str:
    """Log experiences and outcomes for learning."""

def memory_get_context(limit: int = 5) -> str:
    """Retrieve recently relevant nodes for context injection."""

def memory_relate(label: str, subject_id: str, object_id: str) -> str:
    """Create semantic relationship between nodes."""
```

---

## Priority 1: Advanced REPL Functions

### Why High Value

These functions enable sophisticated processing strategies that RLM can discover and employ:

- **map_reduce()**: Process arbitrarily large contexts via parallel decomposition
- **find_relevant()**: Smart context narrowing with LLM-assisted scoring
- **extract_functions()**: Structured code understanding without manual parsing

### map_reduce() (from Recurse)

```python
def map_reduce(
    ctx: RLMContext,
    map_prompt: str,
    reduce_prompt: str,
    n_chunks: int = 4,
    model: str = "auto"
) -> str:
    """
    Apply map-reduce pattern to large content.

    1. Partition context into n_chunks
    2. Map: Apply map_prompt to each chunk (parallel via llm_batch)
    3. Reduce: Combine results with reduce_prompt

    Example:
        map_reduce(
            large_codebase,
            map_prompt="List all TODO comments in this code",
            reduce_prompt="Combine these TODO lists, remove duplicates, prioritize by urgency"
        )
    """
    chunks = partition(ctx, n_chunks)

    # Parallel map phase
    map_results = llm_batch(
        prompts=[map_prompt] * len(chunks),
        contexts=[str(chunk) for chunk in chunks],
        model=model
    )

    # Reduce phase
    combined = "\n---\n".join(map_results)
    return llm_call(reduce_prompt, combined, model=model)
```

### find_relevant() (from Recurse)

```python
def find_relevant(
    ctx: RLMContext,
    query: str,
    top_k: int = 5,
    model: str = "auto"
) -> list[tuple[str, float]]:
    """
    Find sections most relevant to query.

    1. Partition into ~50-line chunks
    2. Keyword filtering for initial candidates
    3. LLM relevance scoring for final ranking
    4. Return top-k with scores
    """
    sections = partition_by_lines(ctx, n=50, overlap_lines=5)

    # Keyword pre-filter
    keywords = set(query.lower().split())
    candidates = []
    for section in sections:
        section_words = set(str(section).lower().split())
        overlap = len(keywords & section_words)
        if overlap > 0:
            candidates.append((section, overlap))

    # If too many candidates, use LLM scoring
    if len(candidates) > top_k * 2:
        scores = llm_batch(
            prompts=[f"Rate relevance 0-10 for query '{query}':\n\n{c[0]}"
                     for c in candidates[:20]],
            contexts=[""] * min(20, len(candidates)),
            model="fast"
        )
        scored = [(c[0], float(s)) for c, s in zip(candidates, scores)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    candidates.sort(key=lambda x: x[1], reverse=True)
    return [(str(c[0]), c[1]) for c in candidates[:top_k]]
```

### extract_functions() (from Recurse)

```python
def extract_functions(
    ctx: RLMContext,
    language: str = "python"
) -> list[dict]:
    """
    Extract function definitions using language-specific patterns.

    Returns list of:
    {
        "name": str,
        "signature": str,
        "docstring": str | None,
        "start_line": int,
        "end_line": int
    }
    """
    patterns = {
        "python": r'def\s+(\w+)\s*\([^)]*\)\s*(?:->\s*[^:]+)?:',
        "go": r'func\s+(?:\([^)]+\)\s+)?(\w+)\s*\([^)]*\)',
        "javascript": r'(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:function|\([^)]*\)\s*=>))',
        "typescript": r'(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*(?::\s*[^=]+)?\s*=\s*(?:async\s+)?(?:function|\([^)]*\)\s*=>))',
    }
    # Implementation uses regex + context extraction
```

---

## Priority 2: Deciduous Reasoning Traces

### Why Valuable

Structured decision tracking enables:
- **Explainability**: Understand why RLM made specific choices
- **Learning from mistakes**: Link outcomes to decisions
- **Git integration**: Connect code changes to reasoning
- **Replay and analysis**: Review decision trees for optimization

### Decision Graph Structure

```
GOAL ──spawns──► DECISION ──considers──► OPTION
                      │                     │
                      │                     ├── chooses (one)
                      │                     └── rejects (rest, with reason)
                      │
                      └──implements──► ACTION ──produces──► OUTCOME
                                                              │
OBSERVATION ──informs──► DECISION ◄────feedback───────────────┘
```

### Integration Points

1. **Trajectory Events**: Current trajectory system → decision graph mapping
2. **Git Hooks**: Link commits to decision nodes via branch/hash
3. **Memory Integration**: Decision nodes stored in hypergraph memory
4. **Query Interface**: "What decisions led to this commit?"

---

## Priority 3: Enhanced Budget Tracking

### Current vs Recurse

| Metric | rlm-claude-code | Recurse |
|--------|-----------------|---------|
| Token counts | input, output | + cached tokens |
| Cost tracking | Per-model USD | Same |
| Recursion tracking | depth only | + sub_call_count, repl_executions |
| Time tracking | latency_ms | + session_duration, wall_clock |
| Alerts | Budget threshold | + token threshold, depth warning |
| Limits | session_budget | + per_task, per_call, depth_limit |

### Enhanced Schema

```python
@dataclass
class EnhancedBudgetMetrics:
    # Token tracking
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0  # NEW: prompt caching

    # Cost
    total_cost_usd: float = 0.0

    # Execution metrics
    recursion_depth: int = 0
    max_depth_reached: int = 0
    sub_call_count: int = 0  # NEW
    repl_executions: int = 0  # NEW

    # Time
    session_start: float = 0.0
    session_duration_seconds: float = 0.0
    wall_clock_seconds: float = 0.0  # NEW: includes wait time

    # Granular limits
    limits: BudgetLimits = field(default_factory=BudgetLimits)

@dataclass
class BudgetLimits:
    max_cost_per_task: float = 5.0
    max_cost_per_session: float = 25.0
    max_tokens_per_call: int = 8000  # NEW
    max_recursive_calls: int = 10  # NEW
    max_repl_executions: int = 50  # NEW
    cost_alert_threshold: float = 0.8  # 80%
    token_alert_threshold: float = 0.75  # NEW
```

---

## Detailed Execution Plan

### Phase 0: Quick Wins (3-4 days)

Implement standalone REPL functions that don't require memory infrastructure.

#### Day 1: map_reduce()
```
Files to modify:
- src/repl_environment.py (add _map_reduce method)
- src/types.py (add MapReduceResult type if needed)
- tests/unit/test_repl_environment.py (add tests)
```

#### Day 2: find_relevant()
```
Files to modify:
- src/repl_environment.py (add _find_relevant method)
- tests/unit/test_repl_environment.py (add tests)
```

#### Day 3: extract_functions()
```
Files to modify:
- src/repl_environment.py (add _extract_functions method)
- tests/unit/test_repl_environment.py (add tests)
```

#### Day 4: Integration & Documentation
```
- Wire up to REPL globals
- Add to user-guide.md
- Run full test suite
```

### Phase 1: Memory Foundation (1 week)

#### Day 1-2: Schema & Storage
```
New files:
- src/memory/__init__.py
- src/memory/schema.py (Python wrapper for SQLite schema)
- src/memory/store.py (CRUD operations)
- schema/memory.sql (SQLite schema)
- tests/unit/test_memory_store.py
```

#### Day 3-4: Node & Edge Operations
```
New files:
- src/memory/node.py (Node class with confidence, tier)
- src/memory/edge.py (Hyperedge class with membership)
- tests/unit/test_memory_node.py
- tests/unit/test_memory_edge.py
```

#### Day 5: REPL Integration
```
Files to modify:
- src/repl_environment.py (add memory_* functions)
- src/types.py (add MemoryNode, MemoryEdge types)
Tests:
- tests/unit/test_repl_memory.py
- tests/integration/test_memory_repl.py
```

### Phase 2: Memory Evolution (1 week)

#### Day 1-2: Tier System
```
New files:
- src/memory/tiers.py (TaskTier, SessionTier, LongTermTier)
- tests/unit/test_memory_tiers.py
```

#### Day 3-4: Evolution Operations
```
New files:
- src/memory/evolution.py (consolidate, promote, decay)
- tests/unit/test_memory_evolution.py
- tests/property/test_evolution_invariants.py
```

#### Day 5: Lifecycle Hooks
```
Files to modify:
- src/orchestrator.py (add memory lifecycle calls)
- src/trajectory.py (emit memory events)
```

### Phase 3: Reasoning Traces (1 week)

#### Day 1-2: Decision Node Types
```
New files:
- src/memory/reasoning.py (DecisionNode, TraceEdge types)
- tests/unit/test_reasoning.py
```

#### Day 3-4: Git Integration
```
New files:
- src/memory/git.py (commit linking, diff extraction)
- tests/unit/test_memory_git.py
```

#### Day 5: Trajectory Integration
```
Files to modify:
- src/trajectory.py (TrajectoryEvent → DecisionNode mapping)
- src/trajectory_analysis.py (decision tree queries)
```

### Phase 4: Polish (3 days)

#### Day 1: Enhanced Budget Tracking
```
Files to modify:
- src/cost_tracker.py (add new metrics and limits)
- src/types.py (update BudgetMetrics)
- tests/unit/test_cost_tracker.py
```

#### Day 2: Configuration & Persistence
```
Files to modify:
- src/config.py (add memory config options)
- src/user_preferences.py (memory tier thresholds)
```

#### Day 3: Documentation & Testing
```
Files to update:
- docs/user-guide.md (memory section)
- README.md (memory features)
- Run full test suite
- Performance benchmarks for memory queries
```

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| SQLite performance at scale | Medium | Medium | Add indices, use WAL mode, benchmark |
| Embedding costs (Voyage-3) | Low | Medium | Make optional, cache embeddings |
| Memory bloat over time | Medium | Low | Decay algorithm, archive tier |
| REPL security with memory | Low | High | Same RestrictedPython sandbox |
| Breaking existing tests | Low | Medium | Feature flags, incremental rollout |

---

## Success Metrics

### Phase 0 (Quick Wins)
- [ ] map_reduce() works on 1M+ character contexts
- [ ] find_relevant() returns scored results in <2s
- [ ] extract_functions() handles Python, JS, Go, TS
- [ ] All existing tests still pass

### Phase 1 (Memory Foundation)
- [ ] Nodes can be created, queried, updated, deleted
- [ ] Hyperedges connect nodes with roles
- [ ] REPL memory functions work in sandbox
- [ ] Memory persists across Claude Code sessions

### Phase 2 (Evolution)
- [ ] Task→session consolidation reduces duplicates
- [ ] Session→longterm promotion works with threshold
- [ ] Decay doesn't delete, only archives
- [ ] Evolution audit log captures all changes

### Phase 3 (Reasoning Traces)
- [ ] Decision nodes link to trajectory events
- [ ] Git commits link to decisions
- [ ] "What decided this?" queries work
- [ ] Rejected options captured with reasons

---

## References

- [Recurse SPEC.md](https://github.com/rand/recurse/blob/main/docs/SPEC.md)
- [Recurse bootstrap.py](https://github.com/rand/recurse/blob/main/pkg/python/bootstrap.py) - REPL implementation
- [Recurse memory schema](https://github.com/rand/recurse/blob/main/schema/memory.sql)
- [RLM Paper](https://arxiv.org/abs/2512.24601) - Recursive Language Models
- [HGMem Paper](https://arxiv.org/abs/2512.23959) - Hypergraph Memory
- [Deciduous Paper](https://arxiv.org/abs/2310.05678) - Decision-aware reasoning traces
- [Voyage Embeddings](https://docs.voyageai.com/) - For memory similarity search
