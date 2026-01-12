# SPEC-02: Memory Foundation

## Overview

Implement persistent hypergraph memory with SQLite storage, enabling cross-session knowledge retention.

## Requirements

### Storage Layer

[SPEC-02.01] The system SHALL use SQLite for persistent memory storage.

[SPEC-02.02] The database SHALL be stored at `~/.claude/rlm-memory.db` by default.

[SPEC-02.03] The database location SHALL be configurable via `RLM_MEMORY_DB` environment variable.

[SPEC-02.04] The system SHALL use WAL mode for concurrent read/write performance.

### Node Types

[SPEC-02.05] The system SHALL support node types: "entity", "fact", "experience", "decision", "snippet".

[SPEC-02.06] Each node SHALL have: id (TEXT), type (TEXT), content (TEXT), tier (TEXT), confidence (REAL).

[SPEC-02.07] Each node SHALL track: created_at, updated_at, last_accessed, access_count.

[SPEC-02.08] Nodes MAY have: subtype (TEXT), embedding (BLOB), provenance (TEXT), metadata (JSON).

[SPEC-02.09] Node IDs SHALL be UUIDs generated using `uuid.uuid4()`.

[SPEC-02.10] Node confidence SHALL be constrained to range [0.0, 1.0].

### Hyperedges

[SPEC-02.11] The system SHALL support hyperedge types: "relation", "composition", "causation", "context".

[SPEC-02.12] Each hyperedge SHALL have: id (TEXT), type (TEXT), label (TEXT), weight (REAL).

[SPEC-02.13] Hyperedge weight SHALL be constrained to range [0.0, ∞).

### Membership

[SPEC-02.14] The system SHALL support many-to-many relationships via membership table.

[SPEC-02.15] Membership SHALL specify: hyperedge_id, node_id, role, position.

[SPEC-02.16] Membership roles SHALL include: "subject", "object", "context", "participant", "cause", "effect".

### Tier System

[SPEC-02.17] The system SHALL support tiers: "task", "session", "longterm", "archive".

[SPEC-02.18] New nodes SHALL default to "task" tier.

[SPEC-02.19] Tier transitions SHALL be logged in evolution_log table.

### CRUD Operations

[SPEC-02.20] The system SHALL provide `create_node(type, content, **kwargs) -> str` returning node ID.

[SPEC-02.21] The system SHALL provide `get_node(node_id) -> Node | None`.

[SPEC-02.22] The system SHALL provide `update_node(node_id, **kwargs) -> bool`.

[SPEC-02.23] The system SHALL provide `delete_node(node_id) -> bool` (soft delete to archive).

[SPEC-02.24] The system SHALL provide `query_nodes(type, tier, min_confidence, limit) -> list[Node]`.

[SPEC-02.25] The system SHALL provide `create_edge(type, label, members) -> str` returning edge ID.

[SPEC-02.26] The system SHALL provide `get_related_nodes(node_id, edge_type) -> list[Node]`.

### REPL Integration

[SPEC-02.27] The system SHALL expose `memory_query(query, limit)` in REPL environment.

[SPEC-02.28] The system SHALL expose `memory_add_fact(content, confidence)` in REPL environment.

[SPEC-02.29] The system SHALL expose `memory_add_experience(content, outcome, success)` in REPL environment.

[SPEC-02.30] The system SHALL expose `memory_get_context(limit)` in REPL environment.

[SPEC-02.31] The system SHALL expose `memory_relate(label, subject_id, object_id)` in REPL environment.

## Schema

```sql
CREATE TABLE nodes (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL CHECK(type IN ('entity', 'fact', 'experience', 'decision', 'snippet')),
    subtype TEXT,
    content TEXT NOT NULL,
    embedding BLOB,
    tier TEXT DEFAULT 'task' CHECK(tier IN ('task', 'session', 'longterm', 'archive')),
    confidence REAL DEFAULT 0.5 CHECK(confidence >= 0.0 AND confidence <= 1.0),
    provenance TEXT,
    created_at INTEGER DEFAULT (strftime('%s', 'now') * 1000),
    updated_at INTEGER DEFAULT (strftime('%s', 'now') * 1000),
    last_accessed INTEGER DEFAULT (strftime('%s', 'now') * 1000),
    access_count INTEGER DEFAULT 0,
    metadata JSON DEFAULT '{}'
);

CREATE TABLE hyperedges (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL CHECK(type IN ('relation', 'composition', 'causation', 'context')),
    label TEXT,
    weight REAL DEFAULT 1.0 CHECK(weight >= 0.0)
);

CREATE TABLE membership (
    hyperedge_id TEXT NOT NULL REFERENCES hyperedges(id) ON DELETE CASCADE,
    node_id TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    position INTEGER DEFAULT 0,
    PRIMARY KEY (hyperedge_id, node_id, role)
);

CREATE TABLE evolution_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp INTEGER DEFAULT (strftime('%s', 'now') * 1000),
    operation TEXT NOT NULL,
    node_ids JSON NOT NULL,
    from_tier TEXT,
    to_tier TEXT,
    reasoning TEXT
);

-- Indexes
CREATE INDEX idx_nodes_tier ON nodes(tier);
CREATE INDEX idx_nodes_type ON nodes(type);
CREATE INDEX idx_nodes_confidence ON nodes(confidence);
CREATE INDEX idx_nodes_last_accessed ON nodes(last_accessed);
CREATE INDEX idx_membership_node ON membership(node_id);
CREATE INDEX idx_membership_edge ON membership(hyperedge_id);
```

## Security

[SPEC-02.32] Memory functions in REPL SHALL execute within RestrictedPython sandbox.

[SPEC-02.33] Memory operations SHALL NOT allow arbitrary SQL execution.

[SPEC-02.34] Node content SHALL be sanitized to prevent injection attacks.

## Testing Requirements

[SPEC-02.35] Unit tests SHALL cover all CRUD operations.

[SPEC-02.36] Unit tests SHALL verify tier constraints and confidence bounds.

[SPEC-02.37] Integration tests SHALL verify REPL memory functions.

[SPEC-02.38] Property tests SHALL verify node ID uniqueness and referential integrity.
