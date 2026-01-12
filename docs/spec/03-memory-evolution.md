# SPEC-03: Memory Evolution

## Overview

Implement memory lifecycle management: consolidation, promotion, and decay algorithms.

## Dependencies

- Requires SPEC-02 (Memory Foundation)

## Requirements

### Consolidation (Task → Session)

[SPEC-03.01] The system SHALL provide `consolidate(task_id)` to merge task tier nodes to session tier.

[SPEC-03.02] Consolidation SHALL identify related facts by shared hyperedges.

[SPEC-03.03] Consolidation SHALL merge redundant facts, keeping highest confidence version.

[SPEC-03.04] Consolidation SHALL strengthen frequently-accessed edges by increasing weight.

[SPEC-03.05] Consolidation SHALL preserve detail via "summarizes" edges linking merged nodes.

[SPEC-03.06] Consolidation SHALL log all operations to evolution_log with reasoning.

[SPEC-03.07] Consolidation SHALL be triggered automatically at task completion.

### Promotion (Session → Long-term)

[SPEC-03.08] The system SHALL provide `promote(session_id, threshold)` to move valuable nodes to long-term.

[SPEC-03.09] Promotion SHALL select nodes with confidence >= threshold (default 0.8).

[SPEC-03.10] Promotion SHALL select nodes with access_count above session median.

[SPEC-03.11] Promotion SHALL create "crystallized" summary nodes for complex subgraphs.

[SPEC-03.12] Promotion SHALL preserve original nodes in session tier until confirmed.

[SPEC-03.13] Promotion SHALL log all operations to evolution_log with reasoning.

[SPEC-03.14] Promotion SHALL be triggered at session end (Claude Code shutdown).

### Decay

[SPEC-03.15] The system SHALL provide `decay(factor, min_confidence)` to reduce stale node confidence.

[SPEC-03.16] Decay formula SHALL be: `new_confidence = base_confidence * (factor ^ days_since_access)`.

[SPEC-03.17] Decay SHALL be amplified by access frequency: `amplifier = access_count / (1 + log(access_count))`.

[SPEC-03.18] Nodes below min_confidence (default 0.3) SHALL be moved to "archive" tier, never deleted.

[SPEC-03.19] Decay SHALL run periodically (configurable, default: daily).

[SPEC-03.20] Decay SHALL log all tier transitions to evolution_log.

### Archive Tier

[SPEC-03.21] Archive tier nodes SHALL NOT be returned in normal queries.

[SPEC-03.22] Archive tier nodes SHALL be retrievable via explicit `include_archived=True` parameter.

[SPEC-03.23] Archive tier nodes MAY be restored to long-term tier via `restore_node(node_id)`.

[SPEC-03.24] Archive tier SHALL NOT have automatic deletion; data is preserved indefinitely.

### Configuration

[SPEC-03.25] Evolution parameters SHALL be configurable via `~/.claude/rlm-config.json`:
```json
{
  "memory": {
    "consolidation_threshold": 0.5,
    "promotion_threshold": 0.8,
    "decay_factor": 0.95,
    "decay_min_confidence": 0.3,
    "decay_interval_hours": 24
  }
}
```

## Interface

```python
class MemoryEvolution:
    def consolidate(self, task_id: str) -> ConsolidationResult:
        """Consolidate task tier nodes to session tier."""

    def promote(self, session_id: str, threshold: float = 0.8) -> PromotionResult:
        """Promote session tier nodes to long-term tier."""

    def decay(self, factor: float = 0.95, min_confidence: float = 0.3) -> DecayResult:
        """Apply temporal decay to long-term memory."""

    def restore_node(self, node_id: str) -> bool:
        """Restore archived node to long-term tier."""

@dataclass
class ConsolidationResult:
    merged_count: int
    promoted_count: int
    edges_strengthened: int

@dataclass
class PromotionResult:
    promoted_count: int
    crystallized_count: int

@dataclass
class DecayResult:
    decayed_count: int
    archived_count: int
```

## Testing Requirements

[SPEC-03.26] Unit tests SHALL verify consolidation merges related facts correctly.

[SPEC-03.27] Unit tests SHALL verify promotion respects confidence threshold.

[SPEC-03.28] Unit tests SHALL verify decay formula produces expected confidence values.

[SPEC-03.29] Property tests SHALL verify decay never increases confidence.

[SPEC-03.30] Property tests SHALL verify archived nodes are never deleted.

[SPEC-03.31] Integration tests SHALL verify lifecycle hooks trigger at correct times.
