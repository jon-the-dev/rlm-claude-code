# SPEC-05: Enhanced Budget Tracking

## Overview

Extend existing cost tracking with granular metrics, limits, and alerts from recurse.

## Requirements

### Enhanced Metrics

[SPEC-05.01] The system SHALL track cached_tokens in addition to input/output tokens.

[SPEC-05.02] The system SHALL track sub_call_count (number of recursive LLM calls).

[SPEC-05.03] The system SHALL track repl_executions (number of REPL code executions).

[SPEC-05.04] The system SHALL track wall_clock_seconds (total elapsed time including waits).

[SPEC-05.05] The system SHALL track max_depth_reached (deepest recursion level hit).

### Granular Limits

[SPEC-05.06] The system SHALL support max_cost_per_task limit (default: $5.00).

[SPEC-05.07] The system SHALL support max_cost_per_session limit (default: $25.00).

[SPEC-05.08] The system SHALL support max_tokens_per_call limit (default: 8000).

[SPEC-05.09] The system SHALL support max_recursive_calls limit (default: 10).

[SPEC-05.10] The system SHALL support max_repl_executions limit (default: 50).

### Alert System

[SPEC-05.11] The system SHALL emit warning alert at cost_alert_threshold (default: 80% of budget).

[SPEC-05.12] The system SHALL emit warning alert at token_alert_threshold (default: 75% of limit).

[SPEC-05.13] The system SHALL emit warning when approaching sub_call limit (within 2 of max).

[SPEC-05.14] Alerts SHALL include: level ("warning", "critical"), message, current_value, threshold.

[SPEC-05.15] Alerts SHALL be emitted as TrajectoryEvents of type BUDGET_ALERT.

### Limit Enforcement

[SPEC-05.16] When max_cost_per_task is exceeded, the system SHALL refuse new LLM calls for that task.

[SPEC-05.17] When max_recursive_calls is reached, the system SHALL refuse deeper recursion.

[SPEC-05.18] When max_repl_executions is reached, the system SHALL refuse REPL execution.

[SPEC-05.19] Limit enforcement SHALL be bypassable via `--force` flag for debugging.

### Configuration

[SPEC-05.20] Budget limits SHALL be configurable via `~/.claude/rlm-config.json`:
```json
{
  "budget": {
    "max_cost_per_task": 5.0,
    "max_cost_per_session": 25.0,
    "max_tokens_per_call": 8000,
    "max_recursive_calls": 10,
    "max_repl_executions": 50,
    "cost_alert_threshold": 0.8,
    "token_alert_threshold": 0.75
  }
}
```

[SPEC-05.21] Budget configuration SHALL support per-mode overrides (fast, balanced, thorough).

## Interface

```python
@dataclass
class EnhancedBudgetMetrics:
    # Token tracking
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0

    # Cost
    total_cost_usd: float = 0.0

    # Execution metrics
    recursion_depth: int = 0
    max_depth_reached: int = 0
    sub_call_count: int = 0
    repl_executions: int = 0

    # Time
    session_start: float = 0.0
    session_duration_seconds: float = 0.0
    wall_clock_seconds: float = 0.0

@dataclass
class BudgetLimits:
    max_cost_per_task: float = 5.0
    max_cost_per_session: float = 25.0
    max_tokens_per_call: int = 8000
    max_recursive_calls: int = 10
    max_repl_executions: int = 50
    cost_alert_threshold: float = 0.8
    token_alert_threshold: float = 0.75

@dataclass
class BudgetAlert:
    level: str  # "warning" | "critical"
    message: str
    metric: str
    current_value: float
    threshold: float

class EnhancedBudgetTracker:
    def record_llm_call(self, usage: TokenUsage) -> list[BudgetAlert]:
        """Record LLM call and return any triggered alerts."""

    def record_repl_execution(self) -> list[BudgetAlert]:
        """Record REPL execution and return any triggered alerts."""

    def check_limits(self) -> list[BudgetAlert]:
        """Check all limits and return current alerts."""

    def can_make_llm_call(self) -> tuple[bool, str | None]:
        """Check if LLM call is allowed, return (allowed, reason)."""

    def can_recurse(self) -> tuple[bool, str | None]:
        """Check if recursion is allowed, return (allowed, reason)."""

    def get_metrics(self) -> EnhancedBudgetMetrics:
        """Get current budget metrics."""
```

## Testing Requirements

[SPEC-05.22] Unit tests SHALL verify all new metrics are tracked correctly.

[SPEC-05.23] Unit tests SHALL verify alerts trigger at correct thresholds.

[SPEC-05.24] Unit tests SHALL verify limit enforcement blocks operations.

[SPEC-05.25] Integration tests SHALL verify budget persists across task boundaries.

[SPEC-05.26] Property tests SHALL verify budget metrics never decrease unexpectedly.
