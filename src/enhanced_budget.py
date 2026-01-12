"""
Enhanced budget tracking for RLM-Claude-Code.

Implements: Spec SPEC-05 - Enhanced Budget Tracking
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .cost_tracker import MODEL_COSTS, CostComponent
from .trajectory import TrajectoryEvent, TrajectoryEventType

if TYPE_CHECKING:
    pass


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class EnhancedBudgetMetrics:
    """
    Enhanced budget metrics.

    Implements: Spec SPEC-05.01-05
    """

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
    """
    Budget limits configuration.

    Implements: Spec SPEC-05.06-10
    """

    max_cost_per_task: float = 5.0
    max_cost_per_session: float = 25.0
    max_tokens_per_call: int = 8000
    max_recursive_calls: int = 10
    max_repl_executions: int = 50
    cost_alert_threshold: float = 0.8
    token_alert_threshold: float = 0.75


@dataclass
class BudgetAlert:
    """
    Budget alert.

    Implements: Spec SPEC-05.14
    """

    level: str  # "warning" | "critical"
    message: str
    metric: str
    current_value: float
    threshold: float
    timestamp: float = field(default_factory=time.time)


# =============================================================================
# Default Configuration
# =============================================================================

DEFAULT_LIMITS = BudgetLimits()


# =============================================================================
# EnhancedBudgetTracker Class
# =============================================================================


class EnhancedBudgetTracker:
    """
    Enhanced budget tracking with granular metrics, limits, and alerts.

    Implements: Spec SPEC-05

    Features:
    - Track cached_tokens, sub_call_count, repl_executions
    - Track wall_clock_seconds, max_depth_reached
    - Granular per-task and per-session limits
    - Alert system with warning/critical levels
    - Limit enforcement with force bypass
    - Configuration from file with per-mode overrides
    """

    def __init__(
        self,
        config_path: str | None = None,
        mode: str = "balanced",
    ):
        """
        Initialize enhanced budget tracker.

        Args:
            config_path: Optional path to config file
            mode: Operating mode (fast, balanced, thorough)
        """
        self.mode = mode
        self._limits = self._load_config(config_path, mode)
        self._metrics = EnhancedBudgetMetrics()
        self._alerts: list[BudgetAlert] = []
        self._emitted_alerts: set[str] = set()  # Prevent duplicate alerts
        self._alert_callbacks: list[Callable[[BudgetAlert], None]] = []
        self._alert_event_callbacks: list[Callable[[TrajectoryEvent], None]] = []

        # Task tracking
        self._current_task_id: str | None = None
        self._task_cost: float = 0.0

        # Timing
        self._timing_start: float | None = None

    def _load_config(
        self,
        config_path: str | None,
        mode: str,
    ) -> BudgetLimits:
        """
        Load configuration from file or use defaults.

        Implements: Spec SPEC-05.20-21
        """
        limits = BudgetLimits()

        # Try loading from specified path
        paths_to_try = []
        if config_path:
            paths_to_try.append(Path(config_path))

        # Try default location
        default_path = Path.home() / ".claude" / "rlm-config.json"
        paths_to_try.append(default_path)

        config_data: dict[str, Any] = {}
        for path in paths_to_try:
            if path.exists():
                try:
                    with open(path) as f:
                        config_data = json.load(f)
                        break
                except (json.JSONDecodeError, OSError):
                    pass

        if "budget" not in config_data:
            return limits

        budget_config = config_data["budget"]

        # Apply base configuration
        if "max_cost_per_task" in budget_config:
            limits.max_cost_per_task = budget_config["max_cost_per_task"]
        if "max_cost_per_session" in budget_config:
            limits.max_cost_per_session = budget_config["max_cost_per_session"]
        if "max_tokens_per_call" in budget_config:
            limits.max_tokens_per_call = budget_config["max_tokens_per_call"]
        if "max_recursive_calls" in budget_config:
            limits.max_recursive_calls = budget_config["max_recursive_calls"]
        if "max_repl_executions" in budget_config:
            limits.max_repl_executions = budget_config["max_repl_executions"]
        if "cost_alert_threshold" in budget_config:
            limits.cost_alert_threshold = budget_config["cost_alert_threshold"]
        if "token_alert_threshold" in budget_config:
            limits.token_alert_threshold = budget_config["token_alert_threshold"]

        # Apply per-mode overrides (SPEC-05.21)
        if "modes" in budget_config and mode in budget_config["modes"]:
            mode_config = budget_config["modes"][mode]
            if "max_cost_per_task" in mode_config:
                limits.max_cost_per_task = mode_config["max_cost_per_task"]
            if "max_cost_per_session" in mode_config:
                limits.max_cost_per_session = mode_config["max_cost_per_session"]
            if "max_tokens_per_call" in mode_config:
                limits.max_tokens_per_call = mode_config["max_tokens_per_call"]
            if "max_recursive_calls" in mode_config:
                limits.max_recursive_calls = mode_config["max_recursive_calls"]
            if "max_repl_executions" in mode_config:
                limits.max_repl_executions = mode_config["max_repl_executions"]
            if "cost_alert_threshold" in mode_config:
                limits.cost_alert_threshold = mode_config["cost_alert_threshold"]
            if "token_alert_threshold" in mode_config:
                limits.token_alert_threshold = mode_config["token_alert_threshold"]

        return limits

    # =========================================================================
    # Configuration
    # =========================================================================

    def set_limits(self, limits: BudgetLimits) -> None:
        """Set budget limits."""
        self._limits = limits

    def get_limits(self) -> BudgetLimits:
        """Get current budget limits."""
        return self._limits

    # =========================================================================
    # Recording Methods
    # =========================================================================

    def record_llm_call(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str,
        component: CostComponent,
        cached_tokens: int = 0,
        latency_ms: float = 0.0,
    ) -> list[BudgetAlert]:
        """
        Record LLM call and return any triggered alerts.

        Implements: Spec SPEC-05.01-02
        """
        # Update token metrics
        self._metrics.input_tokens += input_tokens
        self._metrics.output_tokens += output_tokens
        self._metrics.cached_tokens += cached_tokens

        # Calculate cost
        costs = MODEL_COSTS.get(model, MODEL_COSTS.get("sonnet", {}))
        input_cost = (input_tokens / 1_000_000) * costs.get("input", 3.0)
        output_cost = (output_tokens / 1_000_000) * costs.get("output", 15.0)
        call_cost = input_cost + output_cost

        self._metrics.total_cost_usd += call_cost
        self._task_cost += call_cost

        # Track sub-calls (SPEC-05.02)
        if component == CostComponent.RECURSIVE_CALL:
            self._metrics.sub_call_count += 1

        # Check for alerts
        return self._check_and_emit_alerts()

    def record_repl_execution(self) -> list[BudgetAlert]:
        """
        Record REPL execution and return any triggered alerts.

        Implements: Spec SPEC-05.03
        """
        self._metrics.repl_executions += 1
        return self._check_and_emit_alerts()

    def record_depth(self, depth: int) -> None:
        """
        Record current recursion depth.

        Implements: Spec SPEC-05.05
        """
        self._metrics.recursion_depth = depth
        if depth > self._metrics.max_depth_reached:
            self._metrics.max_depth_reached = depth

    # =========================================================================
    # Timing Methods (SPEC-05.04)
    # =========================================================================

    def start_timing(self) -> None:
        """Start wall clock timing."""
        self._timing_start = time.time()
        if self._metrics.session_start == 0.0:
            self._metrics.session_start = self._timing_start

    def stop_timing(self) -> None:
        """Stop wall clock timing and update metrics."""
        if self._timing_start is not None:
            elapsed = time.time() - self._timing_start
            self._metrics.wall_clock_seconds += elapsed
            self._metrics.session_duration_seconds = (
                time.time() - self._metrics.session_start
            )
            self._timing_start = None

    # =========================================================================
    # Task Management (SPEC-05.25)
    # =========================================================================

    def start_task(self, task_id: str) -> None:
        """Start a new task."""
        self._current_task_id = task_id
        self._task_cost = 0.0

    def end_task(self) -> None:
        """End current task."""
        self._current_task_id = None

    # =========================================================================
    # Limit Checks (SPEC-05.16-19)
    # =========================================================================

    def can_make_llm_call(self, force: bool = False) -> tuple[bool, str | None]:
        """
        Check if LLM call is allowed.

        Implements: Spec SPEC-05.16, SPEC-05.19

        Args:
            force: Bypass limits for debugging

        Returns:
            (allowed, reason) tuple
        """
        if force:
            return True, None

        # Check task cost limit
        if self._task_cost >= self._limits.max_cost_per_task:
            return False, (
                f"Task cost limit exceeded: ${self._task_cost:.2f} >= "
                f"${self._limits.max_cost_per_task:.2f}"
            )

        # Check session cost limit
        if self._metrics.total_cost_usd >= self._limits.max_cost_per_session:
            return False, (
                f"Session cost limit exceeded: ${self._metrics.total_cost_usd:.2f} >= "
                f"${self._limits.max_cost_per_session:.2f}"
            )

        return True, None

    def can_recurse(self, force: bool = False) -> tuple[bool, str | None]:
        """
        Check if recursion is allowed.

        Implements: Spec SPEC-05.17, SPEC-05.19

        Args:
            force: Bypass limits for debugging

        Returns:
            (allowed, reason) tuple
        """
        if force:
            return True, None

        if self._metrics.sub_call_count >= self._limits.max_recursive_calls:
            return False, (
                f"Recursive call limit reached: {self._metrics.sub_call_count} >= "
                f"{self._limits.max_recursive_calls}"
            )

        return True, None

    def can_execute_repl(self, force: bool = False) -> tuple[bool, str | None]:
        """
        Check if REPL execution is allowed.

        Implements: Spec SPEC-05.18, SPEC-05.19

        Args:
            force: Bypass limits for debugging

        Returns:
            (allowed, reason) tuple
        """
        if force:
            return True, None

        if self._metrics.repl_executions >= self._limits.max_repl_executions:
            return False, (
                f"REPL execution limit reached: {self._metrics.repl_executions} >= "
                f"{self._limits.max_repl_executions}"
            )

        return True, None

    # =========================================================================
    # Alert System (SPEC-05.11-15)
    # =========================================================================

    def check_limits(self) -> list[BudgetAlert]:
        """
        Check all limits and return current alerts.

        Implements: Spec SPEC-05.11-13
        """
        return self._check_and_emit_alerts()

    def _check_and_emit_alerts(self) -> list[BudgetAlert]:
        """Check all limits and emit any new alerts."""
        new_alerts: list[BudgetAlert] = []

        # Cost alert (SPEC-05.11)
        cost_fraction = self._task_cost / self._limits.max_cost_per_task
        if cost_fraction >= 1.0:
            alert = self._create_alert(
                level="critical",
                metric="cost",
                current_value=self._task_cost,
                threshold=self._limits.max_cost_per_task,
                message=(
                    f"Task cost limit exceeded: ${self._task_cost:.2f} / "
                    f"${self._limits.max_cost_per_task:.2f}"
                ),
            )
            if alert:
                new_alerts.append(alert)
        elif cost_fraction >= self._limits.cost_alert_threshold:
            alert = self._create_alert(
                level="warning",
                metric="cost",
                current_value=self._task_cost,
                threshold=self._limits.max_cost_per_task * self._limits.cost_alert_threshold,
                message=(
                    f"Approaching task cost limit: ${self._task_cost:.2f} / "
                    f"${self._limits.max_cost_per_task:.2f} ({cost_fraction:.0%})"
                ),
            )
            if alert:
                new_alerts.append(alert)

        # Recursive call warning (SPEC-05.13)
        remaining_calls = self._limits.max_recursive_calls - self._metrics.sub_call_count
        if remaining_calls <= 0:
            alert = self._create_alert(
                level="critical",
                metric="recursive_calls",
                current_value=self._metrics.sub_call_count,
                threshold=self._limits.max_recursive_calls,
                message=(
                    f"Recursive call limit reached: {self._metrics.sub_call_count} / "
                    f"{self._limits.max_recursive_calls}"
                ),
            )
            if alert:
                new_alerts.append(alert)
        elif remaining_calls <= 2:
            alert = self._create_alert(
                level="warning",
                metric="recursive_calls",
                current_value=self._metrics.sub_call_count,
                threshold=self._limits.max_recursive_calls - 2,
                message=(
                    f"Approaching recursive call limit: {self._metrics.sub_call_count} / "
                    f"{self._limits.max_recursive_calls} ({remaining_calls} remaining)"
                ),
            )
            if alert:
                new_alerts.append(alert)

        # REPL execution warning
        remaining_repl = self._limits.max_repl_executions - self._metrics.repl_executions
        if remaining_repl <= 0:
            alert = self._create_alert(
                level="critical",
                metric="repl_executions",
                current_value=self._metrics.repl_executions,
                threshold=self._limits.max_repl_executions,
                message=(
                    f"REPL execution limit reached: {self._metrics.repl_executions} / "
                    f"{self._limits.max_repl_executions}"
                ),
            )
            if alert:
                new_alerts.append(alert)
        elif remaining_repl <= 5:
            alert = self._create_alert(
                level="warning",
                metric="repl_executions",
                current_value=self._metrics.repl_executions,
                threshold=self._limits.max_repl_executions - 5,
                message=(
                    f"Approaching REPL execution limit: {self._metrics.repl_executions} / "
                    f"{self._limits.max_repl_executions} ({remaining_repl} remaining)"
                ),
            )
            if alert:
                new_alerts.append(alert)

        return new_alerts

    def _create_alert(
        self,
        level: str,
        metric: str,
        current_value: float,
        threshold: float,
        message: str,
    ) -> BudgetAlert | None:
        """Create alert if not already emitted."""
        alert_key = f"{level}:{metric}"
        if alert_key in self._emitted_alerts:
            return None

        self._emitted_alerts.add(alert_key)

        alert = BudgetAlert(
            level=level,
            metric=metric,
            current_value=current_value,
            threshold=threshold,
            message=message,
        )

        self._alerts.append(alert)

        # Notify callbacks
        for callback in self._alert_callbacks:
            callback(alert)

        # Emit as TrajectoryEvent (SPEC-05.15)
        event = TrajectoryEvent(
            type=TrajectoryEventType.BUDGET_ALERT,
            depth=0,
            content=message,
            metadata={
                "level": level,
                "metric": metric,
                "current_value": current_value,
                "threshold": threshold,
            },
        )

        for callback in self._alert_event_callbacks:
            callback(event)

        return alert

    def on_alert(self, callback: Callable[[BudgetAlert], None]) -> None:
        """Register callback for budget alerts."""
        self._alert_callbacks.append(callback)

    def on_alert_event(self, callback: Callable[[TrajectoryEvent], None]) -> None:
        """Register callback for budget alert trajectory events."""
        self._alert_event_callbacks.append(callback)

    # =========================================================================
    # Metrics Access
    # =========================================================================

    def get_metrics(self) -> EnhancedBudgetMetrics:
        """
        Get current budget metrics.

        Returns:
            EnhancedBudgetMetrics with current values
        """
        # Update session duration if timing is active
        if self._timing_start is not None:
            elapsed = time.time() - self._timing_start
            self._metrics.wall_clock_seconds += elapsed
            self._timing_start = time.time()

        if self._metrics.session_start > 0:
            self._metrics.session_duration_seconds = (
                time.time() - self._metrics.session_start
            )

        return EnhancedBudgetMetrics(
            input_tokens=self._metrics.input_tokens,
            output_tokens=self._metrics.output_tokens,
            cached_tokens=self._metrics.cached_tokens,
            total_cost_usd=self._metrics.total_cost_usd,
            recursion_depth=self._metrics.recursion_depth,
            max_depth_reached=self._metrics.max_depth_reached,
            sub_call_count=self._metrics.sub_call_count,
            repl_executions=self._metrics.repl_executions,
            session_start=self._metrics.session_start,
            session_duration_seconds=self._metrics.session_duration_seconds,
            wall_clock_seconds=self._metrics.wall_clock_seconds,
        )

    def get_alerts(self) -> list[BudgetAlert]:
        """Get all alerts that have been emitted."""
        return self._alerts.copy()

    def reset(self) -> None:
        """Reset all tracking."""
        self._metrics = EnhancedBudgetMetrics()
        self._alerts.clear()
        self._emitted_alerts.clear()
        self._current_task_id = None
        self._task_cost = 0.0
        self._timing_start = None


__all__ = [
    "BudgetAlert",
    "BudgetLimits",
    "EnhancedBudgetMetrics",
    "EnhancedBudgetTracker",
]
