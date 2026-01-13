"""
User interaction and steering support for RLM sessions.

Implements: SPEC-12.06

Contains:
- SteeringPoint for decision points
- InteractiveOrchestrator for user interaction
- Auto-steering policy
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol


class SteeringDecision(Enum):
    """Decisions available at steering points."""

    CONTINUE = "continue"
    STOP = "stop"
    ADJUST_DEPTH = "adjust_depth"
    ADJUST_MODEL = "adjust_model"
    PROVIDE_CONTEXT = "provide_context"
    SKIP_STEP = "skip_step"


@dataclass
class SteeringPoint:
    """
    A point where user steering is available.

    Implements: SPEC-12.06

    Captures the decision context and available options
    at points where user intervention might be valuable.
    """

    turn: int
    depth: int
    decision_type: str
    options: list[str]
    context: str
    current_state: dict[str, Any] = field(default_factory=dict)
    recommendation: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "turn": self.turn,
            "depth": self.depth,
            "decision_type": self.decision_type,
            "options": self.options,
            "context": self.context,
            "current_state": self.current_state,
            "recommendation": self.recommendation,
            "metadata": self.metadata,
        }


class SteeringCallback(Protocol):
    """Protocol for user steering callbacks."""

    def __call__(self, point: SteeringPoint) -> SteeringDecision:
        """
        Get user decision at a steering point.

        Args:
            point: The steering point context

        Returns:
            User's steering decision
        """
        ...


class AutoSteeringPolicy:
    """
    Automatic steering policy for non-interactive use.

    Implements: SPEC-12.06

    Provides sensible defaults when user interaction is not available.
    """

    def __init__(
        self,
        max_turns_before_stop: int = 50,
        max_depth: int = 3,
        cost_threshold_usd: float = 1.0,
        confidence_threshold: float = 0.3,
    ):
        """
        Initialize auto-steering policy.

        Args:
            max_turns_before_stop: Maximum turns before forcing stop
            max_depth: Maximum recursion depth
            cost_threshold_usd: Maximum cost before stopping
            confidence_threshold: Minimum confidence to continue
        """
        self.max_turns_before_stop = max_turns_before_stop
        self.max_depth = max_depth
        self.cost_threshold_usd = cost_threshold_usd
        self.confidence_threshold = confidence_threshold

    def decide(self, point: SteeringPoint) -> SteeringDecision:
        """
        Make automatic steering decision.

        Args:
            point: The steering point context

        Returns:
            Automatic steering decision
        """
        # Check turn limit
        if point.turn >= self.max_turns_before_stop:
            return SteeringDecision.STOP

        # Check depth limit
        if point.depth >= self.max_depth:
            return SteeringDecision.STOP

        # Check cost if available
        cost = point.current_state.get("cost_usd", 0.0)
        if cost >= self.cost_threshold_usd:
            return SteeringDecision.STOP

        # Check confidence if available
        confidence = point.current_state.get("confidence", 1.0)
        if confidence < self.confidence_threshold:
            return SteeringDecision.ADJUST_DEPTH

        # Default: continue
        return SteeringDecision.CONTINUE

    def __call__(self, point: SteeringPoint) -> SteeringDecision:
        """Make policy callable for use as SteeringCallback."""
        return self.decide(point)


class InteractiveOrchestrator:
    """
    Orchestrator wrapper with user interaction support.

    Implements: SPEC-12.06

    Provides:
    - Steering point detection
    - User callback integration
    - Auto-steering fallback
    """

    def __init__(
        self,
        callback: SteeringCallback | None = None,
        auto_policy: AutoSteeringPolicy | None = None,
    ):
        """
        Initialize interactive orchestrator.

        Args:
            callback: Optional user steering callback
            auto_policy: Auto-steering policy for non-interactive use
        """
        self.callback = callback
        self.auto_policy = auto_policy or AutoSteeringPolicy()
        self._steering_history: list[tuple[SteeringPoint, SteeringDecision]] = []

    def should_steer(
        self,
        turn: int,
        depth: int,
        confidence: float | None = None,
        cost_usd: float | None = None,
    ) -> bool:
        """
        Check if steering should occur at this point.

        Args:
            turn: Current turn number
            depth: Current recursion depth
            confidence: Optional current confidence
            cost_usd: Optional accumulated cost

        Returns:
            True if steering point should trigger
        """
        # Steer at depth transitions
        if depth > 0 and turn % 5 == 0:
            return True

        # Steer on low confidence
        if confidence is not None and confidence < 0.4:
            return True

        # Steer on high cost
        if cost_usd is not None and cost_usd > 0.5:
            return True

        # Steer at turn milestones
        if turn in {10, 20, 30, 40}:
            return True

        return False

    def create_steering_point(
        self,
        turn: int,
        depth: int,
        context: str,
        decision_type: str = "continue_or_stop",
        current_state: dict[str, Any] | None = None,
    ) -> SteeringPoint:
        """
        Create a steering point for user decision.

        Args:
            turn: Current turn number
            depth: Current recursion depth
            context: Human-readable context description
            decision_type: Type of decision needed
            current_state: Current orchestration state

        Returns:
            SteeringPoint for user decision
        """
        # Determine options based on decision type
        if decision_type == "continue_or_stop":
            options = ["continue", "stop", "adjust_depth"]
        elif decision_type == "model_selection":
            options = ["haiku", "sonnet", "opus"]
        elif decision_type == "depth_adjustment":
            options = ["increase", "decrease", "maintain"]
        else:
            options = ["continue", "stop"]

        return SteeringPoint(
            turn=turn,
            depth=depth,
            decision_type=decision_type,
            options=options,
            context=context,
            current_state=current_state or {},
        )

    def get_decision(self, point: SteeringPoint) -> SteeringDecision:
        """
        Get steering decision from user or auto-policy.

        Args:
            point: The steering point

        Returns:
            Steering decision
        """
        if self.callback is not None:
            try:
                decision = self.callback(point)
            except Exception:
                # Fallback to auto on callback error
                decision = self.auto_policy.decide(point)
        else:
            decision = self.auto_policy.decide(point)

        # Record decision
        self._steering_history.append((point, decision))

        return decision

    def get_steering_history(self) -> list[tuple[SteeringPoint, SteeringDecision]]:
        """Get history of steering decisions."""
        return self._steering_history.copy()

    def clear_history(self) -> None:
        """Clear steering history."""
        self._steering_history.clear()


__all__ = [
    "AutoSteeringPolicy",
    "InteractiveOrchestrator",
    "SteeringCallback",
    "SteeringDecision",
    "SteeringPoint",
]
