"""
Deciduous-style reasoning traces with git integration.

Implements: Spec SPEC-04 Reasoning Traces
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .memory_store import MemoryStore
    from .trajectory import TrajectoryEvent


# =============================================================================
# Data Types
# =============================================================================


@dataclass
class DecisionNode:
    """
    A node in the decision graph.

    Implements: Spec SPEC-04.01-03
    """

    id: str
    decision_type: str  # goal, decision, option, action, outcome, observation
    content: str
    confidence: float = 0.5
    prompt: str | None = None
    files: list[str] = field(default_factory=list)
    branch: str | None = None
    commit_hash: str | None = None
    parent_id: str | None = None


@dataclass
class DecisionTree:
    """
    A tree of decision nodes.

    Implements: Spec SPEC-04.16
    """

    root: DecisionNode
    children: list[DecisionTree] = field(default_factory=list)


@dataclass
class RejectedOption:
    """An option that was rejected with a reason."""

    id: str
    content: str
    reason: str


# =============================================================================
# Schema Extension
# =============================================================================

DECISIONS_SCHEMA_SQL = """
-- Decisions extension table
CREATE TABLE IF NOT EXISTS decisions (
    node_id TEXT PRIMARY KEY REFERENCES nodes(id) ON DELETE CASCADE,
    decision_type TEXT NOT NULL CHECK(decision_type IN
        ('goal', 'decision', 'option', 'action', 'outcome', 'observation')),
    confidence REAL DEFAULT 0.5,
    prompt TEXT,
    files JSON DEFAULT '[]',
    branch TEXT,
    commit_hash TEXT,
    parent_id TEXT REFERENCES decisions(node_id)
);

CREATE INDEX IF NOT EXISTS idx_decisions_type ON decisions(decision_type);
CREATE INDEX IF NOT EXISTS idx_decisions_parent ON decisions(parent_id);
CREATE INDEX IF NOT EXISTS idx_decisions_commit ON decisions(commit_hash);
"""


# =============================================================================
# Valid Types
# =============================================================================

VALID_DECISION_TYPES = frozenset({
    "goal",
    "decision",
    "option",
    "action",
    "outcome",
    "observation",
})


# =============================================================================
# ReasoningTraces Class
# =============================================================================


class ReasoningTraces:
    """
    Deciduous-style reasoning traces with git integration.

    Implements: Spec SPEC-04

    Features:
    - Decision node types: goal, decision, option, action, outcome, observation
    - Hyperedge relationships: spawns, considers, chooses, rejects, implements, produces, informs
    - Git integration: branch and commit linking
    - Trajectory mapping: auto-create decisions from trajectory events
    """

    def __init__(self, store: MemoryStore):
        """
        Initialize reasoning traces.

        Args:
            store: MemoryStore instance for persistence
        """
        self.store = store
        self._trajectory_mapping_enabled = True
        self._init_schema()

    def _init_schema(self) -> None:
        """Initialize the decisions table schema."""
        import sqlite3

        conn = sqlite3.connect(self.store.db_path)
        try:
            conn.executescript(DECISIONS_SCHEMA_SQL)
            conn.commit()
        finally:
            conn.close()

    # =========================================================================
    # Node Creation (SPEC-04.01-03)
    # =========================================================================

    def create_goal(
        self,
        content: str,
        prompt: str | None = None,
        files: list[str] | None = None,
        branch: str | None = None,
    ) -> str:
        """
        Create a goal node.

        Implements: Spec SPEC-04.01

        Args:
            content: Goal description
            prompt: Optional user prompt that triggered this goal
            files: Optional list of relevant files
            branch: Optional git branch

        Returns:
            Node ID
        """
        return self._create_decision_node(
            decision_type="goal",
            content=content,
            prompt=prompt,
            files=files or [],
            branch=branch,
            parent_id=None,
        )

    def create_decision(
        self,
        goal_id: str,
        content: str,
        prompt: str | None = None,
    ) -> str:
        """
        Create a decision node under a goal.

        Implements: Spec SPEC-04.04

        Args:
            goal_id: Parent goal ID
            content: Decision description

        Returns:
            Node ID
        """
        decision_id = self._create_decision_node(
            decision_type="decision",
            content=content,
            prompt=prompt,
            parent_id=goal_id,
        )

        # Create "spawns" edge from goal to decision
        self.store.create_edge(
            edge_type="relation",
            label="spawns",
            members=[
                {"node_id": goal_id, "role": "subject", "position": 0},
                {"node_id": decision_id, "role": "object", "position": 1},
            ],
        )

        return decision_id

    def add_option(self, decision_id: str, content: str) -> str:
        """
        Add an option to a decision.

        Implements: Spec SPEC-04.05

        Args:
            decision_id: Parent decision ID
            content: Option description

        Returns:
            Node ID
        """
        option_id = self._create_decision_node(
            decision_type="option",
            content=content,
            parent_id=decision_id,
        )

        # Create "considers" edge from decision to option
        self.store.create_edge(
            edge_type="relation",
            label="considers",
            members=[
                {"node_id": decision_id, "role": "subject", "position": 0},
                {"node_id": option_id, "role": "object", "position": 1},
            ],
        )

        return option_id

    def choose_option(self, decision_id: str, option_id: str) -> None:
        """
        Mark an option as chosen.

        Implements: Spec SPEC-04.06

        Args:
            decision_id: Decision ID
            option_id: Chosen option ID
        """
        # Create "chooses" edge
        self.store.create_edge(
            edge_type="relation",
            label="chooses",
            members=[
                {"node_id": decision_id, "role": "subject", "position": 0},
                {"node_id": option_id, "role": "object", "position": 1},
            ],
        )

        # Update option confidence
        self.store.update_node(option_id, confidence=1.0)

    def reject_option(self, decision_id: str, option_id: str, reason: str) -> None:
        """
        Mark an option as rejected with reason.

        Implements: Spec SPEC-04.07

        Args:
            decision_id: Decision ID
            option_id: Rejected option ID
            reason: Reason for rejection
        """
        # Create "rejects" edge with reason as metadata
        self.store.create_edge(
            edge_type="relation",
            label="rejects",
            members=[
                {"node_id": decision_id, "role": "subject", "position": 0},
                {"node_id": option_id, "role": "object", "position": 1},
            ],
        )

        # Store reason in option metadata
        option_node = self.store.get_node(option_id)
        if option_node:
            metadata = option_node.metadata.copy()
            metadata["rejection_reason"] = reason
            self.store.update_node(option_id, metadata=metadata, confidence=0.0)

    def create_action(self, decision_id: str, content: str) -> str:
        """
        Create an action implementing a decision.

        Implements: Spec SPEC-04.08

        Args:
            decision_id: Parent decision ID
            content: Action description

        Returns:
            Node ID
        """
        action_id = self._create_decision_node(
            decision_type="action",
            content=content,
            parent_id=decision_id,
        )

        # Create "implements" edge
        self.store.create_edge(
            edge_type="relation",
            label="implements",
            members=[
                {"node_id": decision_id, "role": "subject", "position": 0},
                {"node_id": action_id, "role": "object", "position": 1},
            ],
        )

        return action_id

    def create_outcome(
        self,
        action_id: str,
        content: str,
        success: bool,
    ) -> str:
        """
        Create an outcome from an action.

        Implements: Spec SPEC-04.09

        Args:
            action_id: Parent action ID
            content: Outcome description
            success: Whether the outcome was successful

        Returns:
            Node ID
        """
        outcome_id = self._create_decision_node(
            decision_type="outcome",
            content=content,
            parent_id=action_id,
            confidence=1.0 if success else 0.3,
        )

        # Store success in metadata
        self._update_decision_metadata(outcome_id, {"success": success})

        # Create "produces" edge
        self.store.create_edge(
            edge_type="relation",
            label="produces",
            members=[
                {"node_id": action_id, "role": "subject", "position": 0},
                {"node_id": outcome_id, "role": "object", "position": 1},
            ],
        )

        return outcome_id

    def create_observation(self, content: str) -> str:
        """
        Create an observation node.

        Implements: Spec SPEC-04.10

        Args:
            content: Observation content

        Returns:
            Node ID
        """
        return self._create_decision_node(
            decision_type="observation",
            content=content,
        )

    def link_observation(self, observation_id: str, decision_id: str) -> None:
        """
        Link an observation to a decision (informs).

        Implements: Spec SPEC-04.10

        Args:
            observation_id: Observation node ID
            decision_id: Decision node ID
        """
        self.store.create_edge(
            edge_type="relation",
            label="informs",
            members=[
                {"node_id": observation_id, "role": "subject", "position": 0},
                {"node_id": decision_id, "role": "object", "position": 1},
            ],
        )

    # =========================================================================
    # Git Integration (SPEC-04.11-15)
    # =========================================================================

    def link_commit(self, decision_id: str, commit_hash: str) -> None:
        """
        Link a decision to a git commit.

        Implements: Spec SPEC-04.11

        Args:
            decision_id: Decision node ID
            commit_hash: Git commit hash
        """
        self._update_decision(decision_id, commit_hash=commit_hash)

    def get_decisions_for_commit(self, commit_hash: str) -> list[DecisionNode]:
        """
        Get all decisions linked to a commit.

        Implements: Spec SPEC-04.14

        Args:
            commit_hash: Git commit hash

        Returns:
            List of DecisionNode objects
        """
        import sqlite3

        conn = sqlite3.connect(self.store.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(
                """
                SELECT d.*, n.content
                FROM decisions d
                JOIN nodes n ON d.node_id = n.id
                WHERE d.commit_hash = ?
                """,
                (commit_hash,),
            )

            results = []
            for row in cursor.fetchall():
                results.append(self._row_to_decision_node(row))

            return results
        finally:
            conn.close()

    # =========================================================================
    # Query Interface (SPEC-04.16-19)
    # =========================================================================

    def get_decision_node(self, node_id: str) -> DecisionNode | None:
        """
        Get a decision node by ID.

        Args:
            node_id: Node ID

        Returns:
            DecisionNode or None if not found
        """
        import sqlite3

        conn = sqlite3.connect(self.store.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(
                """
                SELECT d.*, n.content
                FROM decisions d
                JOIN nodes n ON d.node_id = n.id
                WHERE d.node_id = ?
                """,
                (node_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None

            return self._row_to_decision_node(row)
        finally:
            conn.close()

    def get_decision_tree(self, goal_id: str) -> DecisionTree | None:
        """
        Get the decision tree rooted at a goal.

        Implements: Spec SPEC-04.16

        Args:
            goal_id: Root goal ID

        Returns:
            DecisionTree or None if goal not found
        """
        root = self.get_decision_node(goal_id)
        if root is None:
            return None

        return self._build_tree(root)

    def _build_tree(self, node: DecisionNode) -> DecisionTree:
        """Recursively build a decision tree."""
        # Get children via hyperedges
        related = self.store.get_related_nodes(node.id)
        children = []

        for rel in related:
            child_node = self.get_decision_node(rel.id)
            if child_node and child_node.parent_id == node.id:
                children.append(self._build_tree(child_node))

        return DecisionTree(root=node, children=children)

    def get_rejected_options(self, decision_id: str) -> list[RejectedOption]:
        """
        Get rejected options for a decision.

        Implements: Spec SPEC-04.17

        Args:
            decision_id: Decision ID

        Returns:
            List of RejectedOption objects
        """
        # Find "rejects" edges from this decision
        edges = self.store.query_edges(label="rejects")

        rejected = []
        for edge in edges:
            members = self.store.get_edge_members(edge.id)
            # Check if this edge is from our decision
            subject = next((m for m in members if m["role"] == "subject"), None)
            obj = next((m for m in members if m["role"] == "object"), None)

            if subject and subject["node_id"] == decision_id and obj:
                option_node = self.store.get_node(obj["node_id"])
                if option_node:
                    reason = option_node.metadata.get("rejection_reason", "")
                    rejected.append(
                        RejectedOption(
                            id=option_node.id,
                            content=option_node.content,
                            reason=reason,
                        )
                    )

        return rejected

    def get_outcome(self, goal_id: str) -> DecisionNode | None:
        """
        Get the outcome for a goal (if any).

        Implements: Spec SPEC-04.18

        Args:
            goal_id: Goal ID

        Returns:
            Outcome DecisionNode or None
        """
        # Traverse the tree to find outcome nodes
        tree = self.get_decision_tree(goal_id)
        if tree is None:
            return None

        return self._find_outcome_in_tree(tree)

    def _find_outcome_in_tree(self, tree: DecisionTree) -> DecisionNode | None:
        """Recursively search for an outcome node."""
        if tree.root.decision_type == "outcome":
            return tree.root

        for child in tree.children:
            result = self._find_outcome_in_tree(child)
            if result:
                return result

        return None

    def get_informing_observations(self, decision_id: str) -> list[DecisionNode]:
        """
        Get observations that inform a decision.

        Implements: Spec SPEC-04.19

        Args:
            decision_id: Decision ID

        Returns:
            List of observation DecisionNodes
        """
        # Find "informs" edges to this decision
        edges = self.store.query_edges(label="informs")

        observations = []
        for edge in edges:
            members = self.store.get_edge_members(edge.id)
            # Check if this edge points to our decision
            obj = next((m for m in members if m["role"] == "object"), None)
            subject = next((m for m in members if m["role"] == "subject"), None)

            if obj and obj["node_id"] == decision_id and subject:
                obs_node = self.get_decision_node(subject["node_id"])
                if obs_node and obs_node.decision_type == "observation":
                    observations.append(obs_node)

        return observations

    # =========================================================================
    # Trajectory Integration (SPEC-04.20-24)
    # =========================================================================

    def set_trajectory_mapping(self, enabled: bool) -> None:
        """
        Enable or disable trajectory-to-decision mapping.

        Implements: Spec SPEC-04.24

        Args:
            enabled: Whether to enable mapping
        """
        self._trajectory_mapping_enabled = enabled

    def from_trajectory_event(self, event: TrajectoryEvent) -> str | None:
        """
        Create a decision node from a trajectory event.

        Implements: Spec SPEC-04.20-23

        Args:
            event: TrajectoryEvent to convert

        Returns:
            Node ID or None if mapping is disabled
        """
        if not self._trajectory_mapping_enabled:
            return None

        from .trajectory import TrajectoryEventType

        # Map event type to decision type
        type_mapping = {
            TrajectoryEventType.RLM_START: "goal",
            TrajectoryEventType.RECURSE_START: "goal",
            TrajectoryEventType.ANALYZE: "decision",
            TrajectoryEventType.REASON: "decision",
            TrajectoryEventType.REPL_EXEC: "action",
            TrajectoryEventType.TOOL_USE: "action",
            TrajectoryEventType.FINAL: "outcome",
            TrajectoryEventType.RECURSE_END: "outcome",
            TrajectoryEventType.REPL_RESULT: "observation",
            TrajectoryEventType.ERROR: "observation",
        }

        decision_type = type_mapping.get(event.type, "observation")

        # Get parent from metadata
        parent_id = None
        if hasattr(event, "metadata") and event.metadata:
            parent_id = event.metadata.get("parent_id")

        # Create the node
        return self._create_decision_node(
            decision_type=decision_type,
            content=event.content,
            parent_id=parent_id,
        )

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _create_decision_node(
        self,
        decision_type: str,
        content: str,
        prompt: str | None = None,
        files: list[str] | None = None,
        branch: str | None = None,
        commit_hash: str | None = None,
        parent_id: str | None = None,
        confidence: float = 0.5,
    ) -> str:
        """Create a decision node in both nodes and decisions tables."""
        import sqlite3

        # Create the base node
        node_id = self.store.create_node(
            node_type="decision",
            content=content,
            subtype=decision_type,
            confidence=confidence,
            metadata={
                "decision_type": decision_type,
                "prompt": prompt,
                "files": files or [],
            },
        )

        # Create the decision extension record
        conn = sqlite3.connect(self.store.db_path)
        try:
            conn.execute(
                """
                INSERT INTO decisions (node_id, decision_type, confidence, prompt, files, branch, commit_hash, parent_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    node_id,
                    decision_type,
                    confidence,
                    prompt,
                    json.dumps(files or []),
                    branch,
                    commit_hash,
                    parent_id,
                ),
            )
            conn.commit()
        finally:
            conn.close()

        return node_id

    def _update_decision(self, node_id: str, **kwargs: Any) -> bool:
        """Update decision-specific fields."""
        import sqlite3

        if not kwargs:
            return False

        conn = sqlite3.connect(self.store.db_path)
        try:
            updates = []
            values = []
            for key, value in kwargs.items():
                if key == "files":
                    updates.append(f"{key} = ?")
                    values.append(json.dumps(value))
                else:
                    updates.append(f"{key} = ?")
                    values.append(value)

            values.append(node_id)

            query = f"UPDATE decisions SET {', '.join(updates)} WHERE node_id = ?"
            cursor = conn.execute(query, values)
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def _update_decision_metadata(self, node_id: str, extra: dict[str, Any]) -> None:
        """Update decision metadata in the nodes table."""
        node = self.store.get_node(node_id)
        if node:
            metadata = node.metadata.copy()
            metadata.update(extra)
            self.store.update_node(node_id, metadata=metadata)

    def _row_to_decision_node(self, row) -> DecisionNode:
        """Convert a database row to a DecisionNode."""
        files = row["files"]
        if isinstance(files, str):
            files = json.loads(files)

        return DecisionNode(
            id=row["node_id"],
            decision_type=row["decision_type"],
            content=row["content"],
            confidence=row["confidence"] or 0.5,
            prompt=row["prompt"],
            files=files or [],
            branch=row["branch"],
            commit_hash=row["commit_hash"],
            parent_id=row["parent_id"],
        )


__all__ = [
    "ReasoningTraces",
    "DecisionNode",
    "DecisionTree",
    "RejectedOption",
]
