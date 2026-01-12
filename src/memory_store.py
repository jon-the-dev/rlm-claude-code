"""
Persistent hypergraph memory with SQLite storage.

Implements: Spec SPEC-02 Memory Foundation
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class Node:
    """
    A node in the hypergraph memory.

    Implements: Spec SPEC-02.05-10
    """

    id: str
    type: str  # entity, fact, experience, decision, snippet
    content: str
    tier: str = "task"  # task, session, longterm, archive
    confidence: float = 0.5
    subtype: str | None = None
    embedding: bytes | None = None
    provenance: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: int = 0
    updated_at: int = 0
    last_accessed: int = 0
    access_count: int = 0


@dataclass
class Hyperedge:
    """
    A hyperedge connecting multiple nodes.

    Implements: Spec SPEC-02.11-13
    """

    id: str
    type: str  # relation, composition, causation, context
    label: str | None = None
    weight: float = 1.0


@dataclass
class EvolutionLogEntry:
    """Entry in the evolution log for tier transitions."""

    id: int
    timestamp: int
    operation: str
    node_ids: list[str]
    from_tier: str | None
    to_tier: str | None
    reasoning: str | None


# =============================================================================
# Schema Definition
# =============================================================================

SCHEMA_SQL = """
-- Nodes table
CREATE TABLE IF NOT EXISTS nodes (
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

-- Hyperedges table
CREATE TABLE IF NOT EXISTS hyperedges (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL CHECK(type IN ('relation', 'composition', 'causation', 'context')),
    label TEXT,
    weight REAL DEFAULT 1.0 CHECK(weight >= 0.0)
);

-- Membership table (many-to-many)
CREATE TABLE IF NOT EXISTS membership (
    hyperedge_id TEXT NOT NULL REFERENCES hyperedges(id) ON DELETE CASCADE,
    node_id TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    position INTEGER DEFAULT 0,
    PRIMARY KEY (hyperedge_id, node_id, role)
);

-- Evolution log table
CREATE TABLE IF NOT EXISTS evolution_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp INTEGER DEFAULT (strftime('%s', 'now') * 1000),
    operation TEXT NOT NULL,
    node_ids JSON NOT NULL,
    from_tier TEXT,
    to_tier TEXT,
    reasoning TEXT
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_nodes_tier ON nodes(tier);
CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(type);
CREATE INDEX IF NOT EXISTS idx_nodes_confidence ON nodes(confidence);
CREATE INDEX IF NOT EXISTS idx_nodes_last_accessed ON nodes(last_accessed);
CREATE INDEX IF NOT EXISTS idx_membership_node ON membership(node_id);
CREATE INDEX IF NOT EXISTS idx_membership_edge ON membership(hyperedge_id);
"""


# =============================================================================
# MemoryStore Class
# =============================================================================


class MemoryStore:
    """
    Persistent hypergraph memory store using SQLite.

    Implements: Spec SPEC-02.01-26

    Features:
    - SQLite with WAL mode for concurrent access
    - Node CRUD operations with tier system
    - Hyperedge support for many-to-many relationships
    - Evolution logging for tier transitions
    """

    # Valid node types (SPEC-02.05)
    VALID_NODE_TYPES = frozenset({"entity", "fact", "experience", "decision", "snippet"})

    # Valid tiers (SPEC-02.17)
    VALID_TIERS = frozenset({"task", "session", "longterm", "archive"})

    # Valid edge types (SPEC-02.11)
    VALID_EDGE_TYPES = frozenset({"relation", "composition", "causation", "context"})

    # Valid membership roles (SPEC-02.16)
    VALID_ROLES = frozenset(
        {"subject", "object", "context", "participant", "cause", "effect"}
    )

    def __init__(self, db_path: str | None = None):
        """
        Initialize memory store.

        Args:
            db_path: Path to SQLite database. If None, uses default or env var.

        Implements: Spec SPEC-02.01-04
        """
        if db_path is None:
            db_path = os.environ.get("RLM_MEMORY_DB") or self._get_default_db_path()

        self.db_path = db_path
        self._ensure_directory()
        self._init_database()

    def _get_default_db_path(self) -> str:
        """
        Get default database path.

        Implements: Spec SPEC-02.02
        """
        return str(Path.home() / ".claude" / "rlm-memory.db")

    def _ensure_directory(self) -> None:
        """Ensure the database directory exists."""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

    def _init_database(self) -> None:
        """
        Initialize database with schema.

        Implements: Spec SPEC-02.04 (WAL mode)
        """
        conn = sqlite3.connect(self.db_path)
        try:
            # Enable WAL mode for concurrent access
            conn.execute("PRAGMA journal_mode=WAL")
            # Enable foreign keys
            conn.execute("PRAGMA foreign_keys=ON")
            # Create schema
            conn.executescript(SCHEMA_SQL)
            conn.commit()
        finally:
            conn.close()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with proper settings."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    # =========================================================================
    # Node CRUD Operations (SPEC-02.20-24)
    # =========================================================================

    def create_node(
        self,
        node_type: str,
        content: str,
        tier: str = "task",
        confidence: float = 0.5,
        subtype: str | None = None,
        embedding: bytes | None = None,
        provenance: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Create a new node.

        Implements: Spec SPEC-02.20

        Args:
            node_type: Type of node (entity, fact, experience, decision, snippet)
            content: Node content
            tier: Memory tier (task, session, longterm, archive)
            confidence: Confidence score [0.0, 1.0]
            subtype: Optional subtype
            embedding: Optional embedding blob
            provenance: Optional provenance string
            metadata: Optional metadata dict

        Returns:
            Node ID (UUID)

        Raises:
            ValueError: If parameters are invalid
        """
        # Validate node type (SPEC-02.05)
        if node_type not in self.VALID_NODE_TYPES:
            raise ValueError(
                f"Invalid node type: {node_type}. Must be one of: {self.VALID_NODE_TYPES}"
            )

        # Validate tier (SPEC-02.17)
        if tier not in self.VALID_TIERS:
            raise ValueError(
                f"Invalid tier: {tier}. Must be one of: {self.VALID_TIERS}"
            )

        # Validate confidence (SPEC-02.10)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got: {confidence}")

        # Generate UUID (SPEC-02.09)
        node_id = str(uuid.uuid4())

        # Serialize metadata
        metadata_json = json.dumps(metadata or {})

        conn = self._get_connection()
        try:
            conn.execute(
                """
                INSERT INTO nodes (id, type, subtype, content, embedding, tier,
                                   confidence, provenance, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    node_id,
                    node_type,
                    subtype,
                    content,
                    embedding,
                    tier,
                    confidence,
                    provenance,
                    metadata_json,
                ),
            )
            conn.commit()
        finally:
            conn.close()

        return node_id

    def get_node(self, node_id: str, include_archived: bool = False) -> Node | None:
        """
        Get a node by ID.

        Implements: Spec SPEC-02.21

        Args:
            node_id: Node ID to retrieve
            include_archived: If True, include archived nodes

        Returns:
            Node object or None if not found
        """
        conn = self._get_connection()
        try:
            # Update access tracking
            conn.execute(
                """
                UPDATE nodes
                SET last_accessed = strftime('%s', 'now') * 1000,
                    access_count = access_count + 1
                WHERE id = ?
                """,
                (node_id,),
            )
            conn.commit()

            # Build query
            query = "SELECT * FROM nodes WHERE id = ?"
            if not include_archived:
                query += " AND tier != 'archive'"

            cursor = conn.execute(query, (node_id,))
            row = cursor.fetchone()

            if row is None:
                return None

            return self._row_to_node(row)
        finally:
            conn.close()

    def update_node(self, node_id: str, **kwargs: Any) -> bool:
        """
        Update a node.

        Implements: Spec SPEC-02.22

        Args:
            node_id: Node ID to update
            **kwargs: Fields to update (content, confidence, tier, etc.)

        Returns:
            True if updated, False if node not found
        """
        if not kwargs:
            return False

        # Validate confidence if provided
        if "confidence" in kwargs:
            confidence = kwargs["confidence"]
            if not 0.0 <= confidence <= 1.0:
                raise ValueError(f"Confidence must be between 0.0 and 1.0, got: {confidence}")

        # Validate tier if provided
        if "tier" in kwargs:
            tier = kwargs["tier"]
            if tier not in self.VALID_TIERS:
                raise ValueError(f"Invalid tier: {tier}")

        conn = self._get_connection()
        try:
            # Check if node exists and get current tier for logging
            cursor = conn.execute(
                "SELECT tier FROM nodes WHERE id = ?", (node_id,)
            )
            row = cursor.fetchone()
            if row is None:
                return False

            old_tier = row["tier"]

            # Build update query
            updates = []
            values = []
            for key, value in kwargs.items():
                if key == "metadata":
                    updates.append(f"{key} = ?")
                    values.append(json.dumps(value))
                else:
                    updates.append(f"{key} = ?")
                    values.append(value)

            # Always update updated_at
            updates.append("updated_at = strftime('%s', 'now') * 1000")

            values.append(node_id)

            query = f"UPDATE nodes SET {', '.join(updates)} WHERE id = ?"
            conn.execute(query, values)

            # Log tier transition if tier changed (SPEC-02.19)
            if "tier" in kwargs and kwargs["tier"] != old_tier:
                self._log_evolution(
                    conn,
                    operation="tier_transition",
                    node_ids=[node_id],
                    from_tier=old_tier,
                    to_tier=kwargs["tier"],
                )

            conn.commit()
            return True
        finally:
            conn.close()

    def delete_node(self, node_id: str) -> bool:
        """
        Soft delete a node (move to archive tier).

        Implements: Spec SPEC-02.23

        Args:
            node_id: Node ID to delete

        Returns:
            True if deleted, False if node not found
        """
        return self.update_node(node_id, tier="archive")

    def hard_delete_node(self, node_id: str) -> bool:
        """
        Permanently delete a node.

        Args:
            node_id: Node ID to delete

        Returns:
            True if deleted, False if node not found
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute("DELETE FROM nodes WHERE id = ?", (node_id,))
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def query_nodes(
        self,
        node_type: str | None = None,
        tier: str | None = None,
        min_confidence: float | None = None,
        limit: int | None = None,
        include_archived: bool = False,
    ) -> list[Node]:
        """
        Query nodes with filters.

        Implements: Spec SPEC-02.24

        Args:
            node_type: Filter by node type
            tier: Filter by tier
            min_confidence: Minimum confidence threshold
            limit: Maximum number of results
            include_archived: Include archived nodes

        Returns:
            List of matching nodes
        """
        conditions = []
        values: list[Any] = []

        if not include_archived:
            conditions.append("tier != 'archive'")

        if node_type is not None:
            conditions.append("type = ?")
            values.append(node_type)

        if tier is not None:
            conditions.append("tier = ?")
            values.append(tier)

        if min_confidence is not None:
            conditions.append("confidence >= ?")
            values.append(min_confidence)

        query = "SELECT * FROM nodes"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY last_accessed DESC"

        if limit is not None:
            query += " LIMIT ?"
            values.append(limit)

        conn = self._get_connection()
        try:
            cursor = conn.execute(query, values)
            return [self._row_to_node(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    # =========================================================================
    # Hyperedge Operations (SPEC-02.25-26)
    # =========================================================================

    def create_edge(
        self,
        edge_type: str,
        label: str | None,
        members: list[dict[str, Any]],
        weight: float = 1.0,
    ) -> str:
        """
        Create a hyperedge connecting nodes.

        Implements: Spec SPEC-02.25

        Args:
            edge_type: Type of edge (relation, composition, causation, context)
            label: Edge label
            members: List of dicts with node_id, role, and optional position
            weight: Edge weight (>= 0.0)

        Returns:
            Edge ID (UUID)
        """
        # Validate edge type
        if edge_type not in self.VALID_EDGE_TYPES:
            raise ValueError(
                f"Invalid edge type: {edge_type}. Must be one of: {self.VALID_EDGE_TYPES}"
            )

        # Validate weight
        if weight < 0.0:
            raise ValueError(f"Edge weight must be >= 0.0, got: {weight}")

        edge_id = str(uuid.uuid4())

        conn = self._get_connection()
        try:
            # Create edge
            conn.execute(
                "INSERT INTO hyperedges (id, type, label, weight) VALUES (?, ?, ?, ?)",
                (edge_id, edge_type, label, weight),
            )

            # Create memberships
            for i, member in enumerate(members):
                node_id = member["node_id"]
                role = member["role"]
                position = member.get("position", i)

                conn.execute(
                    """
                    INSERT INTO membership (hyperedge_id, node_id, role, position)
                    VALUES (?, ?, ?, ?)
                    """,
                    (edge_id, node_id, role, position),
                )

            conn.commit()
        finally:
            conn.close()

        return edge_id

    def get_edge(self, edge_id: str) -> Hyperedge | None:
        """Get a hyperedge by ID."""
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "SELECT * FROM hyperedges WHERE id = ?", (edge_id,)
            )
            row = cursor.fetchone()
            if row is None:
                return None

            return Hyperedge(
                id=row["id"],
                type=row["type"],
                label=row["label"],
                weight=row["weight"],
            )
        finally:
            conn.close()

    def get_edge_members(self, edge_id: str) -> list[dict[str, Any]]:
        """Get members of a hyperedge."""
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """
                SELECT node_id, role, position
                FROM membership
                WHERE hyperedge_id = ?
                ORDER BY position
                """,
                (edge_id,),
            )
            return [
                {"node_id": row["node_id"], "role": row["role"], "position": row["position"]}
                for row in cursor.fetchall()
            ]
        finally:
            conn.close()

    def update_edge(self, edge_id: str, **kwargs: Any) -> bool:
        """Update a hyperedge."""
        if not kwargs:
            return False

        if "weight" in kwargs and kwargs["weight"] < 0.0:
            raise ValueError("Edge weight must be >= 0.0")

        conn = self._get_connection()
        try:
            updates = []
            values = []
            for key, value in kwargs.items():
                updates.append(f"{key} = ?")
                values.append(value)

            values.append(edge_id)

            query = f"UPDATE hyperedges SET {', '.join(updates)} WHERE id = ?"
            cursor = conn.execute(query, values)
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def delete_edge(self, edge_id: str) -> bool:
        """Delete a hyperedge."""
        conn = self._get_connection()
        try:
            cursor = conn.execute("DELETE FROM hyperedges WHERE id = ?", (edge_id,))
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def get_related_nodes(
        self,
        node_id: str,
        edge_type: str | None = None,
    ) -> list[Node]:
        """
        Get nodes related to a given node via hyperedges.

        Implements: Spec SPEC-02.26

        Args:
            node_id: Source node ID
            edge_type: Optional filter by edge type

        Returns:
            List of related nodes
        """
        conn = self._get_connection()
        try:
            # Find all edges this node participates in
            query = """
                SELECT DISTINCT n.* FROM nodes n
                JOIN membership m1 ON n.id = m1.node_id
                JOIN membership m2 ON m1.hyperedge_id = m2.hyperedge_id
                WHERE m2.node_id = ? AND n.id != ? AND n.tier != 'archive'
            """
            values: list[Any] = [node_id, node_id]

            if edge_type is not None:
                query = """
                    SELECT DISTINCT n.* FROM nodes n
                    JOIN membership m1 ON n.id = m1.node_id
                    JOIN membership m2 ON m1.hyperedge_id = m2.hyperedge_id
                    JOIN hyperedges h ON m1.hyperedge_id = h.id
                    WHERE m2.node_id = ? AND n.id != ? AND n.tier != 'archive'
                    AND h.type = ?
                """
                values.append(edge_type)

            cursor = conn.execute(query, values)
            return [self._row_to_node(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def query_edges(
        self,
        edge_type: str | None = None,
        label: str | None = None,
    ) -> list[Hyperedge]:
        """Query hyperedges with filters."""
        conditions = []
        values: list[Any] = []

        if edge_type is not None:
            conditions.append("type = ?")
            values.append(edge_type)

        if label is not None:
            conditions.append("label = ?")
            values.append(label)

        query = "SELECT * FROM hyperedges"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        conn = self._get_connection()
        try:
            cursor = conn.execute(query, values)
            return [
                Hyperedge(
                    id=row["id"],
                    type=row["type"],
                    label=row["label"],
                    weight=row["weight"],
                )
                for row in cursor.fetchall()
            ]
        finally:
            conn.close()

    def get_edges_for_node(self, node_id: str) -> list[Hyperedge]:
        """Get all edges a node participates in."""
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """
                SELECT DISTINCT h.* FROM hyperedges h
                JOIN membership m ON h.id = m.hyperedge_id
                WHERE m.node_id = ?
                """,
                (node_id,),
            )
            return [
                Hyperedge(
                    id=row["id"],
                    type=row["type"],
                    label=row["label"],
                    weight=row["weight"],
                )
                for row in cursor.fetchall()
            ]
        finally:
            conn.close()

    # =========================================================================
    # Evolution Log (SPEC-02.19)
    # =========================================================================

    def _log_evolution(
        self,
        conn: sqlite3.Connection,
        operation: str,
        node_ids: list[str],
        from_tier: str | None = None,
        to_tier: str | None = None,
        reasoning: str | None = None,
    ) -> None:
        """Log an evolution event."""
        conn.execute(
            """
            INSERT INTO evolution_log (operation, node_ids, from_tier, to_tier, reasoning)
            VALUES (?, ?, ?, ?, ?)
            """,
            (operation, json.dumps(node_ids), from_tier, to_tier, reasoning),
        )

    def get_evolution_log(
        self,
        node_id: str | None = None,
        operation_type: str | None = None,
    ) -> list[EvolutionLogEntry]:
        """
        Get evolution log entries.

        Args:
            node_id: Optional filter by node ID
            operation_type: Optional filter by operation type

        Returns:
            List of EvolutionLogEntry objects
        """
        conn = self._get_connection()
        try:
            conditions = []
            values: list[Any] = []

            if node_id is not None:
                conditions.append("node_ids LIKE ?")
                values.append(f'%"{node_id}"%')

            if operation_type is not None:
                conditions.append("operation = ?")
                values.append(operation_type)

            query = "SELECT * FROM evolution_log"
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            query += " ORDER BY timestamp DESC"

            cursor = conn.execute(query, values)

            return [
                EvolutionLogEntry(
                    id=row["id"],
                    timestamp=row["timestamp"],
                    operation=row["operation"],
                    node_ids=json.loads(row["node_ids"]),
                    from_tier=row["from_tier"],
                    to_tier=row["to_tier"],
                    reasoning=row["reasoning"],
                )
                for row in cursor.fetchall()
            ]
        finally:
            conn.close()

    def log_evolution(
        self,
        operation: str,
        node_ids: list[str],
        from_tier: str | None = None,
        to_tier: str | None = None,
        reasoning: str | None = None,
    ) -> None:
        """
        Public method to log an evolution event.

        Implements: Spec SPEC-03.06, SPEC-03.13, SPEC-03.20

        Args:
            operation: Type of operation (consolidate, promote, decay, etc.)
            node_ids: List of affected node IDs
            from_tier: Original tier (if applicable)
            to_tier: New tier (if applicable)
            reasoning: Explanation of why this operation occurred
        """
        conn = self._get_connection()
        try:
            self._log_evolution(conn, operation, node_ids, from_tier, to_tier, reasoning)
            conn.commit()
        finally:
            conn.close()

    def _set_last_accessed(self, node_id: str, timestamp: Any) -> bool:
        """
        Set the last_accessed time for a node (for testing).

        Args:
            node_id: Node ID to update
            timestamp: datetime object

        Returns:
            True if updated, False if node not found
        """
        # Convert datetime to milliseconds timestamp
        from datetime import datetime

        if isinstance(timestamp, datetime):
            ts_ms = int(timestamp.timestamp() * 1000)
        else:
            ts_ms = timestamp

        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "UPDATE nodes SET last_accessed = ? WHERE id = ?",
                (ts_ms, node_id),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def get_nodes_by_metadata(
        self,
        key: str,
        value: Any,
        tier: str | None = None,
        include_archived: bool = False,
    ) -> list[Node]:
        """
        Get nodes by metadata key-value.

        Args:
            key: Metadata key to search
            value: Metadata value to match
            tier: Optional tier filter
            include_archived: Include archived nodes

        Returns:
            List of matching nodes
        """
        conn = self._get_connection()
        try:
            # Use JSON extract for metadata search
            # json_extract returns the raw value, so compare directly
            conditions = [f"json_extract(metadata, '$.{key}') = ?"]
            values: list[Any] = [value]

            if not include_archived:
                conditions.append("tier != 'archive'")

            if tier is not None:
                conditions.append("tier = ?")
                values.append(tier)

            query = "SELECT * FROM nodes WHERE " + " AND ".join(conditions)
            cursor = conn.execute(query, values)
            return [self._row_to_node(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    # =========================================================================
    # Helpers
    # =========================================================================

    def _row_to_node(self, row: sqlite3.Row) -> Node:
        """Convert a database row to a Node object."""
        metadata = row["metadata"]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)

        return Node(
            id=row["id"],
            type=row["type"],
            content=row["content"],
            tier=row["tier"],
            confidence=row["confidence"],
            subtype=row["subtype"],
            embedding=row["embedding"],
            provenance=row["provenance"],
            metadata=metadata or {},
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            last_accessed=row["last_accessed"],
            access_count=row["access_count"],
        )


__all__ = ["MemoryStore", "Node", "Hyperedge", "EvolutionLogEntry"]
