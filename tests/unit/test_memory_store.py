"""
Unit tests for memory storage layer.

Implements: Spec SPEC-02 tests for schema, nodes, and CRUD operations.
"""

import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_db_path():
    """Create a temporary database path."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    yield path
    # Cleanup
    if os.path.exists(path):
        os.unlink(path)
    # Also cleanup WAL and SHM files
    for suffix in ["-wal", "-shm"]:
        wal_path = path + suffix
        if os.path.exists(wal_path):
            os.unlink(wal_path)


@pytest.fixture
def memory_store(temp_db_path):
    """Create a MemoryStore instance with temporary database."""
    from src.memory_store import MemoryStore

    store = MemoryStore(db_path=temp_db_path)
    return store


# =============================================================================
# SPEC-02.01-04: Storage Layer Tests
# =============================================================================


class TestStorageLayer:
    """Tests for SQLite storage configuration."""

    def test_uses_sqlite_database(self, temp_db_path):
        """
        Memory store should use SQLite for storage.

        @trace SPEC-02.01
        """
        from src.memory_store import MemoryStore

        _store = MemoryStore(db_path=temp_db_path)  # noqa: F841 - creates DB

        # Verify file is created
        assert os.path.exists(temp_db_path)

        # Verify it's a valid SQLite database
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.execute("SELECT sqlite_version()")
        version = cursor.fetchone()[0]
        conn.close()

        assert version is not None

    def test_default_database_location(self, monkeypatch):
        """
        Default database location should be ~/.claude/rlm-memory.db.

        @trace SPEC-02.02
        """
        from src.memory_store import MemoryStore

        # Clear env var if set
        monkeypatch.delenv("RLM_MEMORY_DB", raising=False)

        store = MemoryStore.__new__(MemoryStore)
        default_path = store._get_default_db_path()

        expected = Path.home() / ".claude" / "rlm-memory.db"
        assert Path(default_path) == expected

    def test_database_location_configurable_via_env(self, temp_db_path, monkeypatch):
        """
        Database location should be configurable via RLM_MEMORY_DB env var.

        @trace SPEC-02.03
        """
        from src.memory_store import MemoryStore

        custom_path = temp_db_path + ".custom"
        monkeypatch.setenv("RLM_MEMORY_DB", custom_path)

        store = MemoryStore()

        # Should use the custom path
        assert store.db_path == custom_path

        # Cleanup
        if os.path.exists(custom_path):
            os.unlink(custom_path)

    def test_uses_wal_mode(self, memory_store, temp_db_path):  # noqa: ARG002
        """
        Database should use WAL mode for performance.

        @trace SPEC-02.04
        """
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.execute("PRAGMA journal_mode")
        mode = cursor.fetchone()[0]
        conn.close()

        assert mode.lower() == "wal"


# =============================================================================
# SPEC-02.05-10: Node Types and Constraints
# =============================================================================


class TestNodeTypes:
    """Tests for node type support and constraints."""

    @pytest.mark.parametrize(
        "node_type",
        ["entity", "fact", "experience", "decision", "snippet"],
    )
    def test_supports_all_node_types(self, memory_store, node_type):
        """
        System should support all specified node types.

        @trace SPEC-02.05
        """
        node_id = memory_store.create_node(
            node_type=node_type,
            content=f"Test {node_type} content",
        )

        assert node_id is not None

        node = memory_store.get_node(node_id)
        assert node is not None
        assert node.type == node_type

    def test_invalid_node_type_rejected(self, memory_store):
        """
        Invalid node types should be rejected.

        @trace SPEC-02.05
        """
        with pytest.raises((ValueError, sqlite3.IntegrityError)):
            memory_store.create_node(
                node_type="invalid_type",
                content="Test content",
            )

    def test_node_has_required_fields(self, memory_store):
        """
        Each node should have id, type, content, tier, confidence.

        @trace SPEC-02.06
        """
        node_id = memory_store.create_node(
            node_type="fact",
            content="Test fact content",
        )

        node = memory_store.get_node(node_id)

        # Required fields
        assert hasattr(node, "id") and node.id is not None
        assert hasattr(node, "type") and node.type == "fact"
        assert hasattr(node, "content") and node.content == "Test fact content"
        assert hasattr(node, "tier") and node.tier is not None
        assert hasattr(node, "confidence") and node.confidence is not None

    def test_node_tracks_timestamps(self, memory_store):
        """
        Each node should track created_at, updated_at, last_accessed, access_count.

        @trace SPEC-02.07
        """
        node_id = memory_store.create_node(
            node_type="fact",
            content="Test content",
        )

        node = memory_store.get_node(node_id)

        # Timestamp fields
        assert hasattr(node, "created_at") and node.created_at is not None
        assert hasattr(node, "updated_at") and node.updated_at is not None
        assert hasattr(node, "last_accessed") and node.last_accessed is not None
        assert hasattr(node, "access_count") and isinstance(node.access_count, int)

    def test_node_optional_fields(self, memory_store):
        """
        Nodes may have optional fields: subtype, embedding, provenance, metadata.

        @trace SPEC-02.08
        """
        node_id = memory_store.create_node(
            node_type="entity",
            content="Test entity",
            subtype="person",
            provenance="user_input",
            metadata={"source": "conversation"},
        )

        node = memory_store.get_node(node_id)

        assert node.subtype == "person"
        assert node.provenance == "user_input"
        assert node.metadata == {"source": "conversation"}

    def test_node_id_is_uuid(self, memory_store):
        """
        Node IDs should be UUIDs.

        @trace SPEC-02.09
        """
        import uuid

        node_id = memory_store.create_node(
            node_type="fact",
            content="Test content",
        )

        # Should be valid UUID
        parsed = uuid.UUID(node_id)
        assert str(parsed) == node_id

    def test_confidence_constrained_to_valid_range(self, memory_store):
        """
        Node confidence should be constrained to [0.0, 1.0].

        @trace SPEC-02.10
        """
        # Valid confidence values
        node_id = memory_store.create_node(
            node_type="fact",
            content="Test",
            confidence=0.0,
        )
        node = memory_store.get_node(node_id)
        assert node.confidence == 0.0

        node_id = memory_store.create_node(
            node_type="fact",
            content="Test",
            confidence=1.0,
        )
        node = memory_store.get_node(node_id)
        assert node.confidence == 1.0

        node_id = memory_store.create_node(
            node_type="fact",
            content="Test",
            confidence=0.5,
        )
        node = memory_store.get_node(node_id)
        assert node.confidence == 0.5

    def test_confidence_below_zero_rejected(self, memory_store):
        """
        Confidence below 0.0 should be rejected.

        @trace SPEC-02.10
        """
        with pytest.raises((ValueError, sqlite3.IntegrityError)):
            memory_store.create_node(
                node_type="fact",
                content="Test",
                confidence=-0.1,
            )

    def test_confidence_above_one_rejected(self, memory_store):
        """
        Confidence above 1.0 should be rejected.

        @trace SPEC-02.10
        """
        with pytest.raises((ValueError, sqlite3.IntegrityError)):
            memory_store.create_node(
                node_type="fact",
                content="Test",
                confidence=1.1,
            )


# =============================================================================
# SPEC-02.17-19: Tier System
# =============================================================================


class TestTierSystem:
    """Tests for memory tier system."""

    @pytest.mark.parametrize(
        "tier",
        ["task", "session", "longterm", "archive"],
    )
    def test_supports_all_tiers(self, memory_store, tier):
        """
        System should support all specified tiers.

        @trace SPEC-02.17
        """
        node_id = memory_store.create_node(
            node_type="fact",
            content="Test content",
            tier=tier,
        )

        # Archive nodes require include_archived=True to retrieve
        include_archived = tier == "archive"
        node = memory_store.get_node(node_id, include_archived=include_archived)
        assert node.tier == tier

    def test_default_tier_is_task(self, memory_store):
        """
        New nodes should default to 'task' tier.

        @trace SPEC-02.18
        """
        node_id = memory_store.create_node(
            node_type="fact",
            content="Test content",
            # No tier specified
        )

        node = memory_store.get_node(node_id)
        assert node.tier == "task"

    def test_invalid_tier_rejected(self, memory_store):
        """
        Invalid tier values should be rejected.

        @trace SPEC-02.17
        """
        with pytest.raises((ValueError, sqlite3.IntegrityError)):
            memory_store.create_node(
                node_type="fact",
                content="Test",
                tier="invalid_tier",
            )

    def test_tier_transition_logged(self, memory_store):
        """
        Tier transitions should be logged in evolution_log.

        @trace SPEC-02.19
        """
        node_id = memory_store.create_node(
            node_type="fact",
            content="Test content",
            tier="task",
        )

        # Transition tier
        memory_store.update_node(node_id, tier="session")

        # Check evolution log
        logs = memory_store.get_evolution_log(node_id)
        assert len(logs) >= 1

        # Find the tier transition log
        tier_logs = [log for log in logs if log.from_tier == "task"]
        assert len(tier_logs) >= 1
        assert tier_logs[0].to_tier == "session"


# =============================================================================
# SPEC-02.20-26: CRUD Operations
# =============================================================================


class TestCreateNode:
    """Tests for create_node operation."""

    def test_create_node_returns_id(self, memory_store):
        """
        create_node should return the node ID.

        @trace SPEC-02.20
        """
        node_id = memory_store.create_node(
            node_type="fact",
            content="Test content",
        )

        assert node_id is not None
        assert isinstance(node_id, str)
        assert len(node_id) == 36  # UUID length

    def test_create_node_with_all_kwargs(self, memory_store):
        """
        create_node should accept all optional kwargs.

        @trace SPEC-02.20
        """
        node_id = memory_store.create_node(
            node_type="experience",
            content="Test experience",
            subtype="debugging",
            tier="session",
            confidence=0.9,
            provenance="tool_output",
            metadata={"tool": "bash", "exit_code": 0},
        )

        node = memory_store.get_node(node_id)
        assert node.type == "experience"
        assert node.subtype == "debugging"
        assert node.tier == "session"
        assert node.confidence == 0.9
        assert node.provenance == "tool_output"
        assert node.metadata["tool"] == "bash"


class TestGetNode:
    """Tests for get_node operation."""

    def test_get_node_returns_node(self, memory_store):
        """
        get_node should return the node object.

        @trace SPEC-02.21
        """
        node_id = memory_store.create_node(
            node_type="fact",
            content="Test content",
        )

        node = memory_store.get_node(node_id)

        assert node is not None
        assert node.id == node_id
        assert node.content == "Test content"

    def test_get_node_nonexistent_returns_none(self, memory_store):
        """
        get_node should return None for nonexistent IDs.

        @trace SPEC-02.21
        """
        import uuid

        fake_id = str(uuid.uuid4())
        node = memory_store.get_node(fake_id)

        assert node is None

    def test_get_node_updates_access_tracking(self, memory_store):
        """
        get_node should update last_accessed and access_count.

        @trace SPEC-02.21, SPEC-02.07
        """
        node_id = memory_store.create_node(
            node_type="fact",
            content="Test content",
        )

        # Get node multiple times
        node1 = memory_store.get_node(node_id)
        initial_access_count = node1.access_count

        memory_store.get_node(node_id)
        memory_store.get_node(node_id)

        node2 = memory_store.get_node(node_id)

        assert node2.access_count > initial_access_count


class TestUpdateNode:
    """Tests for update_node operation."""

    def test_update_node_returns_success(self, memory_store):
        """
        update_node should return True on success.

        @trace SPEC-02.22
        """
        node_id = memory_store.create_node(
            node_type="fact",
            content="Original content",
        )

        result = memory_store.update_node(node_id, content="Updated content")

        assert result is True

    def test_update_node_modifies_content(self, memory_store):
        """
        update_node should modify node content.

        @trace SPEC-02.22
        """
        node_id = memory_store.create_node(
            node_type="fact",
            content="Original content",
        )

        memory_store.update_node(node_id, content="Updated content")

        node = memory_store.get_node(node_id)
        assert node.content == "Updated content"

    def test_update_node_modifies_confidence(self, memory_store):
        """
        update_node should modify confidence.

        @trace SPEC-02.22
        """
        node_id = memory_store.create_node(
            node_type="fact",
            content="Test",
            confidence=0.5,
        )

        memory_store.update_node(node_id, confidence=0.9)

        node = memory_store.get_node(node_id)
        assert node.confidence == 0.9

    def test_update_node_nonexistent_returns_false(self, memory_store):
        """
        update_node should return False for nonexistent IDs.

        @trace SPEC-02.22
        """
        import uuid

        fake_id = str(uuid.uuid4())
        result = memory_store.update_node(fake_id, content="Test")

        assert result is False

    def test_update_node_updates_timestamp(self, memory_store):
        """
        update_node should update the updated_at timestamp.

        @trace SPEC-02.22, SPEC-02.07
        """
        import time

        node_id = memory_store.create_node(
            node_type="fact",
            content="Original",
        )

        node1 = memory_store.get_node(node_id)
        original_updated = node1.updated_at

        time.sleep(0.01)  # Small delay

        memory_store.update_node(node_id, content="Updated")

        node2 = memory_store.get_node(node_id)
        assert node2.updated_at >= original_updated


class TestDeleteNode:
    """Tests for delete_node operation."""

    def test_delete_node_returns_success(self, memory_store):
        """
        delete_node should return True on success.

        @trace SPEC-02.23
        """
        node_id = memory_store.create_node(
            node_type="fact",
            content="Test content",
        )

        result = memory_store.delete_node(node_id)

        assert result is True

    def test_delete_node_soft_deletes_to_archive(self, memory_store):
        """
        delete_node should soft delete to archive tier.

        @trace SPEC-02.23
        """
        node_id = memory_store.create_node(
            node_type="fact",
            content="Test content",
            tier="task",
        )

        memory_store.delete_node(node_id)

        # Node should still exist but be in archive tier
        node = memory_store.get_node(node_id, include_archived=True)
        assert node is not None
        assert node.tier == "archive"

    def test_delete_node_not_returned_by_default(self, memory_store):
        """
        Deleted (archived) nodes should not be returned by default.

        @trace SPEC-02.23
        """
        node_id = memory_store.create_node(
            node_type="fact",
            content="Test content",
        )

        memory_store.delete_node(node_id)

        # Without include_archived, should return None
        node = memory_store.get_node(node_id)
        assert node is None

    def test_delete_node_nonexistent_returns_false(self, memory_store):
        """
        delete_node should return False for nonexistent IDs.

        @trace SPEC-02.23
        """
        import uuid

        fake_id = str(uuid.uuid4())
        result = memory_store.delete_node(fake_id)

        assert result is False


class TestQueryNodes:
    """Tests for query_nodes operation."""

    def test_query_nodes_by_type(self, memory_store):
        """
        query_nodes should filter by type.

        @trace SPEC-02.24
        """
        # Create nodes of different types
        memory_store.create_node(node_type="fact", content="Fact 1")
        memory_store.create_node(node_type="fact", content="Fact 2")
        memory_store.create_node(node_type="entity", content="Entity 1")

        results = memory_store.query_nodes(node_type="fact")

        assert len(results) == 2
        assert all(n.type == "fact" for n in results)

    def test_query_nodes_by_tier(self, memory_store):
        """
        query_nodes should filter by tier.

        @trace SPEC-02.24
        """
        memory_store.create_node(node_type="fact", content="Task fact", tier="task")
        memory_store.create_node(
            node_type="fact", content="Session fact", tier="session"
        )

        results = memory_store.query_nodes(tier="session")

        assert len(results) == 1
        assert results[0].tier == "session"

    def test_query_nodes_by_min_confidence(self, memory_store):
        """
        query_nodes should filter by minimum confidence.

        @trace SPEC-02.24
        """
        memory_store.create_node(
            node_type="fact", content="Low confidence", confidence=0.3
        )
        memory_store.create_node(
            node_type="fact", content="High confidence", confidence=0.9
        )

        results = memory_store.query_nodes(min_confidence=0.5)

        assert len(results) == 1
        assert results[0].confidence >= 0.5

    def test_query_nodes_with_limit(self, memory_store):
        """
        query_nodes should respect limit parameter.

        @trace SPEC-02.24
        """
        for i in range(10):
            memory_store.create_node(node_type="fact", content=f"Fact {i}")

        results = memory_store.query_nodes(limit=5)

        assert len(results) == 5

    def test_query_nodes_combined_filters(self, memory_store):
        """
        query_nodes should support combined filters.

        @trace SPEC-02.24
        """
        memory_store.create_node(
            node_type="fact", content="Target", tier="session", confidence=0.9
        )
        memory_store.create_node(
            node_type="fact", content="Wrong tier", tier="task", confidence=0.9
        )
        memory_store.create_node(
            node_type="entity", content="Wrong type", tier="session", confidence=0.9
        )

        results = memory_store.query_nodes(
            node_type="fact", tier="session", min_confidence=0.8
        )

        assert len(results) == 1
        assert results[0].content == "Target"

    def test_query_nodes_excludes_archived_by_default(self, memory_store):
        """
        query_nodes should exclude archived nodes by default.

        @trace SPEC-02.24
        """
        node_id = memory_store.create_node(node_type="fact", content="To be archived")
        memory_store.delete_node(node_id)  # Soft delete to archive

        results = memory_store.query_nodes(node_type="fact")

        assert len(results) == 0


# =============================================================================
# Schema Tests
# =============================================================================


class TestSchemaCreation:
    """Tests for database schema creation."""

    def test_nodes_table_exists(self, memory_store, temp_db_path):  # noqa: ARG002
        """
        Nodes table should be created.

        @trace SPEC-02.06
        """
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='nodes'"
        )
        result = cursor.fetchone()
        conn.close()

        assert result is not None

    def test_hyperedges_table_exists(self, memory_store, temp_db_path):  # noqa: ARG002
        """
        Hyperedges table should be created.

        @trace SPEC-02.12
        """
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='hyperedges'"
        )
        result = cursor.fetchone()
        conn.close()

        assert result is not None

    def test_membership_table_exists(self, memory_store, temp_db_path):  # noqa: ARG002
        """
        Membership table should be created.

        @trace SPEC-02.14
        """
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='membership'"
        )
        result = cursor.fetchone()
        conn.close()

        assert result is not None

    def test_evolution_log_table_exists(self, memory_store, temp_db_path):  # noqa: ARG002
        """
        Evolution log table should be created.

        @trace SPEC-02.19
        """
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='evolution_log'"
        )
        result = cursor.fetchone()
        conn.close()

        assert result is not None

    def test_indexes_created(self, memory_store, temp_db_path):  # noqa: ARG002
        """
        Required indexes should be created.

        @trace SPEC-02
        """
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'"
        )
        indexes = [row[0] for row in cursor.fetchall()]
        conn.close()

        expected_indexes = [
            "idx_nodes_tier",
            "idx_nodes_type",
            "idx_nodes_confidence",
            "idx_nodes_last_accessed",
            "idx_membership_node",
            "idx_membership_edge",
        ]

        for idx in expected_indexes:
            assert idx in indexes, f"Missing index: {idx}"


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_content_allowed(self, memory_store):
        """
        Empty content should be allowed.

        @trace SPEC-02.06
        """
        node_id = memory_store.create_node(
            node_type="fact",
            content="",
        )

        node = memory_store.get_node(node_id)
        assert node.content == ""

    def test_large_content_handled(self, memory_store):
        """
        Large content should be handled.

        @trace SPEC-02.06
        """
        large_content = "x" * 100000

        node_id = memory_store.create_node(
            node_type="snippet",
            content=large_content,
        )

        node = memory_store.get_node(node_id)
        assert len(node.content) == 100000

    def test_unicode_content_handled(self, memory_store):
        """
        Unicode content should be handled correctly.

        @trace SPEC-02.06
        """
        unicode_content = "Hello 世界 🌍 مرحبا"

        node_id = memory_store.create_node(
            node_type="fact",
            content=unicode_content,
        )

        node = memory_store.get_node(node_id)
        assert node.content == unicode_content

    def test_special_characters_in_content(self, memory_store):
        """
        Special characters should not cause issues.

        @trace SPEC-02.34
        """
        special_content = "Test'; DROP TABLE nodes; --"

        node_id = memory_store.create_node(
            node_type="fact",
            content=special_content,
        )

        node = memory_store.get_node(node_id)
        assert node.content == special_content

        # Verify table still exists
        nodes = memory_store.query_nodes()
        assert len(nodes) >= 1

    def test_json_metadata_roundtrip(self, memory_store):
        """
        JSON metadata should roundtrip correctly.

        @trace SPEC-02.08
        """
        metadata = {
            "string": "value",
            "number": 42,
            "float": 3.14,
            "bool": True,
            "null": None,
            "array": [1, 2, 3],
            "nested": {"key": "value"},
        }

        node_id = memory_store.create_node(
            node_type="fact",
            content="Test",
            metadata=metadata,
        )

        node = memory_store.get_node(node_id)
        assert node.metadata == metadata

    def test_concurrent_access(self, temp_db_path):
        """
        Multiple store instances should handle concurrent access.

        @trace SPEC-02.04
        """
        from src.memory_store import MemoryStore

        store1 = MemoryStore(db_path=temp_db_path)
        store2 = MemoryStore(db_path=temp_db_path)

        # Create from store1
        node_id = store1.create_node(node_type="fact", content="From store 1")

        # Read from store2
        node = store2.get_node(node_id)
        assert node is not None
        assert node.content == "From store 1"

        # Update from store2
        store2.update_node(node_id, content="Updated by store 2")

        # Read updated from store1
        node = store1.get_node(node_id)
        assert node.content == "Updated by store 2"


# =============================================================================
# Default Confidence Tests
# =============================================================================


class TestDefaultConfidence:
    """Tests for default confidence behavior."""

    def test_default_confidence_is_half(self, memory_store):
        """
        Default confidence should be 0.5.

        @trace SPEC-02.06
        """
        node_id = memory_store.create_node(
            node_type="fact",
            content="Test",
            # No confidence specified
        )

        node = memory_store.get_node(node_id)
        assert node.confidence == 0.5
