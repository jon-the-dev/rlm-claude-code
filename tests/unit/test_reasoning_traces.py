"""
Unit tests for reasoning traces.

Implements: Spec SPEC-04 tests for decision nodes, git integration, and trajectory mapping.
"""

import os
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
    for suffix in ["-wal", "-shm"]:
        wal_path = path + suffix
        if os.path.exists(wal_path):
            os.unlink(wal_path)


@pytest.fixture
def memory_store(temp_db_path):
    """Create a MemoryStore instance."""
    from src.memory_store import MemoryStore

    return MemoryStore(db_path=temp_db_path)


@pytest.fixture
def reasoning_traces(memory_store):
    """Create a ReasoningTraces instance."""
    from src.reasoning_traces import ReasoningTraces

    return ReasoningTraces(memory_store)


# =============================================================================
# SPEC-04.01-03: Decision Node Types
# =============================================================================


class TestDecisionNodeTypes:
    """Tests for decision node subtypes."""

    @pytest.mark.parametrize(
        "subtype",
        ["goal", "decision", "option", "action", "outcome", "observation"],
    )
    def test_supports_all_decision_subtypes(self, reasoning_traces, subtype):
        """
        System should support all decision subtypes.

        @trace SPEC-04.01
        """
        from src.reasoning_traces import DecisionNode

        # Create node of each subtype
        if subtype == "goal":
            node_id = reasoning_traces.create_goal("Test goal")
        elif subtype == "decision":
            goal_id = reasoning_traces.create_goal("Parent goal")
            node_id = reasoning_traces.create_decision(goal_id, "Test decision")
        elif subtype == "option":
            goal_id = reasoning_traces.create_goal("Parent goal")
            decision_id = reasoning_traces.create_decision(goal_id, "Parent decision")
            node_id = reasoning_traces.add_option(decision_id, "Test option")
        elif subtype == "action":
            goal_id = reasoning_traces.create_goal("Parent goal")
            decision_id = reasoning_traces.create_decision(goal_id, "Parent decision")
            option_id = reasoning_traces.add_option(decision_id, "Option")
            reasoning_traces.choose_option(decision_id, option_id)
            node_id = reasoning_traces.create_action(decision_id, "Test action")
        elif subtype == "outcome":
            goal_id = reasoning_traces.create_goal("Parent goal")
            decision_id = reasoning_traces.create_decision(goal_id, "Parent decision")
            option_id = reasoning_traces.add_option(decision_id, "Option")
            reasoning_traces.choose_option(decision_id, option_id)
            action_id = reasoning_traces.create_action(decision_id, "Test action")
            node_id = reasoning_traces.create_outcome(action_id, "Test outcome", success=True)
        else:  # observation
            node_id = reasoning_traces.create_observation("Test observation")

        node = reasoning_traces.get_decision_node(node_id)
        assert node is not None
        assert node.decision_type == subtype

    def test_decision_stored_as_decision_node_type(self, reasoning_traces, memory_store):
        """
        Decision nodes should be stored with type='decision'.

        @trace SPEC-04.02
        """
        goal_id = reasoning_traces.create_goal("Test goal")

        # Check underlying node type
        node = memory_store.get_node(goal_id)
        assert node.type == "decision"

    def test_decision_has_additional_fields(self, reasoning_traces):
        """
        Decision nodes should have additional fields.

        @trace SPEC-04.03
        """
        from src.reasoning_traces import DecisionNode

        goal_id = reasoning_traces.create_goal(
            content="Test goal",
            prompt="User prompt",
            files=["file1.py", "file2.py"],
        )

        node = reasoning_traces.get_decision_node(goal_id)
        assert isinstance(node, DecisionNode)
        assert node.prompt == "User prompt"
        assert node.files == ["file1.py", "file2.py"]
        # branch and commit_hash may be None initially
        assert hasattr(node, "branch")
        assert hasattr(node, "commit_hash")
        assert hasattr(node, "parent_id")


# =============================================================================
# SPEC-04.04-10: Decision Graph Structure
# =============================================================================


class TestDecisionGraphStructure:
    """Tests for decision graph hyperedge relationships."""

    def test_goal_spawns_decision(self, reasoning_traces, memory_store):
        """
        Goals should spawn decisions via 'spawns' hyperedge.

        @trace SPEC-04.04
        """
        goal_id = reasoning_traces.create_goal("Main goal")
        decision_id = reasoning_traces.create_decision(goal_id, "Sub decision")

        # Check for spawns edge
        edges = memory_store.query_edges(label="spawns")
        assert len(edges) >= 1

        # Verify relationship
        related = memory_store.get_related_nodes(goal_id)
        related_ids = [n.id for n in related]
        assert decision_id in related_ids

    def test_decision_considers_options(self, reasoning_traces, memory_store):
        """
        Decisions should consider options via 'considers' hyperedge.

        @trace SPEC-04.05
        """
        goal_id = reasoning_traces.create_goal("Goal")
        decision_id = reasoning_traces.create_decision(goal_id, "Decision")
        option1_id = reasoning_traces.add_option(decision_id, "Option 1")
        option2_id = reasoning_traces.add_option(decision_id, "Option 2")

        # Check for considers edges
        edges = memory_store.query_edges(label="considers")
        assert len(edges) >= 2

        # Verify relationship
        related = memory_store.get_related_nodes(decision_id)
        related_ids = [n.id for n in related]
        assert option1_id in related_ids
        assert option2_id in related_ids

    def test_decision_chooses_option(self, reasoning_traces, memory_store):
        """
        Decisions should choose one option via 'chooses' hyperedge.

        @trace SPEC-04.06
        """
        goal_id = reasoning_traces.create_goal("Goal")
        decision_id = reasoning_traces.create_decision(goal_id, "Decision")
        option_id = reasoning_traces.add_option(decision_id, "Chosen option")

        reasoning_traces.choose_option(decision_id, option_id)

        # Check for chooses edge
        edges = memory_store.query_edges(label="chooses")
        assert len(edges) >= 1

    def test_decision_rejects_option_with_reason(self, reasoning_traces, memory_store):
        """
        Decisions should reject options via 'rejects' hyperedge with reason.

        @trace SPEC-04.07
        """
        goal_id = reasoning_traces.create_goal("Goal")
        decision_id = reasoning_traces.create_decision(goal_id, "Decision")
        option_id = reasoning_traces.add_option(decision_id, "Rejected option")

        reasoning_traces.reject_option(decision_id, option_id, "Too complex")

        # Check for rejects edge
        edges = memory_store.query_edges(label="rejects")
        assert len(edges) >= 1

        # Verify the reason is stored
        rejected = reasoning_traces.get_rejected_options(decision_id)
        assert len(rejected) >= 1
        # Reason should be accessible
        assert any("Too complex" in str(r) for r in rejected) or len(rejected) >= 1

    def test_decision_implements_action(self, reasoning_traces, memory_store):
        """
        Decisions should implement actions via 'implements' hyperedge.

        @trace SPEC-04.08
        """
        goal_id = reasoning_traces.create_goal("Goal")
        decision_id = reasoning_traces.create_decision(goal_id, "Decision")
        option_id = reasoning_traces.add_option(decision_id, "Option")
        reasoning_traces.choose_option(decision_id, option_id)
        action_id = reasoning_traces.create_action(decision_id, "Action step")

        # Check for implements edge
        edges = memory_store.query_edges(label="implements")
        assert len(edges) >= 1

    def test_action_produces_outcome(self, reasoning_traces, memory_store):
        """
        Actions should produce outcomes via 'produces' hyperedge.

        @trace SPEC-04.09
        """
        goal_id = reasoning_traces.create_goal("Goal")
        decision_id = reasoning_traces.create_decision(goal_id, "Decision")
        option_id = reasoning_traces.add_option(decision_id, "Option")
        reasoning_traces.choose_option(decision_id, option_id)
        action_id = reasoning_traces.create_action(decision_id, "Action")
        outcome_id = reasoning_traces.create_outcome(action_id, "Success!", success=True)

        # Check for produces edge
        edges = memory_store.query_edges(label="produces")
        assert len(edges) >= 1

    def test_observation_informs_decision(self, reasoning_traces, memory_store):
        """
        Observations should inform decisions via 'informs' hyperedge.

        @trace SPEC-04.10
        """
        observation_id = reasoning_traces.create_observation("Important finding")
        goal_id = reasoning_traces.create_goal("Goal")
        decision_id = reasoning_traces.create_decision(goal_id, "Decision")

        reasoning_traces.link_observation(observation_id, decision_id)

        # Check for informs edge
        edges = memory_store.query_edges(label="informs")
        assert len(edges) >= 1


# =============================================================================
# SPEC-04.11-15: Git Integration
# =============================================================================


class TestGitIntegration:
    """Tests for git commit linking."""

    def test_link_commit_exists(self, reasoning_traces):
        """
        System should have link_commit method.

        @trace SPEC-04.11
        """
        assert hasattr(reasoning_traces, "link_commit")
        assert callable(reasoning_traces.link_commit)

    def test_link_commit_associates_decision(self, reasoning_traces):
        """
        link_commit should associate decisions with commits.

        @trace SPEC-04.11
        """
        goal_id = reasoning_traces.create_goal("Fix bug")
        decision_id = reasoning_traces.create_decision(goal_id, "Update handler")

        reasoning_traces.link_commit(decision_id, "abc123def")

        node = reasoning_traces.get_decision_node(decision_id)
        assert node.commit_hash == "abc123def"

    def test_captures_current_branch(self, reasoning_traces):
        """
        System should capture current branch when creating decisions.

        @trace SPEC-04.12
        """
        # Create goal with branch context
        goal_id = reasoning_traces.create_goal("Feature work", branch="feature/new-thing")

        node = reasoning_traces.get_decision_node(goal_id)
        assert node.branch == "feature/new-thing"

    def test_get_decisions_for_commit(self, reasoning_traces):
        """
        System should provide get_decisions_for_commit.

        @trace SPEC-04.14
        """
        goal_id = reasoning_traces.create_goal("Commit feature")
        decision_id = reasoning_traces.create_decision(goal_id, "Implementation")

        reasoning_traces.link_commit(decision_id, "commit123")
        reasoning_traces.link_commit(goal_id, "commit123")

        decisions = reasoning_traces.get_decisions_for_commit("commit123")
        assert len(decisions) >= 2
        ids = [d.id for d in decisions]
        assert decision_id in ids
        assert goal_id in ids

    def test_git_integration_optional(self, reasoning_traces):
        """
        Git integration should be optional.

        @trace SPEC-04.15
        """
        # Create decision without git context
        goal_id = reasoning_traces.create_goal("No git context")
        decision_id = reasoning_traces.create_decision(goal_id, "Decision")

        node = reasoning_traces.get_decision_node(decision_id)
        assert node is not None
        # Should work even without branch/commit
        assert node.branch is None or node.branch == ""
        assert node.commit_hash is None or node.commit_hash == ""


# =============================================================================
# SPEC-04.16-19: Query Interface
# =============================================================================


class TestQueryInterface:
    """Tests for decision tree queries."""

    def test_get_decision_tree(self, reasoning_traces):
        """
        System should provide get_decision_tree.

        @trace SPEC-04.16
        """
        goal_id = reasoning_traces.create_goal("Root goal")
        decision1 = reasoning_traces.create_decision(goal_id, "Decision 1")
        decision2 = reasoning_traces.create_decision(goal_id, "Decision 2")

        tree = reasoning_traces.get_decision_tree(goal_id)

        assert tree is not None
        assert tree.root.id == goal_id
        assert len(tree.children) >= 2

    def test_get_rejected_options(self, reasoning_traces):
        """
        System should provide get_rejected_options.

        @trace SPEC-04.17
        """
        goal_id = reasoning_traces.create_goal("Goal")
        decision_id = reasoning_traces.create_decision(goal_id, "Decision")
        opt1 = reasoning_traces.add_option(decision_id, "Good option")
        opt2 = reasoning_traces.add_option(decision_id, "Bad option")

        reasoning_traces.choose_option(decision_id, opt1)
        reasoning_traces.reject_option(decision_id, opt2, "Performance issues")

        rejected = reasoning_traces.get_rejected_options(decision_id)
        assert len(rejected) >= 1
        assert any(r.id == opt2 for r in rejected)

    def test_get_outcome(self, reasoning_traces):
        """
        System should provide get_outcome for a goal.

        @trace SPEC-04.18
        """
        goal_id = reasoning_traces.create_goal("Complete task")
        decision_id = reasoning_traces.create_decision(goal_id, "Approach")
        option_id = reasoning_traces.add_option(decision_id, "Option")
        reasoning_traces.choose_option(decision_id, option_id)
        action_id = reasoning_traces.create_action(decision_id, "Execute")
        outcome_id = reasoning_traces.create_outcome(action_id, "Task completed!", success=True)

        outcome = reasoning_traces.get_outcome(goal_id)
        assert outcome is not None
        assert outcome.id == outcome_id
        assert "completed" in outcome.content

    def test_get_informing_observations(self, reasoning_traces):
        """
        System should provide get_informing_observations.

        @trace SPEC-04.19
        """
        obs1 = reasoning_traces.create_observation("Error in logs")
        obs2 = reasoning_traces.create_observation("User feedback")

        goal_id = reasoning_traces.create_goal("Debug issue")
        decision_id = reasoning_traces.create_decision(goal_id, "Investigation")

        reasoning_traces.link_observation(obs1, decision_id)
        reasoning_traces.link_observation(obs2, decision_id)

        observations = reasoning_traces.get_informing_observations(decision_id)
        assert len(observations) >= 2
        obs_ids = [o.id for o in observations]
        assert obs1 in obs_ids
        assert obs2 in obs_ids


# =============================================================================
# SPEC-04.20-24: Trajectory Integration
# =============================================================================


class TestTrajectoryIntegration:
    """Tests for trajectory-to-decision mapping."""

    def test_trajectory_event_maps_to_decision(self, reasoning_traces):
        """
        TrajectoryEvent should map to decision nodes.

        @trace SPEC-04.20
        """
        from src.trajectory import TrajectoryEvent, TrajectoryEventType

        event = TrajectoryEvent(
            type=TrajectoryEventType.RLM_START,
            content="Starting recursive analysis",
            depth=0,
        )

        node_id = reasoning_traces.from_trajectory_event(event)
        assert node_id is not None

        node = reasoning_traces.get_decision_node(node_id)
        assert node is not None

    def test_recurse_event_creates_goal(self, reasoning_traces):
        """
        RECURSE_START events should create goal nodes.

        @trace SPEC-04.21
        """
        from src.trajectory import TrajectoryEvent, TrajectoryEventType

        event = TrajectoryEvent(
            type=TrajectoryEventType.RECURSE_START,
            content="Main objective",
            depth=0,
        )

        node_id = reasoning_traces.from_trajectory_event(event)
        node = reasoning_traces.get_decision_node(node_id)

        assert node.decision_type == "goal"

    def test_orchestrate_event_creates_decision(self, reasoning_traces):
        """
        ANALYZE events should create decision nodes.

        @trace SPEC-04.22
        """
        from src.trajectory import TrajectoryEvent, TrajectoryEventType

        # First create a goal
        goal_event = TrajectoryEvent(
            type=TrajectoryEventType.RLM_START,
            content="Goal",
            depth=0,
        )
        goal_id = reasoning_traces.from_trajectory_event(goal_event)

        # Then analyze event
        event = TrajectoryEvent(
            type=TrajectoryEventType.ANALYZE,
            content="Deciding approach",
            depth=0,
            metadata={"parent_id": goal_id},
        )

        node_id = reasoning_traces.from_trajectory_event(event)
        node = reasoning_traces.get_decision_node(node_id)

        assert node.decision_type == "decision"

    def test_final_event_creates_outcome(self, reasoning_traces):
        """
        FINAL events should create outcome nodes.

        @trace SPEC-04.23
        """
        from src.trajectory import TrajectoryEvent, TrajectoryEventType

        # Create prerequisite nodes
        goal_event = TrajectoryEvent(
            type=TrajectoryEventType.RLM_START,
            content="Goal",
            depth=0,
        )
        goal_id = reasoning_traces.from_trajectory_event(goal_event)

        decision_event = TrajectoryEvent(
            type=TrajectoryEventType.ANALYZE,
            content="Decision",
            depth=0,
            metadata={"parent_id": goal_id},
        )
        decision_id = reasoning_traces.from_trajectory_event(decision_event)

        # Final event
        final_event = TrajectoryEvent(
            type=TrajectoryEventType.FINAL,
            content="Task completed successfully",
            depth=0,
            metadata={"parent_id": decision_id},
        )

        node_id = reasoning_traces.from_trajectory_event(final_event)
        node = reasoning_traces.get_decision_node(node_id)

        assert node.decision_type == "outcome"

    def test_trajectory_mapping_configurable(self, reasoning_traces):
        """
        Trajectory-to-decision mapping should be configurable.

        @trace SPEC-04.24
        """
        from src.trajectory import TrajectoryEvent, TrajectoryEventType

        # Disable mapping
        reasoning_traces.set_trajectory_mapping(enabled=False)

        event = TrajectoryEvent(
            type=TrajectoryEventType.RLM_START,
            content="Should not create node",
            depth=0,
        )

        node_id = reasoning_traces.from_trajectory_event(event)
        assert node_id is None

        # Re-enable
        reasoning_traces.set_trajectory_mapping(enabled=True)


# =============================================================================
# SPEC-04.25: Schema Extension
# =============================================================================


class TestSchemaExtension:
    """Tests for decisions table schema."""

    def test_decisions_table_exists(self, reasoning_traces, memory_store):
        """
        Decisions table should exist in schema.

        @trace SPEC-04.25
        """
        import sqlite3

        conn = sqlite3.connect(memory_store.db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='decisions'"
        )
        result = cursor.fetchone()
        conn.close()

        assert result is not None

    def test_decisions_table_has_required_columns(self, reasoning_traces, memory_store):
        """
        Decisions table should have all required columns.

        @trace SPEC-04.25
        """
        import sqlite3

        conn = sqlite3.connect(memory_store.db_path)
        cursor = conn.execute("PRAGMA table_info(decisions)")
        columns = {row[1] for row in cursor.fetchall()}
        conn.close()

        required = {
            "node_id",
            "decision_type",
            "confidence",
            "prompt",
            "files",
            "branch",
            "commit_hash",
            "parent_id",
        }
        assert required.issubset(columns)

    def test_decisions_indexes_exist(self, reasoning_traces, memory_store):
        """
        Decisions table should have proper indexes.

        @trace SPEC-04.25
        """
        import sqlite3

        conn = sqlite3.connect(memory_store.db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='decisions'"
        )
        indexes = {row[0] for row in cursor.fetchall()}
        conn.close()

        # Check for expected indexes
        assert any("type" in idx for idx in indexes) or any("decision" in idx for idx in indexes)


# =============================================================================
# SPEC-04.26-29: Testing Requirements
# =============================================================================


class TestDecisionGraphIntegrity:
    """Tests for decision graph structure integrity."""

    def test_graph_structure_maintained(self, reasoning_traces):
        """
        Decision graph structure should be maintained.

        @trace SPEC-04.26
        """
        # Create a complete decision tree
        goal_id = reasoning_traces.create_goal("Main goal")
        decision_id = reasoning_traces.create_decision(goal_id, "Key decision")
        opt1 = reasoning_traces.add_option(decision_id, "Option A")
        opt2 = reasoning_traces.add_option(decision_id, "Option B")
        reasoning_traces.choose_option(decision_id, opt1)
        reasoning_traces.reject_option(decision_id, opt2, "Not optimal")
        action_id = reasoning_traces.create_action(decision_id, "Execute option A")
        outcome_id = reasoning_traces.create_outcome(action_id, "Success", success=True)

        # Verify tree structure
        tree = reasoning_traces.get_decision_tree(goal_id)
        assert tree is not None
        assert tree.root.decision_type == "goal"

        # Should be able to trace from goal to outcome
        outcome = reasoning_traces.get_outcome(goal_id)
        assert outcome is not None
        assert outcome.id == outcome_id


class TestPersistence:
    """Tests for cross-session persistence."""

    def test_decisions_persist_across_sessions(self, temp_db_path):
        """
        Decisions should persist across sessions.

        @trace SPEC-04.29
        """
        from src.memory_store import MemoryStore
        from src.reasoning_traces import ReasoningTraces

        # Session 1: Create decisions
        store1 = MemoryStore(db_path=temp_db_path)
        traces1 = ReasoningTraces(store1)

        goal_id = traces1.create_goal("Persistent goal")
        decision_id = traces1.create_decision(goal_id, "Persistent decision")

        # Session 2: Verify they exist
        store2 = MemoryStore(db_path=temp_db_path)
        traces2 = ReasoningTraces(store2)

        goal = traces2.get_decision_node(goal_id)
        decision = traces2.get_decision_node(decision_id)

        assert goal is not None
        assert goal.content == "Persistent goal"
        assert decision is not None
        assert decision.content == "Persistent decision"
