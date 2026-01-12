"""
RLM-Claude-Code: Recursive Language Model integration for Claude Code.

Transform Claude Code into an RLM agent for unbounded context handling
and improved reasoning via REPL-based context decomposition.

Implements intelligent orchestration with:
- Automatic complexity-based RLM activation
- Claude-powered orchestration decisions
- Strategy learning from successful trajectories
- User-configurable preferences and budgets
"""

__version__ = "0.1.0"

# Core orchestration
from .orchestrator import RLMOrchestrator
from .intelligent_orchestrator import IntelligentOrchestrator
from .auto_activation import AutoActivator, check_auto_activation

# Context and REPL
from .context_manager import externalize_context
from .repl_environment import RLMEnvironment
from .types import SessionContext

# Complexity and routing
from .complexity_classifier import should_activate_rlm
from .orchestration_schema import ExecutionMode, OrchestrationPlan, ToolAccessLevel

# Trajectory and analysis
from .trajectory import TrajectoryEvent, TrajectoryRenderer
from .trajectory_analysis import TrajectoryAnalyzer, StrategyType

# User preferences and tools
from .user_preferences import UserPreferences, PreferencesManager
from .tool_bridge import ToolBridge, ToolPermissions
from .strategy_cache import StrategyCache

__all__ = [
    # Core
    "RLMOrchestrator",
    "IntelligentOrchestrator",
    "AutoActivator",
    "check_auto_activation",
    # Context
    "SessionContext",
    "externalize_context",
    "RLMEnvironment",
    # Complexity
    "should_activate_rlm",
    "ExecutionMode",
    "OrchestrationPlan",
    "ToolAccessLevel",
    # Trajectory
    "TrajectoryEvent",
    "TrajectoryRenderer",
    "TrajectoryAnalyzer",
    "StrategyType",
    # Preferences and tools
    "UserPreferences",
    "PreferencesManager",
    "ToolBridge",
    "ToolPermissions",
    "StrategyCache",
]
