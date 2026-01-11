"""
Model routing integration for RLM-Claude-Code.

Implements: Spec §5.3 Router Configuration
"""

from __future__ import annotations

from dataclasses import dataclass

from .api_client import MultiProviderClient, get_client
from .config import RLMConfig, default_config
from .cost_tracker import CostComponent


@dataclass
class CompletionResult:
    """Result of a model completion."""

    content: str
    model: str
    tokens_used: int
    input_tokens: int = 0
    output_tokens: int = 0


class ModelRouter:
    """
    Routes model calls based on RLM depth and query type.

    Implements: Spec §5.3 Router Configuration
    """

    def __init__(
        self,
        config: RLMConfig | None = None,
        client: MultiProviderClient | None = None,
    ):
        """
        Initialize router.

        Args:
            config: RLM configuration
            client: LLM client (uses global if not provided)
        """
        self.config = config or default_config
        self._client = client

    @property
    def client(self) -> MultiProviderClient:
        """Get LLM client, initializing if needed."""
        if self._client is None:
            self._client = get_client()
        return self._client

    def get_model(self, depth: int, query_type: str | None = None) -> str:
        """
        Get model for given depth and query type.

        Implements: Spec §5.3 Model Selection by Depth

        Args:
            depth: Current recursion depth
            query_type: Optional query type hint (reserved for future smart routing)

        Returns:
            Model identifier string
        """
        # query_type reserved for future smart routing by query type
        _ = query_type
        if depth == 0:
            return self.config.models.root_model
        elif depth == 1:
            return self.config.models.recursive_depth_1
        else:
            return self.config.models.recursive_depth_2

    async def complete(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 4000,
        depth: int = 0,
        system: str | None = None,
    ) -> CompletionResult:
        """
        Make a model completion call.

        Implements: Spec §5.1 API Integration

        Args:
            model: Model identifier
            prompt: Prompt to complete
            max_tokens: Max tokens in response
            depth: Current depth (for logging/cost tracking)
            system: Optional system prompt

        Returns:
            CompletionResult with response
        """
        # Determine cost component based on depth
        if depth == 0:
            component = CostComponent.ROOT_PROMPT
        else:
            component = CostComponent.RECURSIVE_CALL

        messages = [{"role": "user", "content": prompt}]

        response = await self.client.complete(
            messages=messages,
            system=system,
            model=model,
            max_tokens=max_tokens,
            component=component,
        )

        return CompletionResult(
            content=response.content,
            model=response.model,
            tokens_used=response.input_tokens + response.output_tokens,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )


def generate_router_config() -> dict:
    """
    Generate claude-code-router compatible configuration.

    Implements: Spec §5.3 Router Configuration
    """
    config = default_config

    return {
        "Router": {
            "default": "anthropic,claude-sonnet-4",
            "rlm_root": f"anthropic,{config.models.root_model}",
            "rlm_recursive": f"anthropic,{config.models.recursive_depth_1}",
            "rlm_recursive_deep": f"anthropic,{config.models.recursive_depth_2}",
            "rlm_max_depth": config.depth.max,
            "rlm_mode": config.activation.mode,
            "rlm_simple_bypass": config.hybrid.simple_query_bypass,
        },
        "rlm": {
            "activation": {
                "mode": config.activation.mode,
                "fallback_token_threshold": config.activation.fallback_token_threshold,
                "complexity_score_threshold": config.activation.complexity_score_threshold,
            },
            "depth": {
                "default": config.depth.default,
                "max": config.depth.max,
                "spawn_repl_at_depth_1": config.depth.spawn_repl_at_depth_1,
            },
            "hybrid": {
                "enabled": config.hybrid.enabled,
                "simple_query_bypass": config.hybrid.simple_query_bypass,
                "simple_confidence_threshold": config.hybrid.simple_confidence_threshold,
            },
            "trajectory": {
                "verbosity": config.trajectory.verbosity,
                "streaming": config.trajectory.streaming,
                "colors": config.trajectory.colors,
                "export_enabled": config.trajectory.export_enabled,
                "export_path": config.trajectory.export_path,
            },
            "models": {
                "root_model": config.models.root_model,
                "recursive_depth_1": config.models.recursive_depth_1,
                "recursive_depth_2": config.models.recursive_depth_2,
            },
            "cost_controls": {
                "max_recursive_calls_per_turn": config.cost_controls.max_recursive_calls_per_turn,
                "max_tokens_per_recursive_call": config.cost_controls.max_tokens_per_recursive_call,
                "abort_on_cost_threshold": config.cost_controls.abort_on_cost_threshold,
            },
        },
    }


__all__ = ["ModelRouter", "CompletionResult", "generate_router_config"]
