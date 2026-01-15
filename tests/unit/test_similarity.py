"""
Unit tests for semantic similarity module.

Implements: SPEC-16.08 Unit tests for semantic similarity
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from src.epistemic import (
    EmbeddingSimilarity,
    LLMJudgeSimilarity,
    SemanticSimilarity,
    SimilarityConfig,
    SimilarityMethod,
    SimilarityResult,
    cosine_similarity,
    text_overlap_similarity,
)


@dataclass
class MockAPIResponse:
    """Mock API response for testing."""

    content: str
    input_tokens: int = 100
    output_tokens: int = 50
    model: str = "haiku"


class MockLLMClient:
    """Mock LLM client that returns predefined responses."""

    def __init__(self, responses: list[str] | None = None):
        self.responses = responses or []
        self.call_count = 0
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> MockAPIResponse:
        self.calls.append(
            {
                "messages": messages,
                "system": system,
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )

        response_content = ""
        if self.call_count < len(self.responses):
            response_content = self.responses[self.call_count]
        self.call_count += 1

        return MockAPIResponse(content=response_content, model=model or "haiku")


class MockEmbeddingProvider:
    """Mock embedding provider for testing."""

    def __init__(self, embeddings: dict[str, list[float]] | None = None):
        self.embeddings = embeddings or {}
        self.call_count = 0

    def embed(self, text: str) -> list[float]:
        self.call_count += 1
        if text in self.embeddings:
            return self.embeddings[text]
        # Return a simple hash-based embedding for testing
        return [float(hash(text) % 100) / 100 for _ in range(8)]


class TestSimilarityResult:
    """Tests for SimilarityResult dataclass."""

    def test_basic_creation(self) -> None:
        """Test basic result creation."""
        result = SimilarityResult(
            score=0.85,
            method=SimilarityMethod.EMBEDDING,
        )
        assert result.score == 0.85
        assert result.method == SimilarityMethod.EMBEDDING
        assert result.embedding_score is None
        assert result.llm_score is None

    def test_full_creation(self) -> None:
        """Test result with all fields."""
        result = SimilarityResult(
            score=0.85,
            method=SimilarityMethod.ENSEMBLE,
            embedding_score=0.8,
            llm_score=0.9,
            confidence=0.95,
            reasoning="Both methods agree closely",
        )
        assert result.score == 0.85
        assert result.embedding_score == 0.8
        assert result.llm_score == 0.9
        assert result.confidence == 0.95

    def test_invalid_score_raises(self) -> None:
        """Test that invalid scores raise ValueError."""
        with pytest.raises(ValueError, match="score must be between"):
            SimilarityResult(score=1.5, method=SimilarityMethod.EMBEDDING)

        with pytest.raises(ValueError, match="score must be between"):
            SimilarityResult(score=-0.1, method=SimilarityMethod.EMBEDDING)

    def test_invalid_embedding_score_raises(self) -> None:
        """Test that invalid embedding_score raises ValueError."""
        with pytest.raises(ValueError, match="embedding_score must be between"):
            SimilarityResult(
                score=0.5,
                method=SimilarityMethod.EMBEDDING,
                embedding_score=1.5,
            )

    def test_invalid_llm_score_raises(self) -> None:
        """Test that invalid llm_score raises ValueError."""
        with pytest.raises(ValueError, match="llm_score must be between"):
            SimilarityResult(
                score=0.5,
                method=SimilarityMethod.LLM_JUDGE,
                llm_score=-0.1,
            )

    def test_boundary_scores(self) -> None:
        """Test boundary values are accepted."""
        result = SimilarityResult(
            score=0.0,
            method=SimilarityMethod.EMBEDDING,
            embedding_score=0.0,
            llm_score=1.0,
        )
        assert result.score == 0.0
        assert result.embedding_score == 0.0
        assert result.llm_score == 1.0


class TestSimilarityConfig:
    """Tests for SimilarityConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = SimilarityConfig()
        assert config.embedding_weight == 0.4
        assert config.llm_weight == 0.6
        assert config.borderline_low == 0.4
        assert config.borderline_high == 0.7
        assert config.use_llm_for_borderline is True

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = SimilarityConfig(
            embedding_weight=0.5,
            llm_weight=0.5,
            borderline_low=0.3,
            borderline_high=0.8,
        )
        assert config.embedding_weight == 0.5
        assert config.llm_weight == 0.5

    def test_invalid_embedding_weight(self) -> None:
        """Test invalid embedding_weight raises ValueError."""
        with pytest.raises(ValueError, match="embedding_weight must be between"):
            SimilarityConfig(embedding_weight=1.5)

    def test_invalid_llm_weight(self) -> None:
        """Test invalid llm_weight raises ValueError."""
        with pytest.raises(ValueError, match="llm_weight must be between"):
            SimilarityConfig(llm_weight=-0.1)

    def test_invalid_borderline_order(self) -> None:
        """Test invalid borderline order raises ValueError."""
        with pytest.raises(ValueError, match="borderline_low must be less than"):
            SimilarityConfig(borderline_low=0.8, borderline_high=0.3)


class TestEmbeddingSimilarity:
    """Tests for EmbeddingSimilarity class."""

    def test_identical_texts(self) -> None:
        """Test identical texts have perfect similarity."""
        provider = MockEmbeddingProvider(embeddings={"hello": [1.0, 0.0, 0.0, 0.0]})
        similarity = EmbeddingSimilarity(provider)
        score = similarity.compare("hello", "hello")
        assert score == 1.0

    def test_orthogonal_vectors(self) -> None:
        """Test orthogonal vectors have similarity 0.5 (normalized from 0)."""
        provider = MockEmbeddingProvider(
            embeddings={
                "a": [1.0, 0.0, 0.0, 0.0],
                "b": [0.0, 1.0, 0.0, 0.0],
            }
        )
        similarity = EmbeddingSimilarity(provider)
        score = similarity.compare("a", "b")
        # Cosine of 90 degrees is 0, normalized to 0.5
        assert score == 0.5

    def test_opposite_vectors(self) -> None:
        """Test opposite vectors have similarity 0."""
        provider = MockEmbeddingProvider(
            embeddings={
                "a": [1.0, 0.0],
                "b": [-1.0, 0.0],
            }
        )
        similarity = EmbeddingSimilarity(provider)
        score = similarity.compare("a", "b")
        # Cosine of 180 degrees is -1, normalized to 0
        assert score == 0.0

    def test_caching(self) -> None:
        """Test that embeddings are cached."""
        provider = MockEmbeddingProvider(embeddings={"hello": [1.0, 0.0, 0.0, 0.0]})
        similarity = EmbeddingSimilarity(provider)

        # First comparison
        similarity.compare("hello", "hello")
        assert provider.call_count == 1  # Only one unique text

        # Second comparison with same text
        similarity.compare("hello", "hello")
        assert provider.call_count == 1  # Still one (cached)

    def test_clear_cache(self) -> None:
        """Test cache clearing."""
        provider = MockEmbeddingProvider(embeddings={"hello": [1.0, 0.0, 0.0, 0.0]})
        similarity = EmbeddingSimilarity(provider)

        similarity.compare("hello", "hello")
        assert provider.call_count == 1

        similarity.clear_cache()
        similarity.compare("hello", "hello")
        assert provider.call_count == 2  # Re-embedded after clear

    def test_zero_magnitude_vector(self) -> None:
        """Test handling of zero magnitude vectors."""
        provider = MockEmbeddingProvider(
            embeddings={
                "a": [0.0, 0.0, 0.0, 0.0],
                "b": [1.0, 0.0, 0.0, 0.0],
            }
        )
        similarity = EmbeddingSimilarity(provider)
        score = similarity.compare("a", "b")
        assert score == 0.0


class TestLLMJudgeSimilarity:
    """Tests for LLMJudgeSimilarity class."""

    @pytest.mark.asyncio
    async def test_basic_comparison(self) -> None:
        """Test basic LLM comparison."""
        response = json.dumps(
            {
                "similarity_score": 0.9,
                "reasoning": "The texts convey the same meaning",
            }
        )
        client = MockLLMClient(responses=[response])
        similarity = LLMJudgeSimilarity(client)

        score, reasoning = await similarity.compare("text a", "text b")

        assert score == 0.9
        assert "same meaning" in reasoning

    @pytest.mark.asyncio
    async def test_uses_correct_model(self) -> None:
        """Test that correct model is used."""
        response = json.dumps({"similarity_score": 0.5, "reasoning": "OK"})
        client = MockLLMClient(responses=[response])
        similarity = LLMJudgeSimilarity(client, model="sonnet")

        await similarity.compare("a", "b")

        assert client.calls[0]["model"] == "sonnet"

    @pytest.mark.asyncio
    async def test_handles_invalid_json(self) -> None:
        """Test handling of invalid JSON response."""
        client = MockLLMClient(responses=["Not valid JSON"])
        similarity = LLMJudgeSimilarity(client)

        score, reasoning = await similarity.compare("a", "b")

        assert score == 0.5  # Default
        assert "Unable to parse" in reasoning

    @pytest.mark.asyncio
    async def test_clamps_out_of_range_score(self) -> None:
        """Test that out-of-range scores are clamped."""
        response = json.dumps({"similarity_score": 1.5, "reasoning": "OK"})
        client = MockLLMClient(responses=[response])
        similarity = LLMJudgeSimilarity(client)

        score, _ = await similarity.compare("a", "b")

        assert score == 1.0  # Clamped


class TestSemanticSimilarity:
    """Tests for SemanticSimilarity ensemble class."""

    @pytest.mark.asyncio
    async def test_identical_texts(self) -> None:
        """Test identical texts return perfect similarity."""
        similarity = SemanticSimilarity()
        result = await similarity.compare("hello", "hello")

        assert result.score == 1.0
        assert result.method == SimilarityMethod.EMBEDDING
        assert result.reasoning == "Texts are identical"

    @pytest.mark.asyncio
    async def test_empty_text(self) -> None:
        """Test empty texts return zero similarity."""
        similarity = SemanticSimilarity()
        result = await similarity.compare("hello", "   ")

        assert result.score == 0.0
        assert "empty" in result.reasoning.lower()

    @pytest.mark.asyncio
    async def test_embedding_only(self) -> None:
        """Test embedding-only comparison."""
        provider = MockEmbeddingProvider(
            embeddings={
                "hello world": [1.0, 0.0, 0.0, 0.0],
                "hi there": [0.9, 0.1, 0.0, 0.0],
            }
        )
        similarity = SemanticSimilarity(embedding_provider=provider)

        result = await similarity.compare("hello world", "hi there")

        assert result.method == SimilarityMethod.EMBEDDING
        assert result.embedding_score is not None
        assert result.llm_score is None

    @pytest.mark.asyncio
    async def test_borderline_triggers_llm(self) -> None:
        """Test that borderline scores trigger LLM judge."""
        provider = MockEmbeddingProvider(
            embeddings={
                "text a": [1.0, 0.0, 0.0, 0.0],
                "text b": [0.6, 0.4, 0.0, 0.0],  # Will give borderline score
            }
        )
        response = json.dumps({"similarity_score": 0.8, "reasoning": "Similar"})
        client = MockLLMClient(responses=[response])

        config = SimilarityConfig(
            borderline_low=0.4,
            borderline_high=0.95,
            use_llm_for_borderline=True,
        )
        similarity = SemanticSimilarity(
            embedding_provider=provider,
            llm_client=client,
            config=config,
        )

        result = await similarity.compare("text a", "text b")

        assert result.method == SimilarityMethod.ENSEMBLE
        assert result.embedding_score is not None
        assert result.llm_score == 0.8

    @pytest.mark.asyncio
    async def test_force_llm(self) -> None:
        """Test forcing LLM comparison."""
        provider = MockEmbeddingProvider(
            embeddings={
                "a": [1.0, 0.0, 0.0, 0.0],
                "b": [0.99, 0.01, 0.0, 0.0],  # Very similar - not borderline
            }
        )
        response = json.dumps({"similarity_score": 0.9, "reasoning": "Match"})
        client = MockLLMClient(responses=[response])

        similarity = SemanticSimilarity(
            embedding_provider=provider,
            llm_client=client,
        )

        result = await similarity.compare("a", "b", force_llm=True)

        assert result.method == SimilarityMethod.ENSEMBLE
        assert client.call_count == 1

    @pytest.mark.asyncio
    async def test_llm_only(self) -> None:
        """Test LLM-only comparison when no embedding provider."""
        response = json.dumps({"similarity_score": 0.85, "reasoning": "Good"})
        client = MockLLMClient(responses=[response])

        similarity = SemanticSimilarity(llm_client=client)
        result = await similarity.compare("text a", "text b", force_llm=True)

        assert result.method == SimilarityMethod.LLM_JUDGE
        assert result.llm_score == 0.85
        assert result.embedding_score is None

    def test_sync_comparison(self) -> None:
        """Test synchronous comparison."""
        provider = MockEmbeddingProvider(
            embeddings={
                "hello": [1.0, 0.0, 0.0, 0.0],
                "world": [0.5, 0.5, 0.0, 0.0],
            }
        )
        similarity = SemanticSimilarity(embedding_provider=provider)

        result = similarity.compare_sync("hello", "world")

        assert result.method == SimilarityMethod.EMBEDDING
        assert result.embedding_score is not None

    def test_sync_comparison_identical(self) -> None:
        """Test sync comparison with identical texts."""
        similarity = SemanticSimilarity()
        result = similarity.compare_sync("hello", "hello")
        assert result.score == 1.0

    def test_sync_comparison_empty(self) -> None:
        """Test sync comparison with empty text."""
        similarity = SemanticSimilarity()
        result = similarity.compare_sync("hello", "")
        assert result.score == 0.0

    def test_sync_comparison_no_provider_raises(self) -> None:
        """Test that sync without provider raises error."""
        similarity = SemanticSimilarity()
        with pytest.raises(ValueError, match="No embedding provider"):
            similarity.compare_sync("hello", "world")

    @pytest.mark.asyncio
    async def test_no_methods_available(self) -> None:
        """Test fallback when no methods available."""
        similarity = SemanticSimilarity()  # No providers
        result = await similarity.compare("a", "b")

        assert result.score == 0.5  # Fallback
        assert result.confidence == 0.0


class TestCosineSimilarity:
    """Tests for cosine_similarity helper function."""

    def test_identical_vectors(self) -> None:
        """Test identical vectors."""
        vec = [1.0, 2.0, 3.0]
        assert cosine_similarity(vec, vec) == 1.0

    def test_orthogonal_vectors(self) -> None:
        """Test orthogonal vectors."""
        vec_a = [1.0, 0.0]
        vec_b = [0.0, 1.0]
        assert cosine_similarity(vec_a, vec_b) == 0.5  # Normalized from 0

    def test_opposite_vectors(self) -> None:
        """Test opposite vectors."""
        vec_a = [1.0, 0.0]
        vec_b = [-1.0, 0.0]
        assert cosine_similarity(vec_a, vec_b) == 0.0  # Normalized from -1

    def test_dimension_mismatch_raises(self) -> None:
        """Test dimension mismatch raises error."""
        with pytest.raises(ValueError, match="same dimension"):
            cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_zero_vector(self) -> None:
        """Test zero vector handling."""
        assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0


class TestTextOverlapSimilarity:
    """Tests for text_overlap_similarity helper function."""

    def test_identical_texts(self) -> None:
        """Test identical texts."""
        assert text_overlap_similarity("hello world", "hello world") == 1.0

    def test_no_overlap(self) -> None:
        """Test texts with no overlap."""
        assert text_overlap_similarity("hello world", "foo bar") == 0.0

    def test_partial_overlap(self) -> None:
        """Test partial overlap."""
        score = text_overlap_similarity("hello world", "hello there")
        assert 0.0 < score < 1.0
        # "hello" is shared, "world" and "there" are different
        # Jaccard: 1 / 3 = 0.333...
        assert abs(score - 1 / 3) < 0.01

    def test_case_insensitive(self) -> None:
        """Test case insensitivity."""
        assert text_overlap_similarity("HELLO", "hello") == 1.0

    def test_both_empty(self) -> None:
        """Test both empty strings."""
        assert text_overlap_similarity("", "") == 1.0

    def test_one_empty(self) -> None:
        """Test one empty string."""
        assert text_overlap_similarity("hello", "") == 0.0
        assert text_overlap_similarity("", "world") == 0.0


class TestConfidenceComputation:
    """Tests for confidence computation in SemanticSimilarity."""

    @pytest.mark.asyncio
    async def test_high_agreement_high_confidence(self) -> None:
        """Test that method agreement increases confidence."""
        provider = MockEmbeddingProvider(
            embeddings={
                "text a": [1.0, 0.0, 0.0, 0.0],
                "text b": [0.6, 0.4, 0.0, 0.0],
            }
        )
        response = json.dumps({"similarity_score": 0.85, "reasoning": "OK"})
        client = MockLLMClient(responses=[response])

        config = SimilarityConfig(borderline_low=0.0, borderline_high=1.0)
        similarity = SemanticSimilarity(
            embedding_provider=provider,
            llm_client=client,
            config=config,
        )

        result = await similarity.compare("text a", "text b", force_llm=True)

        # Both scores should be close, so confidence should be high
        if result.embedding_score and result.llm_score:
            agreement = 1.0 - abs(result.embedding_score - result.llm_score)
            assert result.confidence == agreement

    @pytest.mark.asyncio
    async def test_non_borderline_high_confidence(self) -> None:
        """Test that non-borderline scores have high confidence."""
        provider = MockEmbeddingProvider(
            embeddings={
                "a": [1.0, 0.0, 0.0, 0.0],
                "b": [0.99, 0.01, 0.0, 0.0],  # Very similar
            }
        )
        similarity = SemanticSimilarity(embedding_provider=provider)

        result = await similarity.compare("a", "b")

        # Score is far from borderline (0.4-0.7), so confidence should be high
        assert result.confidence > 0.5
