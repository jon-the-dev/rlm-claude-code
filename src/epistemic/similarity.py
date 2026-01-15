"""
Semantic similarity for epistemic verification.

Implements: SPEC-16.08 Semantic similarity module

Provides principled semantic similarity comparison using:
1. Embedding distance (fast first pass)
2. LLM-as-judge (accurate second pass for borderline cases)
3. Ensemble scoring with calibrated thresholds
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from src.api_client import APIResponse
    from src.embedding_retrieval import EmbeddingProvider


class LLMClient(Protocol):
    """Protocol for LLM client to enable dependency injection."""

    async def complete(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> APIResponse: ...


class SimilarityMethod(str, Enum):
    """Method used for similarity comparison."""

    EMBEDDING = "embedding"
    LLM_JUDGE = "llm_judge"
    ENSEMBLE = "ensemble"


@dataclass
class SimilarityResult:
    """
    Result of semantic similarity comparison.

    Attributes:
        score: Overall similarity score (0.0-1.0)
        method: Method used for final score
        embedding_score: Score from embedding comparison (if computed)
        llm_score: Score from LLM judge (if computed)
        confidence: Confidence in the result (0.0-1.0)
        reasoning: Explanation of the similarity assessment
    """

    score: float
    method: SimilarityMethod
    embedding_score: float | None = None
    llm_score: float | None = None
    confidence: float = 1.0
    reasoning: str = ""

    def __post_init__(self) -> None:
        """Validate score ranges."""
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score must be between 0.0 and 1.0, got {self.score}")
        if self.embedding_score is not None and not 0.0 <= self.embedding_score <= 1.0:
            raise ValueError(
                f"embedding_score must be between 0.0 and 1.0, got {self.embedding_score}"
            )
        if self.llm_score is not None and not 0.0 <= self.llm_score <= 1.0:
            raise ValueError(f"llm_score must be between 0.0 and 1.0, got {self.llm_score}")


@dataclass
class SimilarityConfig:
    """
    Configuration for semantic similarity.

    Attributes:
        embedding_weight: Weight for embedding score in ensemble (0.0-1.0)
        llm_weight: Weight for LLM score in ensemble (0.0-1.0)
        borderline_low: Lower threshold for borderline cases
        borderline_high: Upper threshold for borderline cases
        use_llm_for_borderline: Whether to use LLM for borderline cases
        llm_model: Model to use for LLM-as-judge
    """

    embedding_weight: float = 0.4
    llm_weight: float = 0.6
    borderline_low: float = 0.4
    borderline_high: float = 0.7
    use_llm_for_borderline: bool = True
    llm_model: str = "haiku"

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not 0.0 <= self.embedding_weight <= 1.0:
            raise ValueError(
                f"embedding_weight must be between 0.0 and 1.0, got {self.embedding_weight}"
            )
        if not 0.0 <= self.llm_weight <= 1.0:
            raise ValueError(f"llm_weight must be between 0.0 and 1.0, got {self.llm_weight}")
        if self.borderline_low >= self.borderline_high:
            raise ValueError("borderline_low must be less than borderline_high")


# Prompts for LLM-as-judge
LLM_JUDGE_SYSTEM = """You are an expert at evaluating semantic similarity between texts.

Your task is to determine how semantically similar two texts are, considering:
1. Do they convey the same meaning?
2. Do they make the same claims or assertions?
3. Would they answer the same question equivalently?

Ignore superficial differences in wording, formatting, or style.
Focus on the underlying meaning and intent."""

LLM_JUDGE_PROMPT = """Rate the semantic similarity between these two texts.

TEXT A:
{text_a}

TEXT B:
{text_b}

Provide your assessment:
1. similarity_score: 0.0 (completely different meaning) to 1.0 (identical meaning)
2. reasoning: Brief explanation of your assessment

Consider:
- Same meaning with different words → high similarity (0.8-1.0)
- Related but different claims → medium similarity (0.4-0.7)
- Contradictory or unrelated → low similarity (0.0-0.3)

Respond in JSON format:
{{
  "similarity_score": 0.85,
  "reasoning": "Both texts express the same core idea..."
}}"""


class EmbeddingSimilarity:
    """
    Embedding-based semantic similarity.

    Fast comparison using cosine similarity of text embeddings.
    """

    def __init__(self, provider: EmbeddingProvider):
        """
        Initialize embedding similarity.

        Args:
            provider: Embedding provider for computing vectors
        """
        self.provider = provider
        self._cache: dict[str, list[float]] = {}

    def compare(self, text_a: str, text_b: str) -> float:
        """
        Compare two texts using embedding cosine similarity.

        Args:
            text_a: First text
            text_b: Second text

        Returns:
            Similarity score between 0.0 and 1.0
        """
        # Get embeddings (with caching)
        emb_a = self._get_embedding(text_a)
        emb_b = self._get_embedding(text_b)

        # Compute cosine similarity
        return self._cosine_similarity(emb_a, emb_b)

    def _get_embedding(self, text: str) -> list[float]:
        """Get embedding for text, using cache if available."""
        if text not in self._cache:
            self._cache[text] = self.provider.embed(text)
        return self._cache[text]

    def _cosine_similarity(self, vec_a: list[float], vec_b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(vec_a) != len(vec_b):
            raise ValueError("Vectors must have same dimension")

        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        mag_a = math.sqrt(sum(a * a for a in vec_a))
        mag_b = math.sqrt(sum(b * b for b in vec_b))

        if mag_a == 0 or mag_b == 0:
            return 0.0

        # Cosine similarity is in [-1, 1], normalize to [0, 1]
        cosine = dot_product / (mag_a * mag_b)
        return (cosine + 1) / 2

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()


class LLMJudgeSimilarity:
    """
    LLM-as-judge semantic similarity.

    Accurate comparison using Claude to evaluate semantic equivalence.
    More expensive but handles nuanced cases better than embeddings.
    """

    def __init__(self, client: LLMClient, model: str = "haiku"):
        """
        Initialize LLM judge similarity.

        Args:
            client: LLM client for API calls
            model: Model to use for judging
        """
        self.client = client
        self.model = model

    async def compare(self, text_a: str, text_b: str) -> tuple[float, str]:
        """
        Compare two texts using LLM-as-judge.

        Args:
            text_a: First text
            text_b: Second text

        Returns:
            Tuple of (similarity_score, reasoning)
        """
        prompt = LLM_JUDGE_PROMPT.format(text_a=text_a, text_b=text_b)

        response = await self.client.complete(
            messages=[{"role": "user", "content": prompt}],
            system=LLM_JUDGE_SYSTEM,
            model=self.model,
            max_tokens=512,
            temperature=0.0,
        )

        return self._parse_response(response.content)

    def _parse_response(self, content: str) -> tuple[float, str]:
        """Parse the JSON response from LLM judge."""
        # Try to extract JSON from response
        json_match = re.search(r"\{[\s\S]*\}", content)
        if not json_match:
            return 0.5, "Unable to parse response"

        try:
            data = json.loads(json_match.group())
            score = float(data.get("similarity_score", 0.5))
            # Clamp to valid range
            score = max(0.0, min(1.0, score))
            reasoning = str(data.get("reasoning", ""))
            return score, reasoning
        except (json.JSONDecodeError, TypeError, ValueError):
            return 0.5, "Unable to parse response"


class SemanticSimilarity:
    """
    Ensemble semantic similarity using embeddings and LLM-as-judge.

    Combines fast embedding comparison with accurate LLM judgment:
    1. Always compute embedding similarity (fast)
    2. For borderline cases, also use LLM-as-judge
    3. Return weighted ensemble score

    Example:
        >>> similarity = SemanticSimilarity(embedding_provider, llm_client)
        >>> result = await similarity.compare("X equals Y", "Y is the same as X")
        >>> result.score
        0.92
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider | None = None,
        llm_client: LLMClient | None = None,
        config: SimilarityConfig | None = None,
    ):
        """
        Initialize semantic similarity.

        Args:
            embedding_provider: Provider for embedding computation
            llm_client: Client for LLM-as-judge calls
            config: Similarity configuration
        """
        self.config = config or SimilarityConfig()
        self.embedding_sim = EmbeddingSimilarity(embedding_provider) if embedding_provider else None
        self.llm_sim = LLMJudgeSimilarity(llm_client, self.config.llm_model) if llm_client else None

    async def compare(
        self,
        text_a: str,
        text_b: str,
        force_llm: bool = False,
    ) -> SimilarityResult:
        """
        Compare two texts semantically.

        Args:
            text_a: First text
            text_b: Second text
            force_llm: Force LLM comparison even for non-borderline cases

        Returns:
            SimilarityResult with score and metadata
        """
        # Handle identical texts
        if text_a == text_b:
            return SimilarityResult(
                score=1.0,
                method=SimilarityMethod.EMBEDDING,
                embedding_score=1.0,
                confidence=1.0,
                reasoning="Texts are identical",
            )

        # Handle empty texts
        if not text_a.strip() or not text_b.strip():
            return SimilarityResult(
                score=0.0,
                method=SimilarityMethod.EMBEDDING,
                embedding_score=0.0,
                confidence=1.0,
                reasoning="One or both texts are empty",
            )

        embedding_score: float | None = None
        llm_score: float | None = None
        reasoning = ""

        # Step 1: Compute embedding similarity (fast)
        if self.embedding_sim:
            embedding_score = self.embedding_sim.compare(text_a, text_b)

        # Step 2: Determine if we need LLM judge
        need_llm = force_llm or (
            self.config.use_llm_for_borderline
            and embedding_score is not None
            and self.config.borderline_low <= embedding_score <= self.config.borderline_high
        )

        # Step 3: Use LLM judge if needed
        if need_llm and self.llm_sim:
            llm_score, reasoning = await self.llm_sim.compare(text_a, text_b)

        # Step 4: Compute final score
        return self._compute_ensemble_result(embedding_score, llm_score, reasoning)

    def compare_sync(self, text_a: str, text_b: str) -> SimilarityResult:
        """
        Synchronous comparison using only embeddings.

        Useful when async is not available or LLM is not needed.

        Args:
            text_a: First text
            text_b: Second text

        Returns:
            SimilarityResult with embedding-only score
        """
        if text_a == text_b:
            return SimilarityResult(
                score=1.0,
                method=SimilarityMethod.EMBEDDING,
                embedding_score=1.0,
                confidence=1.0,
                reasoning="Texts are identical",
            )

        if not text_a.strip() or not text_b.strip():
            return SimilarityResult(
                score=0.0,
                method=SimilarityMethod.EMBEDDING,
                embedding_score=0.0,
                confidence=1.0,
                reasoning="One or both texts are empty",
            )

        if not self.embedding_sim:
            raise ValueError("No embedding provider configured")

        embedding_score = self.embedding_sim.compare(text_a, text_b)

        return SimilarityResult(
            score=embedding_score,
            method=SimilarityMethod.EMBEDDING,
            embedding_score=embedding_score,
            confidence=self._compute_confidence(embedding_score, None),
            reasoning="Embedding-only comparison",
        )

    def _compute_ensemble_result(
        self,
        embedding_score: float | None,
        llm_score: float | None,
        reasoning: str,
    ) -> SimilarityResult:
        """Compute the final ensemble result."""
        # Case 1: Only embedding score
        if embedding_score is not None and llm_score is None:
            return SimilarityResult(
                score=embedding_score,
                method=SimilarityMethod.EMBEDDING,
                embedding_score=embedding_score,
                confidence=self._compute_confidence(embedding_score, None),
                reasoning="Embedding-only comparison",
            )

        # Case 2: Only LLM score
        if embedding_score is None and llm_score is not None:
            return SimilarityResult(
                score=llm_score,
                method=SimilarityMethod.LLM_JUDGE,
                llm_score=llm_score,
                confidence=self._compute_confidence(None, llm_score),
                reasoning=reasoning,
            )

        # Case 3: Both scores - weighted ensemble
        if embedding_score is not None and llm_score is not None:
            # Normalize weights
            total_weight = self.config.embedding_weight + self.config.llm_weight
            emb_w = self.config.embedding_weight / total_weight
            llm_w = self.config.llm_weight / total_weight

            ensemble_score = emb_w * embedding_score + llm_w * llm_score

            return SimilarityResult(
                score=ensemble_score,
                method=SimilarityMethod.ENSEMBLE,
                embedding_score=embedding_score,
                llm_score=llm_score,
                confidence=self._compute_confidence(embedding_score, llm_score),
                reasoning=reasoning,
            )

        # Case 4: Neither score available
        return SimilarityResult(
            score=0.5,
            method=SimilarityMethod.EMBEDDING,
            confidence=0.0,
            reasoning="No similarity method available",
        )

    def _compute_confidence(
        self,
        embedding_score: float | None,
        llm_score: float | None,
    ) -> float:
        """
        Compute confidence in the similarity result.

        Higher confidence when:
        - Both methods agree
        - Scores are far from borderline thresholds
        """
        if embedding_score is None and llm_score is None:
            return 0.0

        if embedding_score is not None and llm_score is not None:
            # Agreement between methods increases confidence
            agreement = 1.0 - abs(embedding_score - llm_score)
            return agreement

        # Single method confidence based on distance from borderline
        score = embedding_score if embedding_score is not None else llm_score
        if score is None:
            return 0.0

        # Distance from borderline zone increases confidence
        if score < self.config.borderline_low:
            return min(1.0, (self.config.borderline_low - score) / self.config.borderline_low)
        if score > self.config.borderline_high:
            return min(
                1.0,
                (score - self.config.borderline_high) / (1.0 - self.config.borderline_high),
            )

        # In borderline zone - lower confidence
        return 0.5


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    Standalone utility function for direct vector comparison.

    Args:
        vec_a: First vector
        vec_b: Second vector

    Returns:
        Similarity score normalized to [0, 1]
    """
    if len(vec_a) != len(vec_b):
        raise ValueError("Vectors must have same dimension")

    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a * a for a in vec_a))
    mag_b = math.sqrt(sum(b * b for b in vec_b))

    if mag_a == 0 or mag_b == 0:
        return 0.0

    cosine = dot_product / (mag_a * mag_b)
    return (cosine + 1) / 2


def text_overlap_similarity(text_a: str, text_b: str) -> float:
    """
    Simple text overlap similarity.

    Fast baseline using word overlap. Useful when embeddings unavailable.

    Args:
        text_a: First text
        text_b: Second text

    Returns:
        Jaccard similarity of word sets
    """
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())

    if not words_a and not words_b:
        return 1.0
    if not words_a or not words_b:
        return 0.0

    intersection = words_a & words_b
    union = words_a | words_b

    return len(intersection) / len(union)
