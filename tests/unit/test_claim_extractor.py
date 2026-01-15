"""
Unit tests for claim extraction.

Implements: SPEC-16.09 Unit tests for claim extraction
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from src.epistemic import (
    ClaimExtractor,
    ExtractedClaim,
    ExtractionResult,
    extract_evidence_references,
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


class TestExtractedClaim:
    """Tests for ExtractedClaim dataclass."""

    def test_basic_creation(self) -> None:
        """Test basic claim creation."""
        claim = ExtractedClaim(
            claim_id="c1",
            claim_text="The function returns 42",
            evidence_ids=["e1"],
        )
        assert claim.claim_id == "c1"
        assert claim.claim_text == "The function returns 42"
        assert claim.evidence_ids == ["e1"]
        assert claim.confidence == 1.0
        assert not claim.is_critical

    def test_default_values(self) -> None:
        """Test default values are set correctly."""
        claim = ExtractedClaim(claim_id="c1", claim_text="Test")
        assert claim.original_span == ""
        assert claim.evidence_ids == []
        assert claim.metadata == {}


class TestExtractionResult:
    """Tests for ExtractionResult dataclass."""

    def test_basic_creation(self) -> None:
        """Test basic result creation."""
        result = ExtractionResult(
            claims=[],
            response_id="r1",
            total_spans=5,
            extraction_model="haiku",
        )
        assert result.response_id == "r1"
        assert result.total_spans == 5
        assert result.extraction_model == "haiku"


class TestClaimExtractor:
    """Tests for ClaimExtractor class."""

    @pytest.mark.asyncio
    async def test_extract_empty_text(self) -> None:
        """Test extraction from empty text returns no claims."""
        client = MockLLMClient()
        extractor = ClaimExtractor(client)

        result = await extractor.extract_claims("")
        assert len(result.claims) == 0

    @pytest.mark.asyncio
    async def test_extract_trivial_text(self) -> None:
        """Test extraction from very short text returns no claims."""
        client = MockLLMClient()
        extractor = ClaimExtractor(client)

        result = await extractor.extract_claims("Hi")
        assert len(result.claims) == 0

    @pytest.mark.asyncio
    async def test_extract_single_claim(self) -> None:
        """Test extraction of a single claim."""
        response = json.dumps(
            [
                {
                    "claim": "The function calculate_total returns a number",
                    "original_span": "The calculate_total function returns...",
                    "cites_evidence": ["src/utils.py:42"],
                    "is_critical": True,
                    "confidence": 0.95,
                }
            ]
        )
        client = MockLLMClient(responses=[response])
        extractor = ClaimExtractor(client)

        result = await extractor.extract_claims(
            "The calculate_total function returns a number based on input"
        )

        assert len(result.claims) == 1
        claim = result.claims[0]
        assert "calculate_total" in claim.claim_text
        assert claim.is_critical
        assert claim.confidence == 0.95
        assert "src/utils.py:42" in claim.evidence_ids

    @pytest.mark.asyncio
    async def test_extract_multiple_claims(self) -> None:
        """Test extraction of multiple claims."""
        response = json.dumps(
            [
                {"claim": "Claim 1", "confidence": 0.9, "is_critical": False},
                {"claim": "Claim 2", "confidence": 0.8, "is_critical": True},
                {"claim": "Claim 3", "confidence": 0.7, "is_critical": False},
            ]
        )
        client = MockLLMClient(responses=[response])
        extractor = ClaimExtractor(client)

        result = await extractor.extract_claims("Text with multiple claims")

        assert len(result.claims) == 3
        assert result.claims[0].claim_text == "Claim 1"
        assert result.claims[1].claim_text == "Claim 2"
        assert result.claims[2].claim_text == "Claim 3"

    @pytest.mark.asyncio
    async def test_extract_respects_max_claims(self) -> None:
        """Test that max_claims limit is respected."""
        # Create response with 10 claims
        claims = [
            {"claim": f"Claim {i}", "confidence": 0.9 - i * 0.05, "is_critical": i == 5}
            for i in range(10)
        ]
        response = json.dumps(claims)
        client = MockLLMClient(responses=[response])
        extractor = ClaimExtractor(client, max_claims=5)

        result = await extractor.extract_claims("Text with many claims")

        assert len(result.claims) <= 5
        # Critical claim should be preserved
        critical_claims = [c for c in result.claims if c.is_critical]
        assert len(critical_claims) >= 1

    @pytest.mark.asyncio
    async def test_extract_handles_invalid_json(self) -> None:
        """Test extraction handles invalid JSON gracefully."""
        client = MockLLMClient(responses=["This is not JSON"])
        extractor = ClaimExtractor(client)

        result = await extractor.extract_claims("Some text to analyze")

        # Should return empty claims, not crash
        assert len(result.claims) == 0

    @pytest.mark.asyncio
    async def test_extract_handles_partial_json(self) -> None:
        """Test extraction handles partially valid JSON."""
        response = """Here are the claims:
        [
            {"claim": "Valid claim", "confidence": 0.9},
            {"invalid": "missing claim field"},
            {"claim": "Another valid claim"}
        ]
        """
        client = MockLLMClient(responses=[response])
        extractor = ClaimExtractor(client)

        result = await extractor.extract_claims("Text to analyze")

        # Should extract valid claims only
        assert len(result.claims) == 2
        assert result.claims[0].claim_text == "Valid claim"
        assert result.claims[1].claim_text == "Another valid claim"

    @pytest.mark.asyncio
    async def test_extract_uses_correct_model(self) -> None:
        """Test extraction uses specified model."""
        client = MockLLMClient(responses=["[]"])
        extractor = ClaimExtractor(client, default_model="sonnet")

        await extractor.extract_claims("Some longer text that exceeds minimum", model="opus")

        assert client.calls[0]["model"] == "opus"

    @pytest.mark.asyncio
    async def test_extract_uses_default_model(self) -> None:
        """Test extraction uses default model when not specified."""
        client = MockLLMClient(responses=["[]"])
        extractor = ClaimExtractor(client, default_model="sonnet")

        await extractor.extract_claims("Some longer text that exceeds minimum")

        assert client.calls[0]["model"] == "sonnet"

    @pytest.mark.asyncio
    async def test_map_claims_to_evidence_empty(self) -> None:
        """Test mapping with empty claims or evidence."""
        client = MockLLMClient()
        extractor = ClaimExtractor(client)

        # Empty claims
        result = await extractor.map_claims_to_evidence([], {"e1": "evidence"})
        assert result == []

        # Empty evidence
        claims = [ExtractedClaim(claim_id="c1", claim_text="Test")]
        result = await extractor.map_claims_to_evidence(claims, {})
        assert result == claims

    @pytest.mark.asyncio
    async def test_map_claims_to_evidence(self) -> None:
        """Test mapping claims to evidence."""
        mapping_response = json.dumps(
            {
                "0": ["e1", "e2"],
                "1": ["e2"],
            }
        )
        client = MockLLMClient(responses=[mapping_response])
        extractor = ClaimExtractor(client)

        claims = [
            ExtractedClaim(claim_id="c1", claim_text="Claim 1"),
            ExtractedClaim(claim_id="c2", claim_text="Claim 2"),
        ]
        evidence = {
            "e1": "First evidence",
            "e2": "Second evidence",
        }

        result = await extractor.map_claims_to_evidence(claims, evidence)

        assert set(result[0].evidence_ids) == {"e1", "e2"}
        assert result[1].evidence_ids == ["e2"]

    @pytest.mark.asyncio
    async def test_map_claims_preserves_existing_evidence(self) -> None:
        """Test that mapping preserves existing evidence IDs."""
        mapping_response = json.dumps({"0": ["e2"]})
        client = MockLLMClient(responses=[mapping_response])
        extractor = ClaimExtractor(client)

        claims = [
            ExtractedClaim(claim_id="c1", claim_text="Claim 1", evidence_ids=["e1"]),
        ]
        evidence = {"e2": "Second evidence"}

        result = await extractor.map_claims_to_evidence(claims, evidence)

        # Should have both original and new evidence
        assert set(result[0].evidence_ids) == {"e1", "e2"}

    @pytest.mark.asyncio
    async def test_map_claims_handles_invalid_response(self) -> None:
        """Test mapping handles invalid response gracefully."""
        client = MockLLMClient(responses=["Not valid JSON"])
        extractor = ClaimExtractor(client)

        claims = [ExtractedClaim(claim_id="c1", claim_text="Claim 1")]
        evidence = {"e1": "Evidence"}

        result = await extractor.map_claims_to_evidence(claims, evidence)

        # Should return claims unchanged
        assert result[0].evidence_ids == []


class TestExtractEvidenceReferences:
    """Tests for extract_evidence_references function."""

    def test_extract_file_paths(self) -> None:
        """Test extraction of file paths."""
        text = "Look at src/utils.py and tests/test_foo.py"
        refs = extract_evidence_references(text)

        assert "src/utils.py" in refs
        assert "tests/test_foo.py" in refs

    def test_extract_line_numbers(self) -> None:
        """Test extraction of line number references."""
        text = "See line 42, L100, and :55"
        refs = extract_evidence_references(text)

        assert "line:42" in refs
        assert "line:100" in refs
        assert "line:55" in refs

    def test_extract_code_references(self) -> None:
        """Test extraction of inline code references."""
        text = "The `calculate_total` function calls `helper()`"
        refs = extract_evidence_references(text)

        assert "calculate_total" in refs
        assert "helper()" in refs

    def test_extract_mixed_references(self) -> None:
        """Test extraction of mixed reference types."""
        text = "In `src/main.py` at line 42, the `process()` function..."
        refs = extract_evidence_references(text)

        assert "src/main.py" in refs
        assert "line:42" in refs
        assert "process()" in refs

    def test_extract_deduplicates(self) -> None:
        """Test that duplicate references are removed."""
        text = "file.py file.py file.py"
        refs = extract_evidence_references(text)

        # Should only appear once
        assert refs.count("file.py") == 1

    def test_extract_empty_text(self) -> None:
        """Test extraction from empty text."""
        refs = extract_evidence_references("")
        assert refs == []

    def test_extract_no_references(self) -> None:
        """Test extraction from text with no references."""
        text = "This is just plain text with no code references"
        refs = extract_evidence_references(text)

        # May extract "text" as potential file extension match
        # but shouldn't have any line references
        assert not any("line:" in r for r in refs)
