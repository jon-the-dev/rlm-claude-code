"""
Epistemic verification module for hallucination detection.

Implements: SPEC-16 Epistemic Verification

This module provides consistency-based verification for Claude API,
detecting procedural hallucinations where the model "knows but doesn't use"
information correctly.

Example:
    >>> from src.epistemic import ClaimVerification, HallucinationReport
    >>> claim = ClaimVerification(
    ...     claim_id="c1",
    ...     claim_text="The file contains 5 functions",
    ...     evidence_support=0.9,
    ...     evidence_dependence=0.8,
    ... )
    >>> claim.combined_score
    0.8485281374238571
"""

from src.epistemic.claim_extractor import (
    ClaimExtractor,
    ExtractedClaim,
    ExtractionResult,
    extract_evidence_references,
)
from src.epistemic.types import (
    ClaimVerification,
    EpistemicGap,
    FlagReason,
    GapType,
    HallucinationReport,
    OnFailureAction,
    VerificationConfig,
    VerificationMode,
)

__all__ = [
    # Core types
    "ClaimVerification",
    "EpistemicGap",
    "HallucinationReport",
    "VerificationConfig",
    # Claim extraction
    "ClaimExtractor",
    "ExtractedClaim",
    "ExtractionResult",
    "extract_evidence_references",
    # Type aliases
    "GapType",
    "FlagReason",
    "OnFailureAction",
    "VerificationMode",
]
