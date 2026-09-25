from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ExtractedQuote(BaseModel):
    """
    A raw quote extracted from a vendor document.

    Do not normalize prices, currency, or units in this model.
    That belongs to Step 19.
    """

    model_config = ConfigDict(extra="forbid")

    canonical_item_id: str | None = Field(
        default=None,
        description="Canonical buyer item ID only if confidently identifiable.",
    )
    raw_description: str = Field(
        description="Description exactly or closely as stated by the vendor."
    )
    raw_price: float | None = Field(
        default=None,
        description="Raw numeric price stated by the vendor.",
    )
    raw_unit: str | None = Field(
        default=None,
        description="Raw price unit, for example 'per sheet' or 'per bundle of 100'.",
    )
    currency: Literal["INR", "USD", "EUR", "GBP"] | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_text: str = Field(
        min_length=1,
        description="Source wording supporting this extracted value.",
    )
    page_or_section: str | None = Field(
        default=None,
        description="Page, sheet, row, paragraph, or section reference.",
    )
    extraction_note: str | None = None


class ExtractedQuestionnaireAnswer(BaseModel):
    """A raw vendor answer extracted from the response document."""

    model_config = ConfigDict(extra="forbid")

    question_id: str | None = Field(
        default=None,
        description="Canonical buyer question ID only when confidently identified.",
    )
    raw_answer: str = Field(
        min_length=1,
        description="Answer as stated by the vendor.",
    )
    normalized_answer: str | None = Field(
        default=None,
        description="Simple normalized value, such as Yes, No, 7, or Net 45 days.",
    )
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_text: str = Field(min_length=1)
    page_or_section: str | None = None


class VendorDocumentExtraction(BaseModel):
    """
    Complete structured interpretation of one vendor response document.
    """

    model_config = ConfigDict(extra="forbid")

    vendor_name: str | None = Field(
        default=None,
        max_length=255,
        description=(
            "Supplier business name only if it is stated in the source. "
            "Never include document text, rows, reasoning, or JSON."
        ),
    )
    quotes: list[ExtractedQuote] = Field(default_factory=list)
    questionnaire_answers: list[ExtractedQuestionnaireAnswer] = Field(
        default_factory=list
    )
    freight_status: Literal[
        "included",
        "extra",
        "unknown",
        "not_stated",
    ] = "not_stated"
    conditional_terms: list[str] = Field(default_factory=list)
    general_notes: list[str] = Field(default_factory=list)

    @field_validator("conditional_terms", "general_notes")
    @classmethod
    def remove_blank_text(cls, values: list[str]) -> list[str]:
        return [value.strip() for value in values if value.strip()]

    @field_validator("vendor_name")
    @classmethod
    def validate_vendor_name(cls, value: str | None) -> str | None:
        """Reject document text or model reasoning mistakenly placed as a name."""

        if value is None:
            return None

        if "\n" in value or "\r" in value:
            raise ValueError("vendor_name must be a single-line supplier name.")

        cleaned = " ".join(value.split())

        if not cleaned:
            return None

        forbidden_markers = [
            "a1=",
            ".json",
            "quote_extract",
            "->",
            "→",
            "return only json",
            "let's construct",
            "```",
            "{\"",
        ]

        if any(marker in cleaned.lower() for marker in forbidden_markers):
            raise ValueError("vendor_name contains document text or model output.")

        return cleaned


class ReviewCandidate(BaseModel):
    """
    A possible review task. database.py will save these in Step 18 integration.
    """

    model_config = ConfigDict(extra="forbid")

    issue_type: Literal[
        "LOW_CONFIDENCE_EXTRACTION",
        "AMBIGUOUS_ITEM_MATCH",
        "MISSING_LINE_QUOTE",
        "MISSING_QUESTIONNAIRE_ANSWER",
        "FREIGHT_NOT_INCLUDED",
        "CONDITIONAL_DISCOUNT",
        "AMBIGUOUS_COMMERCIAL_TERM",
    ]

    severity: Literal["Amber", "Red"]
    affected_item_id: str | None = None
    evidence_text: str
    current_value: str | None = None
    suggested_resolution: str
