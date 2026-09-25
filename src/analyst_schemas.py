from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class QueryIntent(BaseModel):
    """
    Gemini may only select one approved query intent.
    It cannot write SQL or calculate an award itself.
    """

    model_config = ConfigDict(extra="forbid")

    intent: Literal[
        "vendor_qualification",
        "rfx_line_items",
        "line_item_comparison",
        "missing_quotes",
        "review_summary",
        "discount_evidence",
        "total_spend",
        "vendor_exclusion_reason",
    ]

    item_id: str | None = None
    vendor_name: str | None = None
    qualified_only: bool = False

    explanation: str = Field(
        description=(
            "Brief explanation of why this approved query intent "
            "matches the buyer question."
        )
    )
