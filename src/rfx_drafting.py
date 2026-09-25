import json
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.gemini_client import GeminiClientError, generate_structured


class CanonicalLineItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str
    description: str
    ply: int = Field(ge=1)
    flute: str
    outer_gsm: int = Field(ge=0)
    dimensions_mm: str
    print_requirement: str | None = None
    moisture_resistant: Literal["Yes", "No"]
    annual_quantity_sheets: int = Field(ge=0)
    standard_unit: str
    delivery_location: str


class QuestionnaireQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: str
    question: str
    answer_type: Literal["yes_no", "number", "text"]
    required: bool
    quality_gate: bool


class GeneratedRFxDraft(BaseModel):
    """
    A Gemini proposal. It is not an approved RFx.
    """

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=5)
    scope: str = Field(min_length=20)
    line_items: list[CanonicalLineItem] = Field(min_length=1)
    questionnaire: list[QuestionnaireQuestion] = Field(min_length=1)
    commercial_terms: str = Field(min_length=20)
    assumptions: list[str] = Field(min_length=1)

    @field_validator("assumptions")
    @classmethod
    def validate_assumptions(cls, assumptions: list[str]) -> list[str]:
        cleaned = [assumption.strip() for assumption in assumptions if assumption.strip()]

        if not cleaned:
            raise ValueError("At least one assumption is required.")

        return cleaned


class AssumptionDecision(BaseModel):
    assumption_text: str
    accepted: bool


class RFxSourceContext(BaseModel):
    line_items: list[CanonicalLineItem]
    questionnaire: list[QuestionnaireQuestion]
    commercial_terms: str


def _clean_value(value):
    """Convert Pandas NaN values to Python None."""

    return None if pd.isna(value) else value


def load_rfx_source_context(data_dir: Path) -> RFxSourceContext:
    """
    Load canonical source files from the project's data folder.

    Required:
      data/line_items.csv
      data/questionnaire.json
      data/terms.md
    """

    line_items_file = data_dir / "line_items.csv"
    questionnaire_file = data_dir / "questionnaire.json"
    terms_file = data_dir / "terms.md"

    missing_files = [
        file_path.name
        for file_path in [line_items_file, questionnaire_file, terms_file]
        if not file_path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Missing source file(s): " + ", ".join(missing_files)
        )

    line_items_df = pd.read_csv(line_items_file)

    line_item_records = [
        {column: _clean_value(value) for column, value in row.items()}
        for row in line_items_df.to_dict(orient="records")
    ]

    with open(questionnaire_file, "r", encoding="utf-8") as file:
        questionnaire_records = json.load(file)

    if not isinstance(questionnaire_records, list):
        raise ValueError("questionnaire.json must contain a JSON list.")

    commercial_terms = terms_file.read_text(encoding="utf-8").strip()

    if not commercial_terms:
        raise ValueError("terms.md cannot be empty.")

    return RFxSourceContext(
        line_items=line_item_records,
        questionnaire=questionnaire_records,
        commercial_terms=commercial_terms,
    )


def create_rfx_prompt(
    buyer_request: str,
    canonical_line_items: list[dict],
    canonical_questions: list[dict],
    commercial_terms_template: str,
) -> str:
    """Create the constrained Gemini prompt."""

    return f"""
You are drafting an RFx for corrugated sheets.
Return a proposed draft only.
Do not claim that the buyer has approved it.
Do not assign an RFx ID.
State all assumptions explicitly.
Use INR per sheet, excluding GST, as the intended comparison basis.

The buyer request is:

{buyer_request.strip()}

The following canonical line items are the source of truth.

Rules:
- Select only the canonical items that are in scope for the buyer request.
- If the buyer explicitly asks to exclude, ignore, omit, or remove an item or
  product group, do not return those canonical line items.
- Return each selected item exactly once. At least one line item is required.
- Preserve each item_id exactly.
- Do not invent or rename item IDs. You may omit canonical item IDs only when
  the buyer request clearly makes them out of scope.
- Preserve the technical fields and annual quantities.
- Improve the RFx title and scope when useful.

Canonical line items:
{json.dumps(canonical_line_items, indent=2, ensure_ascii=False)}

The following questionnaire is the source of truth.

Rules:
- Return every listed question exactly once.
- Preserve question_id exactly.
- Q01, Q02, Q03, and Q04 must remain quality-gate questions.
- Do not change the required flag.

Canonical questionnaire:
{json.dumps(canonical_questions, indent=2, ensure_ascii=False)}

Use the following commercial terms as the starting template. You may improve
clarity, but preserve all material buyer requirements:

{commercial_terms_template}

Return only JSON matching the requested schema.
""".strip()


def validate_draft_against_source(
    draft: GeneratedRFxDraft,
    source_context: RFxSourceContext,
) -> None:
    """
    Apply deterministic validation after Gemini's schema validation.
    """

    expected_item_ids = {item.item_id for item in source_context.line_items}
    generated_item_ids = [item.item_id for item in draft.line_items]

    expected_question_ids = {
        question.question_id for question in source_context.questionnaire
    }
    generated_question_ids = [question.question_id for question in draft.questionnaire]

    if len(generated_item_ids) != len(set(generated_item_ids)):
        raise ValueError("Gemini returned duplicate line-item IDs.")

    if len(generated_question_ids) != len(set(generated_question_ids)):
        raise ValueError("Gemini returned duplicate questionnaire IDs.")

    unexpected_item_ids = sorted(
        set(generated_item_ids) - expected_item_ids
    )
    if unexpected_item_ids:
        raise ValueError(
            "Generated line items must come from the canonical source. "
            f"Unexpected: {unexpected_item_ids}."
        )

    if set(generated_question_ids) != expected_question_ids:
        missing = sorted(expected_question_ids - set(generated_question_ids))
        unexpected = sorted(set(generated_question_ids) - expected_question_ids)

        raise ValueError(
            f"Generated question IDs do not match canonical source. "
            f"Missing: {missing or 'None'}; unexpected: {unexpected or 'None'}."
        )

    source_questions = {
        question.question_id: question
        for question in source_context.questionnaire
    }

    for question in draft.questionnaire:
        source_question = source_questions[question.question_id]

        if question.required != source_question.required:
            raise ValueError(
                f"{question.question_id} changed its required status."
            )

        if question.question_id in {"Q01", "Q02", "Q03", "Q04"}:
            if not question.quality_gate:
                raise ValueError(
                    f"{question.question_id} must remain a quality-gate question."
                )


def generate_rfx_draft(
    *,
    buyer_request: str,
    data_dir: Path,
    api_key: str,
    model: str,
) -> GeneratedRFxDraft:
    """
    Generate a Pydantic-validated RFx proposal.

    This function does not approve or save the RFx.
    """

    if not buyer_request.strip():
        raise ValueError("Buyer request cannot be empty.")

    source_context = load_rfx_source_context(data_dir)

    prompt = create_rfx_prompt(
        buyer_request=buyer_request,
        canonical_line_items=[
            item.model_dump() for item in source_context.line_items
        ],
        canonical_questions=[
            question.model_dump() for question in source_context.questionnaire
        ],
        commercial_terms_template=source_context.commercial_terms,
    )

    try:
        draft = generate_structured(
            api_key=api_key,
            model=model,
            prompt=prompt,
            response_model=GeneratedRFxDraft,
        )
    except GeminiClientError:
        raise

    validate_draft_against_source(draft, source_context)

    return draft


def draft_to_streamlit_state(draft: GeneratedRFxDraft) -> dict:
    """Convert Pydantic objects into Streamlit-editable values."""

    return {
        "title": draft.title,
        "scope": draft.scope,
        "line_items": pd.DataFrame(
            [item.model_dump() for item in draft.line_items]
        ),
        "questionnaire": pd.DataFrame(
            [question.model_dump() for question in draft.questionnaire]
        ),
        "commercial_terms": draft.commercial_terms,
        "assumptions": pd.DataFrame(
            [
                {
                    "assumption_text": assumption,
                    "accepted": True,
                }
                for assumption in draft.assumptions
            ]
        ),
    }
