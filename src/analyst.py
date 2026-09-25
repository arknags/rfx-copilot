import json
import re

from src.analyst_schemas import QueryIntent
from src.database import (
    get_comparison_matrix,
    get_discount_evidence,
    get_missing_quotes,
    get_rfx,
    get_review_summary,
    get_vendor_exclusion_reasons,
    get_vendor_qualification,
)
from src.gemini_client import generate_structured


def create_query_intent_prompt(
    *,
    buyer_question: str,
    available_item_ids: list[str],
    available_vendor_names: list[str],
) -> str:
    """Create a constrained prompt for Gemini intent classification."""

    return f"""
You are a procurement analyst assistant.

Your role is only to classify the buyer question into one approved query intent.

You must not:
- write SQL;
- calculate prices;
- invent values;
- select winners;
- make award decisions.

Available item IDs:
{json.dumps(available_item_ids)}

Available vendor names:
{json.dumps(available_vendor_names)}

Buyer question:
{buyer_question}

Choose exactly one intent.

Use:
- vendor_qualification for quality-gate status;
- rfx_line_items when the buyer asks to list, show, or describe all RFx line
  items. Do not require one item ID for this intent;
- line_item_comparison for comparing vendor quotes for an item;
- missing_quotes for missing/non-comparable quote questions;
- review_summary for Needs Review queue questions;
- discount_evidence for rebate/conditional-discount questions;
- total_spend only when the user asks for spending totals;
- vendor_exclusion_reason when the user asks why a vendor is excluded.

Return only valid JSON matching the requested schema.
""".strip()


def _normalise_text(value: str) -> str:
    """Normalise buyer wording for deterministic reference matching."""

    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _resolve_item_id(
    buyer_question: str,
    item_ids: list[str],
) -> str | None:
    """Resolve common item-ID formats such as CS-015 or cs 15."""

    for number in re.findall(r"\bcs[\s_-]*0*(\d{1,3})\b", buyer_question, re.I):
        candidate = f"CS-{int(number):03d}"
        if candidate in item_ids:
            return candidate

    question = _normalise_text(buyer_question)
    return next(
        (
            item_id
            for item_id in item_ids
            if _normalise_text(item_id) in question
        ),
        None,
    )


def _resolve_vendor_name(
    buyer_question: str,
    vendor_names: list[str],
) -> str | None:
    """Resolve short buyer references, for example PrimeBoard to its vendor."""

    question = _normalise_text(buyer_question)
    compact_question = question.replace(" ", "")
    question_tokens = set(question.split())
    ignored_words = {
        "industries",
        "materials",
        "packaging",
        "solutions",
        "works",
        "limited",
        "ltd",
        "private",
    }
    candidates: list[tuple[int, str]] = []

    for vendor_name in vendor_names:
        normalised_vendor = _normalise_text(vendor_name)
        if not normalised_vendor or normalised_vendor == "no quote":
            continue

        if normalised_vendor in question:
            candidates.append((1000 + len(normalised_vendor), vendor_name))
            continue

        distinctive_tokens = [
            token
            for token in normalised_vendor.split()
            if token not in ignored_words
        ]
        compact_alias = "".join(distinctive_tokens)
        matched_tokens = set(distinctive_tokens) & question_tokens

        if compact_alias and compact_alias in compact_question:
            candidates.append((500 + len(compact_alias), vendor_name))
        elif matched_tokens:
            candidates.append((len(matched_tokens), vendor_name))

    if not candidates:
        return None

    candidates.sort(reverse=True)
    best_score, best_vendor = candidates[0]
    tied = [vendor for score, vendor in candidates if score == best_score]
    return best_vendor if len(tied) == 1 else None


def _direct_intent_from_question(buyer_question: str) -> QueryIntent | None:
    """Handle clear, safe questions without relying on model slot filling."""

    question = _normalise_text(buyer_question)

    if (
        ("item" in question or "line item" in question)
        and any(word in question for word in {"list", "show", "what", "all"})
        and ("rfx" in question or "this" in question or "the" in question)
    ):
        return QueryIntent(
            intent="rfx_line_items",
            explanation="The buyer is asking to list the RFx line items.",
        )

    if "exclud" in question and any(
        word in question for word in {"why", "reason"}
    ):
        return QueryIntent(
            intent="vendor_exclusion_reason",
            explanation=(
                "The buyer is asking why a vendor is excluded for a line item."
            ),
        )

    return None


def _enrich_intent_references(
    *,
    intent: QueryIntent,
    buyer_question: str,
    item_ids: list[str],
    vendor_names: list[str],
) -> QueryIntent:
    """Use the approved RFx values to fill and validate model references."""

    resolved_item_id = _resolve_item_id(buyer_question, item_ids)
    if resolved_item_id is None and intent.item_id in item_ids:
        resolved_item_id = intent.item_id

    resolved_vendor_name = _resolve_vendor_name(
        " ".join(
            part
            for part in [buyer_question, intent.vendor_name or ""]
            if part
        ),
        vendor_names,
    )
    if resolved_vendor_name is None and intent.vendor_name in vendor_names:
        resolved_vendor_name = intent.vendor_name

    return intent.model_copy(
        update={
            "item_id": resolved_item_id,
            "vendor_name": resolved_vendor_name,
        }
    )


def classify_buyer_question(
    *,
    buyer_question: str,
    item_ids: list[str],
    vendor_names: list[str],
    api_key: str,
    model: str,
) -> QueryIntent:
    """Classify the question, then resolve references from approved RFx data."""

    direct_intent = _direct_intent_from_question(buyer_question)
    if direct_intent is not None:
        return _enrich_intent_references(
            intent=direct_intent,
            buyer_question=buyer_question,
            item_ids=item_ids,
            vendor_names=vendor_names,
        )

    prompt = create_query_intent_prompt(
        buyer_question=buyer_question,
        available_item_ids=item_ids,
        available_vendor_names=vendor_names,
    )

    intent = generate_structured(
        api_key=api_key,
        model=model,
        prompt=prompt,
        response_model=QueryIntent,
    )

    return _enrich_intent_references(
        intent=intent,
        buyer_question=buyer_question,
        item_ids=item_ids,
        vendor_names=vendor_names,
    )


def execute_query_intent(
    *,
    rfx_id: str,
    intent: QueryIntent,
) -> dict:
    """
    Execute only approved deterministic Python/database functions.
    """

    if intent.intent == "rfx_line_items":
        rfx = get_rfx(rfx_id)
        if rfx is None:
            return {
                "title": "RFx line items",
                "error": "The selected approved RFx could not be found.",
            }

        return {
            "title": f"RFx line items: {rfx['title']}",
            "data": rfx["line_items"],
        }

    if intent.intent == "vendor_qualification":
        return {
            "title": "Vendor qualification",
            "data": get_vendor_qualification(
                rfx_id=rfx_id,
                vendor_name=intent.vendor_name,
            ),
        }

    if intent.intent == "line_item_comparison":
        if not intent.item_id:
            return {
                "title": "Line-item comparison",
                "error": "Please specify a buyer item ID, such as CS-015.",
            }

        rows = [
            row
            for row in get_comparison_matrix(rfx_id)
            if row["item_id"] == intent.item_id
        ]

        if intent.qualified_only:
            qualifications = get_vendor_qualification(rfx_id=rfx_id)
            qualified_vendors = {
                qualification["vendor_name"]
                for qualification in qualifications
                if qualification["qualified"]
            }

            rows = [
                row
                for row in rows
                if row["vendor_name"] in qualified_vendors
            ]

        return {
            "title": f"Line-item comparison: {intent.item_id}",
            "data": rows,
        }

    if intent.intent == "missing_quotes":
        return {
            "title": "Missing or non-comparable quotes",
            "data": get_missing_quotes(
                rfx_id=rfx_id,
                vendor_name=intent.vendor_name,
            ),
        }

    if intent.intent == "review_summary":
        return {
            "title": "Needs Review summary",
            "data": get_review_summary(rfx_id),
        }

    if intent.intent == "discount_evidence":
        return {
            "title": "Conditional discount evidence",
            "data": get_discount_evidence(
                rfx_id=rfx_id,
                vendor_name=intent.vendor_name,
            ),
        }

    if intent.intent == "vendor_exclusion_reason":
        if not intent.item_id or not intent.vendor_name:
            return {
                "title": "Vendor exclusion reason",
                "error": (
                    "Please specify both a vendor name and buyer item ID. "
                    "Example: Why is PrimeBoard excluded for CS-015?"
                ),
            }

        return {
            "title": "Vendor exclusion reason",
            "data": {
                "vendor_name": intent.vendor_name,
                "item_id": intent.item_id,
                "reasons": get_vendor_exclusion_reasons(
                    rfx_id=rfx_id,
                    item_id=intent.item_id,
                    vendor_name=intent.vendor_name,
                ),
            },
        }

    if intent.intent == "total_spend":
        return {
            "title": "Total spend",
            "error": (
                "Total award spend will be calculated in Step 23 after "
                "award-eligibility and tie rules are implemented."
            ),
        }

    return {
        "title": "Unsupported request",
        "error": "The selected intent is not implemented.",
    }
