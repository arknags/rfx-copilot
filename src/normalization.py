import re
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from difflib import SequenceMatcher
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


# -------------------------------------------------------------------
# Demo FX configuration
# -------------------------------------------------------------------
DEMO_USD_TO_INR_RATE = Decimal("83.25")
DEMO_FX_RATE_DATE = "Demo assumption"


# -------------------------------------------------------------------
# Pydantic models
# -------------------------------------------------------------------
class VendorItemAttributes(BaseModel):
    """
    Optional vendor details used for deterministic technical matching.

    These may be available later from structured extraction or buyer review.
    """

    model_config = ConfigDict(extra="forbid")

    ply: int | None = None
    flute: str | None = None
    outer_gsm: int | None = None
    dimensions_mm: str | None = None


class RawQuoteInput(BaseModel):
    """
    Input required to normalize one extracted vendor quote.

    Do not overwrite the original extracted values in the database.
    """

    model_config = ConfigDict(extra="forbid")

    canonical_item_id: str | None = None
    raw_description: str
    raw_price: Decimal | None = None
    raw_unit: str | None = None
    currency: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    freight_status: Literal[
        "included",
        "extra",
        "unknown",
        "not_stated",
    ] = "not_stated"
    vendor_attributes: VendorItemAttributes | None = None


class ItemMatchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    matched_item_id: str | None = None
    method: Literal[
        "exact_item_id",
        "exact_technical_specification",
        "strong_description_match",
        "ai_suggested_match",
        "no_match",
    ]
    confidence: float = Field(ge=0.0, le=1.0)
    review_required: bool
    reason: str


class CurrencyNormalization(BaseModel):
    model_config = ConfigDict(extra="forbid")

    original_currency: str | None
    original_price: Decimal | None
    inr_price: Decimal | None
    fx_rate: Decimal | None
    fx_rate_date: str | None
    formula: str | None
    supported: bool
    reason: str | None = None


class UnitNormalization(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_unit: str | None
    sheets_per_quoted_unit: Decimal | None
    supported: bool
    formula: str | None = None
    reason: str | None = None


class NormalizedQuote(BaseModel):
    """
    A fully auditable normalization result.

    comparable=True means the price is technically comparable as INR/sheet.
    award_eligible=False may still occur when freight is extra or review is pending.
    """

    model_config = ConfigDict(extra="forbid")

    matched_item_id: str | None
    item_match_method: str
    raw_description: str
    raw_price: Decimal | None
    raw_unit: str | None
    raw_currency: str | None

    inr_price_before_unit_conversion: Decimal | None
    normalized_inr_per_sheet: Decimal | None

    currency_formula: str | None
    unit_formula: str | None
    normalization_formula: str | None

    comparable: bool
    review_required: bool
    award_eligible: bool
    reasons: list[str]


# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------
def _decimal(value: Decimal | float | int | str | None) -> Decimal | None:
    """Safely convert a numeric input to Decimal."""

    if value is None:
        return None

    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def _money(value: Decimal) -> Decimal:
    """Round money values consistently for display and storage."""

    return value.quantize(
        Decimal("0.0001"),
        rounding=ROUND_HALF_UP,
    )


def _normalise_text(value: str | None) -> str:
    """Normalize text for deterministic matching."""

    if not value:
        return ""

    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _normalise_dimensions(value: str | None) -> str:
    """Normalize values such as 600x400, 600 x 400, and 600 X 400."""

    if not value:
        return ""

    return re.sub(
        r"\s+",
        "",
        value.lower().replace("×", "x"),
    )


def _item_ids_from_description(
    raw_description: str,
    canonical_item_ids: set[str],
) -> list[str]:
    """Find canonical item IDs explicitly stated in vendor text."""

    upper_description = raw_description.upper()
    found = []

    for item_id in canonical_item_ids:
        if item_id.upper() in upper_description:
            found.append(item_id)

    return found


# -------------------------------------------------------------------
# A. Currency normalization
# -------------------------------------------------------------------
def normalize_currency_to_inr(
    *,
    raw_price: Decimal | float | int | str | None,
    currency: str | None,
    usd_to_inr_rate: Decimal = DEMO_USD_TO_INR_RATE,
    fx_rate_date: str = DEMO_FX_RATE_DATE,
) -> CurrencyNormalization:
    """
    Convert an extracted raw price into INR.

    Supported demo currencies:
      INR
      USD
    """

    price = _decimal(raw_price)

    if price is None:
        return CurrencyNormalization(
            original_currency=currency,
            original_price=None,
            inr_price=None,
            fx_rate=None,
            fx_rate_date=None,
            formula=None,
            supported=False,
            reason="Raw price is missing or invalid.",
        )

    normalized_currency = (currency or "").strip().upper()

    if normalized_currency == "INR":
        return CurrencyNormalization(
            original_currency="INR",
            original_price=price,
            inr_price=_money(price),
            fx_rate=Decimal("1"),
            fx_rate_date=fx_rate_date,
            formula=f"INR {price} × 1",
            supported=True,
        )

    if normalized_currency == "USD":
        inr_price = _money(price * usd_to_inr_rate)

        return CurrencyNormalization(
            original_currency="USD",
            original_price=price,
            inr_price=inr_price,
            fx_rate=usd_to_inr_rate,
            fx_rate_date=fx_rate_date,
            formula=f"USD {price} × {usd_to_inr_rate}",
            supported=True,
        )

    return CurrencyNormalization(
        original_currency=normalized_currency or None,
        original_price=price,
        inr_price=None,
        fx_rate=None,
        fx_rate_date=None,
        formula=None,
        supported=False,
        reason=(
            f"Unsupported currency '{normalized_currency or 'missing'}'. "
            "Only INR and USD are configured for this demo."
        ),
    )


# -------------------------------------------------------------------
# B. Unit normalization
# -------------------------------------------------------------------
def parse_sheets_per_quoted_unit(raw_unit: str | None) -> UnitNormalization:
    """
    Determine how many sheets are represented by the quoted unit.

    Supported:
      per sheet
      per 100 sheets
      per bundle of N sheets
      per gross
    """

    if not raw_unit or not raw_unit.strip():
        return UnitNormalization(
            raw_unit=raw_unit,
            sheets_per_quoted_unit=None,
            supported=False,
            reason="Quote unit is missing.",
        )

    normalized_unit = _normalise_text(raw_unit)

    # Explicit non-comparable units.
    if any(
        marker in normalized_unit
        for marker in ["kg", "kilogram", "kilo", "ton", "tonne"]
    ):
        return UnitNormalization(
            raw_unit=raw_unit,
            sheets_per_quoted_unit=None,
            supported=False,
            reason=(
                "Weight-to-sheet conversion basis unavailable. "
                "Do not convert INR/kg to INR/sheet without traceable sheet weight."
            ),
        )

    # Per sheet / each sheet.
    if normalized_unit in {
        "per sheet",
        "sheet",
        "per each",
        "each",
        "per piece",
        "per pcs",
        "per pc",
    }:
        return UnitNormalization(
            raw_unit=raw_unit,
            sheets_per_quoted_unit=Decimal("1"),
            supported=True,
            formula="Quoted unit equals 1 sheet",
        )

    # Per 100 sheets, per 200 sheets, etc.
    sheets_match = re.search(
        r"(?:per\s+)?(\d+)\s+(?:sheet|sheets|piece|pieces|pcs)",
        normalized_unit,
    )

    if sheets_match:
        number_of_sheets = Decimal(sheets_match.group(1))

        if number_of_sheets <= 0:
            return UnitNormalization(
                raw_unit=raw_unit,
                sheets_per_quoted_unit=None,
                supported=False,
                reason="Quoted sheet count must be greater than zero.",
            )

        return UnitNormalization(
            raw_unit=raw_unit,
            sheets_per_quoted_unit=number_of_sheets,
            supported=True,
            formula=f"Quoted unit equals {number_of_sheets} sheets",
        )

    # Per bundle of 100 sheets / bundle 100 sheets.
    bundle_match = re.search(
        r"bundle(?:\s+of)?\s+(\d+)\s+(?:sheet|sheets|piece|pieces|pcs)",
        normalized_unit,
    )

    if bundle_match:
        number_of_sheets = Decimal(bundle_match.group(1))

        if number_of_sheets <= 0:
            return UnitNormalization(
                raw_unit=raw_unit,
                sheets_per_quoted_unit=None,
                supported=False,
                reason="Bundle sheet count must be greater than zero.",
            )

        return UnitNormalization(
            raw_unit=raw_unit,
            sheets_per_quoted_unit=number_of_sheets,
            supported=True,
            formula=f"Bundle contains {number_of_sheets} sheets",
        )

    # One gross equals 144 sheets.
    if "gross" in normalized_unit:
        return UnitNormalization(
            raw_unit=raw_unit,
            sheets_per_quoted_unit=Decimal("144"),
            supported=True,
            formula="1 gross equals 144 sheets",
        )

    # A plain "per bundle" is intentionally not assumed to mean 100 sheets.
    if "bundle" in normalized_unit:
        return UnitNormalization(
            raw_unit=raw_unit,
            sheets_per_quoted_unit=None,
            supported=False,
            reason=(
                "Bundle size is not stated. "
                "Do not assume a bundle contains 100 sheets."
            ),
        )

    return UnitNormalization(
        raw_unit=raw_unit,
        sheets_per_quoted_unit=None,
        supported=False,
        reason=f"Unsupported quote unit: '{raw_unit}'.",
    )


# -------------------------------------------------------------------
# D. Item matching
# -------------------------------------------------------------------
def match_to_canonical_item(
    *,
    raw_description: str,
    canonical_item_id: str | None,
    canonical_line_items: list[dict],
    vendor_attributes: VendorItemAttributes | None = None,
    ai_confidence: float = 0.0,
) -> ItemMatchResult:
    """
    Match a vendor quote to a canonical buyer item.

    Matching order:
      1. Exact buyer item ID
      2. Exact dimensions + ply + flute + GSM
      3. Strong description match
      4. AI-suggested item ID with visible review requirement
      5. No match
    """

    item_by_id = {
        item["item_id"]: item
        for item in canonical_line_items
    }

    canonical_item_ids = set(item_by_id)

    # 1. Exact item ID supplied by extraction.
    if canonical_item_id and canonical_item_id in item_by_id:
        return ItemMatchResult(
            matched_item_id=canonical_item_id,
            method="exact_item_id",
            confidence=1.0,
            review_required=False,
            reason="Exact canonical buyer item ID was identified.",
        )

    # 1b. Exact item ID visibly present in the vendor's raw description.
    IDs_in_description = _item_ids_from_description(
        raw_description,
        canonical_item_ids,
    )

    if len(IDs_in_description) == 1:
        return ItemMatchResult(
            matched_item_id=IDs_in_description[0],
            method="exact_item_id",
            confidence=1.0,
            review_required=False,
            reason="Exact buyer item ID was found in the vendor description.",
        )

    # 2. Exact technical specification match.
    if vendor_attributes:
        technical_matches = []

        for item in canonical_line_items:
            if (
                vendor_attributes.ply == item.get("ply")
                and _normalise_text(vendor_attributes.flute)
                == _normalise_text(item.get("flute"))
                and vendor_attributes.outer_gsm == item.get("outer_gsm")
                and _normalise_dimensions(vendor_attributes.dimensions_mm)
                == _normalise_dimensions(item.get("dimensions_mm"))
            ):
                technical_matches.append(item["item_id"])

        if len(technical_matches) == 1:
            return ItemMatchResult(
                matched_item_id=technical_matches[0],
                method="exact_technical_specification",
                confidence=1.0,
                review_required=False,
                reason=(
                    "Ply, flute, GSM, and dimensions exactly match one buyer item."
                ),
            )

    # 3. Strong description match.
    normalized_vendor_description = _normalise_text(raw_description)
    scored_matches = []

    for item in canonical_line_items:
        canonical_description = _normalise_text(item.get("description"))

        score = SequenceMatcher(
            None,
            normalized_vendor_description,
            canonical_description,
        ).ratio()

        scored_matches.append((score, item["item_id"]))

    scored_matches.sort(reverse=True)

    if scored_matches:
        best_score, best_item_id = scored_matches[0]
        second_score = scored_matches[1][0] if len(scored_matches) > 1 else 0.0

        # Strong and clearly distinct description match.
        if best_score >= 0.93 and (best_score - second_score) >= 0.08:
            return ItemMatchResult(
                matched_item_id=best_item_id,
                method="strong_description_match",
                confidence=round(best_score, 2),
                review_required=False,
                reason="Vendor description strongly matches one buyer description.",
            )

    # 4. AI-suggested item ID. Keep it visible for buyer review.
    if canonical_item_id and canonical_item_id in item_by_id:
        return ItemMatchResult(
            matched_item_id=canonical_item_id,
            method="ai_suggested_match",
            confidence=max(0.0, min(ai_confidence, 1.0)),
            review_required=True,
            reason=(
                "Gemini suggested an item match, but it requires buyer review "
                "because no deterministic exact match was established."
            ),
        )

    # 5. No match.
    return ItemMatchResult(
        matched_item_id=None,
        method="no_match",
        confidence=0.0,
        review_required=True,
        reason=(
            "No confident canonical item match was found. "
            "Buyer review is required."
        ),
    )


# -------------------------------------------------------------------
# Full quote normalization
# -------------------------------------------------------------------
def normalize_quote(
    *,
    quote: RawQuoteInput,
    canonical_line_items: list[dict],
    usd_to_inr_rate: Decimal = DEMO_USD_TO_INR_RATE,
    fx_rate_date: str = DEMO_FX_RATE_DATE,
) -> NormalizedQuote:
    """
    Normalize one extracted quote into INR per sheet.

    This function is deterministic. It never calls Gemini and never writes SQL.
    """

    reasons: list[str] = []

    item_match = match_to_canonical_item(
        raw_description=quote.raw_description,
        canonical_item_id=quote.canonical_item_id,
        canonical_line_items=canonical_line_items,
        vendor_attributes=quote.vendor_attributes,
        ai_confidence=quote.confidence,
    )

    currency_result = normalize_currency_to_inr(
        raw_price=quote.raw_price,
        currency=quote.currency,
        usd_to_inr_rate=usd_to_inr_rate,
        fx_rate_date=fx_rate_date,
    )

    unit_result = parse_sheets_per_quoted_unit(quote.raw_unit)

    if item_match.review_required:
        reasons.append(item_match.reason)

    if not currency_result.supported:
        reasons.append(currency_result.reason or "Currency conversion failed.")

    if not unit_result.supported:
        reasons.append(unit_result.reason or "Unit conversion failed.")

    if quote.confidence < 0.80:
        reasons.append(
            f"Extraction confidence is {quote.confidence:.2f}, below the 0.80 review threshold."
        )

    if quote.freight_status == "extra":
        reasons.append(
            "Freight is extra. The quote is not eligible for lowest-landed-price award."
        )

    if quote.freight_status in {"unknown", "not_stated"}:
        reasons.append(
            "Freight treatment is unknown or not stated."
        )

    comparable = (
        item_match.matched_item_id is not None
        and currency_result.supported
        and unit_result.supported
        and currency_result.inr_price is not None
        and unit_result.sheets_per_quoted_unit is not None
    )

    normalized_inr_per_sheet = None
    normalization_formula = None

    if comparable:
        normalized_inr_per_sheet = _money(
            currency_result.inr_price
            / unit_result.sheets_per_quoted_unit
        )

        normalization_formula = (
            f"{currency_result.formula} ÷ "
            f"{unit_result.sheets_per_quoted_unit} sheets"
        )

    review_required = len(reasons) > 0

    award_eligible = (
        comparable
        and not review_required
        and quote.freight_status == "included"
    )

    return NormalizedQuote(
        matched_item_id=item_match.matched_item_id,
        item_match_method=item_match.method,
        raw_description=quote.raw_description,
        raw_price=_decimal(quote.raw_price),
        raw_unit=quote.raw_unit,
        raw_currency=(quote.currency or "").upper() or None,
        inr_price_before_unit_conversion=currency_result.inr_price,
        normalized_inr_per_sheet=normalized_inr_per_sheet,
        currency_formula=currency_result.formula,
        unit_formula=unit_result.formula,
        normalization_formula=normalization_formula,
        comparable=comparable,
        review_required=review_required,
        award_eligible=award_eligible,
        reasons=reasons,
    )


# -------------------------------------------------------------------
# Simple deterministic calculation helpers
# -------------------------------------------------------------------
def calculate_annual_line_cost(
    *,
    normalized_inr_per_sheet: Decimal | float | int | str,
    annual_quantity_sheets: int,
) -> Decimal:
    """Calculate annual cost for one RFx line item."""

    unit_price = _decimal(normalized_inr_per_sheet)

    if unit_price is None:
        raise ValueError("Normalized INR-per-sheet price is required.")

    if annual_quantity_sheets < 0:
        raise ValueError("Annual quantity cannot be negative.")

    return _money(unit_price * Decimal(annual_quantity_sheets))