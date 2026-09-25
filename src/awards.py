from decimal import Decimal

from src.database import (
    get_comparison_matrix,
    get_vendor_qualification,
)
from src.normalization import calculate_annual_line_cost


def _decimal(value) -> Decimal | None:
    if value is None:
        return None

    return Decimal(str(value))


def calculate_award_recommendation(rfx_id: str) -> dict:
    """
    Calculate the lowest eligible normalized price for every RFx line item.

    This does not write to the database. It returns deterministic results
    that database.py will save as an award run.
    """

    matrix_rows = get_comparison_matrix(rfx_id)
    qualification_results = get_vendor_qualification(rfx_id=rfx_id)

    qualification_by_vendor = {
        result["vendor_name"]: result
        for result in qualification_results
    }

    line_items: dict[str, list[dict]] = {}

    for row in matrix_rows:
        line_items.setdefault(row["item_id"], []).append(row)

    decisions = []
    total_recommended_spend = Decimal("0")

    for item_id, quotes in line_items.items():
        first_quote = quotes[0]

        item_decision = {
            "item_id": item_id,
            "description": first_quote["description"],
            "annual_quantity_sheets": first_quote["annual_quantity_sheets"],
            "winner_vendor_name": None,
            "selected_price_inr_per_sheet": None,
            "annual_line_cost": None,
            "decision_status": "unawarded",
            "rationale": None,
            "excluded_vendors": [],
        }

        eligible_quotes = []

        for quote in quotes:
            vendor_name = quote["vendor_name"]

            if vendor_name == "No quote":
                continue

            exclusion_reasons = []
            qualification = qualification_by_vendor.get(vendor_name)

            if qualification is None:
                exclusion_reasons.append(
                    "No quality qualification record was found."
                )

            elif not qualification["qualified"]:
                failed = qualification["failed_questions"]
                missing = qualification["missing_questions"]

                if failed:
                    exclusion_reasons.append(
                        "Failed quality gate: " + ", ".join(failed)
                    )

                if missing:
                    exclusion_reasons.append(
                        "Missing quality-gate answer: " + ", ".join(missing)
                    )

            if quote["normalized_inr_per_sheet"] is None:
                exclusion_reasons.append(
                    "No normalized INR-per-sheet price is available."
                )

            if not quote["comparable"]:
                exclusion_reasons.append(
                    "Quote is not comparable to INR per sheet."
                )

            if quote["review_required"]:
                exclusion_reasons.append(
                    "Quote has unresolved review requirements."
                )

            if quote["freight_status"] != "included":
                exclusion_reasons.append(
                    "Freight treatment is not confirmed as included."
                )

            if not quote["award_eligible"]:
                exclusion_reasons.append(
                    "Quote is not currently award eligible."
                )

            if exclusion_reasons:
                item_decision["excluded_vendors"].append(
                    {
                        "vendor_name": vendor_name,
                        "reasons": list(dict.fromkeys(exclusion_reasons)),
                    }
                )
                continue

            eligible_quotes.append(quote)

        if not eligible_quotes:
            item_decision["rationale"] = (
                "No vendor has an eligible, normalized, review-approved quote."
            )
            decisions.append(item_decision)
            continue

        eligible_quotes.sort(
            key=lambda quote: Decimal(
                str(quote["normalized_inr_per_sheet"])
            )
        )

        lowest_price = Decimal(
            str(eligible_quotes[0]["normalized_inr_per_sheet"])
        )

        tied_quotes = [
            quote
            for quote in eligible_quotes
            if Decimal(str(quote["normalized_inr_per_sheet"])) == lowest_price
        ]

        if len(tied_quotes) > 1:
            item_decision["decision_status"] = "tie_requires_buyer_decision"
            item_decision["rationale"] = (
                "Two or more eligible vendors have the same lowest "
                "normalized INR-per-sheet price."
            )

            item_decision["tied_vendors"] = [
                quote["vendor_name"]
                for quote in tied_quotes
            ]

            decisions.append(item_decision)
            continue

        winner = eligible_quotes[0]

        annual_line_cost = calculate_annual_line_cost(
            normalized_inr_per_sheet=lowest_price,
            annual_quantity_sheets=first_quote["annual_quantity_sheets"],
        )

        item_decision["winner_vendor_name"] = winner["vendor_name"]
        item_decision["selected_price_inr_per_sheet"] = float(lowest_price)
        item_decision["annual_line_cost"] = float(annual_line_cost)
        # A calculation may recommend a supplier, but it must never make an
        # award. A buyer makes the final selection in the Streamlit workflow.
        item_decision["decision_status"] = "recommended"
        item_decision["rationale"] = (
            "Recommended: lowest eligible normalized INR-per-sheet quote. "
            "Quality qualified, comparable, review-approved, "
            "and freight included. Buyer confirmation is required before "
            "this line is awarded."
        )

        total_recommended_spend += annual_line_cost
        decisions.append(item_decision)

    return {
        "rfx_id": rfx_id,
        "total_awarded_spend": float(total_recommended_spend),
        "awarded_line_count": sum(
            decision["decision_status"] == "awarded"
            for decision in decisions
        ),
        "recommended_line_count": sum(
            decision["decision_status"] == "recommended"
            for decision in decisions
        ),
        "tie_line_count": sum(
            decision["decision_status"] == "tie_requires_buyer_decision"
            for decision in decisions
        ),
        "unawarded_line_count": sum(
            decision["decision_status"] == "unawarded"
            for decision in decisions
        ),
        "decisions": decisions,
    }
