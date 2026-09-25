import json
from pathlib import Path

import pandas as pd
import streamlit as st
from pydantic import ValidationError
from src.awards import calculate_award_recommendation
from src.database import (
    DuplicateDocumentError,
    get_document_by_checksum,
    get_comparison_matrix,
    get_latest_approved_rfx,
    get_review_task_history,
    initialize_database,
    list_review_tasks,
    list_submissions_for_rfx,
    mark_submission_failed,
    resolve_review_task,
    save_approved_rfx,
    save_document_extraction,
    list_approved_rfx,
    start_document_ingestion,
    normalize_all_quotes_for_rfx,
    get_latest_award_run,
    save_award_run,
    get_award_eligible_vendors_by_item,
    confirm_award_selections,
)
from src.extraction import (
    create_review_candidates,
    extract_vendor_document,
)
from src.analyst import (
    classify_buyer_question,
    execute_query_intent,
)
from src.gemini_client import GeminiClientError
from src.ingestion import (
    IngestionError,
    list_inbox_files,
    prepare_source_document,
)
from src.rfx_drafting import (
    AssumptionDecision,
    GeneratedRFxDraft,
    draft_to_streamlit_state,
    generate_rfx_draft,
    load_rfx_source_context,
)


# -------------------------------------------------------------------
# Project paths
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" 
INBOX_DIR = DATA_DIR / "inbox"
PROCESSED_DIR = DATA_DIR / "processed"


# -------------------------------------------------------------------
# Streamlit configuration
# -------------------------------------------------------------------
st.set_page_config(page_title="RFx Copilot", layout="wide")
st.title("RFx Copilot — Corrugated Sheets")

initialize_database()


# -------------------------------------------------------------------
# Persistent Streamlit state
# -------------------------------------------------------------------
if "draft" not in st.session_state:
    st.session_state.draft = None

if "approved_rfx_id" not in st.session_state:
    st.session_state.approved_rfx_id = None


# -------------------------------------------------------------------
# Active approved RFx
# -------------------------------------------------------------------
approved_rfxs_for_selection = list_approved_rfx()
selected_approved_rfx = None

if approved_rfxs_for_selection:
    approved_rfx_by_id = {
        rfx["rfx_id"]: rfx
        for rfx in approved_rfxs_for_selection
    }
    approved_rfx_ids = list(approved_rfx_by_id)

    selected_rfx_id = st.selectbox(
        "Active approved RFx",
        options=approved_rfx_ids,
        format_func=lambda rfx_id: (
            f"{rfx_id} — {approved_rfx_by_id[rfx_id]['title']}"
        ),
        help=(
            "Vendor inbox, review tasks, comparisons, analyst answers, and "
            "award recommendations use this approved RFx."
        ),
        key="selected_approved_rfx_id",
    )
    selected_approved_rfx = approved_rfx_by_id[selected_rfx_id]
else:
    st.info("Approve an RFx to enable vendor and comparison workflows.")


# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------
def reset_draft_widgets():
    """Clear widget values before loading a newly generated RFx draft."""

    widget_keys = [
        "draft_title",
        "draft_scope",
        "line_items_editor",
        "questionnaire_editor",
        "commercial_terms_editor",
        "assumptions_editor",
    ]

    for key in widget_keys:
        st.session_state.pop(key, None)


def dataframe_records(dataframe: pd.DataFrame) -> list[dict]:
    """Convert editable DataFrame content into Python/Pydantic records."""

    cleaned = dataframe.astype(object).where(pd.notna(dataframe), None)
    return cleaned.to_dict(orient="records")


def editable_draft_to_model(
    draft: dict,
) -> tuple[GeneratedRFxDraft, list[AssumptionDecision]]:
    """Validate buyer-edited Streamlit fields before RFx approval."""

    line_items = dataframe_records(draft["line_items"])
    questionnaire = dataframe_records(draft["questionnaire"])
    assumptions = dataframe_records(draft["assumptions"])

    assumption_decisions = [
        AssumptionDecision(
            assumption_text=str(row["assumption_text"]).strip(),
            accepted=bool(row["accepted"]),
        )
        for row in assumptions
        if row.get("assumption_text")
        and str(row["assumption_text"]).strip()
    ]

    validated_draft = GeneratedRFxDraft(
        title=draft["title"],
        scope=draft["scope"],
        line_items=line_items,
        questionnaire=questionnaire,
        commercial_terms=draft["commercial_terms"],
        assumptions=[
            assumption.assumption_text
            for assumption in assumption_decisions
        ],
    )

    return validated_draft, assumption_decisions


def create_source_file_draft(buyer_request: str) -> GeneratedRFxDraft:
    """Create a proposed RFx from the canonical source files without Gemini."""

    source_context = load_rfx_source_context(DATA_DIR)

    return GeneratedRFxDraft(
        title="Proposed Corrugated Sheet Supply RFx - Chennai Warehouse",
        scope=(
            "Proposed RFx for corrugated-sheet supply to the Chennai warehouse. "
            f"Buyer request: {buyer_request.strip()} "
            "The line items, questionnaire, and commercial terms are loaded "
            "directly from the canonical source files."
        ),
        line_items=source_context.line_items,
        questionnaire=source_context.questionnaire,
        commercial_terms=source_context.commercial_terms,
        assumptions=[
            "This is a proposed draft and requires buyer review before approval.",
            "Commercial comparison is intended to use INR per sheet, excluding GST.",
            "Canonical line items, questionnaire questions, and terms are loaded from the local source files.",
        ],
    )


def render_analyst_answer(intent, result: dict) -> None:
    """Present deterministic analyst results in buyer-friendly language."""

    st.markdown(f"#### {result['title']}")
    st.caption(f"Request understood: {intent.explanation}")

    if result.get("error"):
        st.warning(result["error"])
        return

    data = result.get("data")

    if intent.intent == "review_summary":
        total = data.get("total", 0)
        open_tasks = data.get("open", 0)
        clarification_tasks = data.get("clarification_needed", 0)
        resolved = data.get("resolved", 0)
        excluded = data.get("excluded", 0)

        if open_tasks or clarification_tasks:
            st.warning(
                f"There are {total} review task(s): {open_tasks} open and "
                f"{clarification_tasks} awaiting clarification."
            )
        else:
            st.success(
                f"There are no open review tasks. "
                f"{resolved} task(s) have been resolved."
            )

        st.write(
            f"Total: {total}; resolved: {resolved}; excluded: {excluded}."
        )
        return

    if intent.intent == "vendor_exclusion_reason":
        reasons = data.get("reasons", [])
        vendor_name = data.get("vendor_name", "This vendor")
        item_id = data.get("item_id", "this line item")

        if reasons:
            st.write(
                f"{vendor_name} is not currently award eligible for {item_id} "
                "for the following reason(s):"
            )
            for reason in reasons:
                st.write(f"- {reason}")
        else:
            st.success(
                f"No exclusion reason is currently recorded for "
                f"{vendor_name} on {item_id}."
            )
        return

    if isinstance(data, list):
        if not data:
            st.success("No matching records were found for this question.")
            return

        st.write(
            f"I found {len(data)} matching record(s). "
            "Supporting details are shown below."
        )
        st.dataframe(
            pd.DataFrame(data),
            width="stretch",
            hide_index=True,
        )
        return

    st.write("No additional information is available for this request.")


def ingest_inbox_file(inbox_file, approved_rfx: dict):
    """
    Run the complete Step 18 ingestion flow for one selected document.
    """

    prepared_document = prepare_source_document(
        file_path=inbox_file.path,
        processed_root=PROCESSED_DIR,
        rfx_id=approved_rfx["rfx_id"],
    )

    started_ingestion = start_document_ingestion(
        rfx_id=approved_rfx["rfx_id"],
        prepared_document=prepared_document,
    )

    try:
        extraction = extract_vendor_document(
            rfx_id=approved_rfx["rfx_id"],
            source_document=prepared_document,
            canonical_line_items=approved_rfx["line_items"],
            canonical_questions=approved_rfx["questionnaire"],
            api_key=st.secrets["GEMINI_API_KEY"],
            model=st.secrets["GEMINI_MODEL"],
        )

        review_candidates = create_review_candidates(
            extraction=extraction,
            canonical_line_items=approved_rfx["line_items"],
            canonical_questions=approved_rfx["questionnaire"],
        )
        if review_candidates:
            st.warning(
                f"{prepared_document.filename}: "
                f"{len(review_candidates)} review task(s) created."
            )

            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Issue type": candidate.issue_type,
                            "Severity": candidate.severity,
                            "Affected item": candidate.affected_item_id,
                            "Evidence": candidate.evidence_text,
                            "Current value": candidate.current_value,
                            "Suggested resolution": (
                                candidate.suggested_resolution
                            ),
                        }
                        for candidate in review_candidates
                    ]
                ),
                width="stretch",
                hide_index=True,
            )
        else:
            st.success(
                f"{prepared_document.filename}: no review tasks created."
            )    

        save_document_extraction(
            rfx_id=approved_rfx["rfx_id"],
            submission_id=started_ingestion.submission_id,
            document_id=started_ingestion.document_id,
            extraction=extraction,
            review_candidates=review_candidates,
        )

        return extraction, review_candidates

    except Exception as error:
        mark_submission_failed(
            submission_id=started_ingestion.submission_id,
            reason=str(error),
        )
        raise


# -------------------------------------------------------------------
# Application navigation
# -------------------------------------------------------------------
TAB_OPTIONS = [
    "RFx Draft and Approval",
    "Vendor Inbox",
    "Needs Review",
    "Comparison Matrix",
    "Analyst Copilot",
    "Award Recommendation",
]

active_tab = st.radio(
    "Application section",
    options=TAB_OPTIONS,
    horizontal=True,
    label_visibility="collapsed",
    key="active_application_tab",
)


def render_comparison_filters(rfx_id: str) -> None:
    """Render comparison filters only while the Comparison Matrix is active."""

    matrix_rows = get_comparison_matrix(rfx_id)

    if not matrix_rows:
        return

    matrix_df = pd.DataFrame(matrix_rows)

    available_vendors = sorted(
        matrix_df["vendor_name"]
        .dropna()
        .unique()
        .tolist()
    )
    selected_vendors = st.session_state.get(
        "comparison_filter_vendors",
        [],
    )
    st.session_state["comparison_filter_vendors"] = [
        vendor
        for vendor in selected_vendors
        if vendor in available_vendors
    ]

    with st.sidebar:
        st.markdown("### Comparison filters")

        st.multiselect(
            "Vendors",
            options=available_vendors,
            key="comparison_filter_vendors",
        )
        st.checkbox(
            "Only award-eligible quotes",
            key="comparison_filter_award_eligible",
        )
        st.checkbox(
            "Only lines needing review",
            key="comparison_filter_review_required",
        )
        st.checkbox(
            "Moisture-resistant only",
            key="comparison_filter_moisture_resistant",
        )
        st.checkbox(
            "Printable only",
            key="comparison_filter_printable",
        )


# -------------------------------------------------------------------
# RFx Draft and Approval tab
# -------------------------------------------------------------------
if active_tab == "RFx Draft and Approval":
    st.subheader("1. Buyer request input")

    buyer_request = st.text_area(
        "Describe the sourcing requirement",
        height=130,
        placeholder=(
            "I need annual supply of corrugated sheets for our Chennai warehouse.\n"
            "Include 3-ply, 5-ply, printable, and moisture-resistant grades."
        ),
    )

    ai_column, source_file_column = st.columns(2)

    with ai_column:
        generate_with_gemini = st.button(
            "Generate Draft RFx with Gemini",
            type="primary",
            disabled=not buyer_request.strip(),
            use_container_width=True,
        )

    with source_file_column:
        generate_from_source_files = st.button(
            "Load Draft from Source Files",
            disabled=not buyer_request.strip(),
            use_container_width=True,
        )

    if generate_with_gemini:
        try:
            with st.spinner("Gemini is preparing the RFx draft..."):
                generated_draft = generate_rfx_draft(
                    buyer_request=buyer_request,
                    data_dir=DATA_DIR,
                    api_key=st.secrets["GEMINI_API_KEY"],
                    model=st.secrets["GEMINI_MODEL"],
                )

            reset_draft_widgets()

            st.session_state.draft = draft_to_streamlit_state(generated_draft)
            st.session_state.draft["buyer_request"] = buyer_request.strip()
            st.session_state.approved_rfx_id = None

            st.success(
                "Draft generated. Review and edit it before approval."
            )

        except (
            GeminiClientError,
            FileNotFoundError,
            ValueError,
        ) as error:
            st.error(str(error))

    if generate_from_source_files:
        try:
            generated_draft = create_source_file_draft(buyer_request)

            reset_draft_widgets()

            st.session_state.draft = draft_to_streamlit_state(generated_draft)
            st.session_state.draft["buyer_request"] = buyer_request.strip()
            st.session_state.approved_rfx_id = None

            st.success(
                "Draft loaded from line_items.csv, questionnaire.json, and "
                "terms.md. Gemini was not called."
            )

        except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
            st.error(str(error))

    if st.session_state.draft is not None:
        draft = st.session_state.draft

        st.divider()
        st.subheader("2. Generated RFx Draft")

        st.caption("Buyer request")
        st.info(draft["buyer_request"])

        draft["title"] = st.text_input(
            "RFx title",
            value=draft["title"],
            key="draft_title",
        )

        draft["scope"] = st.text_area(
            "Scope",
            value=draft["scope"],
            height=130,
            key="draft_scope",
        )

        st.markdown("#### Line items")
        st.caption("Use the table controls to add, edit, or remove line items.")

        draft["line_items"] = st.data_editor(
            draft["line_items"],
            num_rows="dynamic",
            width="stretch",
            hide_index=True,
            key="line_items_editor",
            column_config={
                "annual_quantity_sheets": st.column_config.NumberColumn(
                    "Annual quantity (sheets)",
                    min_value=0,
                    step=1000,
                ),
                "ply": st.column_config.NumberColumn(
                    "Ply",
                    min_value=1,
                    step=1,
                ),
                "outer_gsm": st.column_config.NumberColumn(
                    "Outer GSM",
                    min_value=0,
                ),
                "moisture_resistant": st.column_config.SelectboxColumn(
                    "Moisture resistant",
                    options=["Yes", "No"],
                ),
            },
        )

        st.markdown("#### Supplier questionnaire")
        st.caption("Use the table controls to add, edit, or remove questions.")

        draft["questionnaire"] = st.data_editor(
            draft["questionnaire"],
            num_rows="dynamic",
            width="stretch",
            hide_index=True,
            key="questionnaire_editor",
            column_config={
                "answer_type": st.column_config.SelectboxColumn(
                    "Answer type",
                    options=["yes_no", "number", "text"],
                ),
                "required": st.column_config.CheckboxColumn("Required"),
                "quality_gate": st.column_config.CheckboxColumn("Quality gate"),
            },
        )

        st.markdown("#### Commercial terms")

        draft["commercial_terms"] = st.text_area(
            "Terms and conditions",
            value=draft["commercial_terms"],
            height=300,
            key="commercial_terms_editor",
        )

        st.markdown("#### Assumptions")
        st.caption(
            "Uncheck Accepted to reject an assumption. "
            "The assumption remains visible."
        )

        draft["assumptions"] = st.data_editor(
            draft["assumptions"],
            num_rows="dynamic",
            width="stretch",
            hide_index=True,
            key="assumptions_editor",
            column_config={
                "assumption_text": st.column_config.TextColumn(
                    "Assumption",
                    required=True,
                ),
                "accepted": st.column_config.CheckboxColumn("Accepted"),
            },
        )

        rejected_assumptions = draft["assumptions"].loc[
            draft["assumptions"]["accepted"] == False,  # noqa: E712
            "assumption_text",
        ].tolist()

        if rejected_assumptions:
            st.warning(
                "Rejected assumptions: "
                + " | ".join(rejected_assumptions)
            )

        st.divider()
        st.subheader("3. Approve RFx")

        if st.button("Approve RFx", type="primary"):
            try:
                validated_draft, assumption_decisions = (
                    editable_draft_to_model(draft)
                )

                rfx_id = save_approved_rfx(
                    draft=validated_draft,
                    buyer_request=draft["buyer_request"],
                    assumption_decisions=assumption_decisions,
                )

                st.session_state.approved_rfx_id = rfx_id
                st.session_state.draft = None

                st.success(f"RFx approved and saved: {rfx_id}")
                st.rerun()

            except ValidationError as error:
                st.error("Please correct the RFx fields before approval.")
                st.code(str(error))

            except Exception as error:
                st.error(f"Could not save the approved RFx: {error}")

    approved_rfxs = list_approved_rfx()

    if approved_rfxs:
        st.divider()
        st.subheader("Approved RFxs")

        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "RFx ID": rfx["rfx_id"],
                        "Title": rfx["title"],
                        "Approved at": rfx["approved_at"],
                        "Line items": len(rfx["line_items"]),
                    }
                    for rfx in approved_rfxs
                ]
            ),
            width="stretch",
            hide_index=True,
        )

        for rfx in approved_rfxs:
            with st.expander(
                f"{rfx['rfx_id']} — {rfx['title']}",
                expanded=False,
            ):
                st.write(f"**Status:** {rfx['status']}")
                st.write(f"**Approved at:** {rfx['approved_at']}")
                st.write(f"**Scope:** {rfx['scope']}")

                st.markdown("#### Approved line items")
                st.dataframe(
                    pd.DataFrame(rfx["line_items"]),
                    width="stretch",
                    hide_index=True,
                )

                st.markdown("#### Approved questionnaire")
                st.dataframe(
                    pd.DataFrame(rfx["questionnaire"]),
                    width="stretch",
                    hide_index=True,
                )

                st.markdown("#### Commercial terms")
                st.text(rfx["commercial_terms"])

                st.markdown("#### Assumptions")
                st.dataframe(
                    pd.DataFrame(rfx["assumptions"]),
                    width="stretch",
                    hide_index=True,
                )

                st.download_button(
                    "Download approved RFx as JSON",
                    data=json.dumps(rfx, indent=2, default=str),
                    file_name=f"{rfx['rfx_id']}.json",
                    mime="application/json",
                    key=f"download_{rfx['rfx_id']}",
                )


# -------------------------------------------------------------------
# Vendor Inbox tab - Step 18
# -------------------------------------------------------------------
if active_tab == "Vendor Inbox":
    st.subheader("Vendor Inbox")

    latest_approved_rfx = selected_approved_rfx

    if latest_approved_rfx is None:
        st.warning(
            "Approve an RFx before ingesting vendor-response documents."
        )
        st.stop()

    st.caption(
        f"Documents will be linked to approved RFx: "
        f"{latest_approved_rfx['rfx_id']}"
    )

    INBOX_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    inbox_files = list_inbox_files(INBOX_DIR)

    if not inbox_files:
        st.info(
            "No supported vendor files found. Add the five Step 17 files to "
            "data/inbox and refresh this page."
        )
    else:
        st.markdown("#### Files available for ingestion")

        for inbox_file in inbox_files:
            existing_document = get_document_by_checksum(
                rfx_id=latest_approved_rfx["rfx_id"],
                sha256_checksum=inbox_file.sha256_checksum,
            )

            left_column, middle_column, right_column = st.columns(
                [4, 3, 2]
            )

            with left_column:
                st.markdown(f"**{inbox_file.filename}**")
                st.caption(
                    f"{inbox_file.extension.upper()} | "
                    f"{inbox_file.file_size_bytes:,} bytes"
                )

            with middle_column:
                if existing_document:
                    st.success(
                        f"Already ingested - "
                        f"{existing_document['status']}"
                    )
                else:
                    st.info("New")

            with right_column:
                if existing_document:
                    st.button(
                        "Already ingested",
                        disabled=True,
                        key=f"done_{inbox_file.sha256_checksum}",
                    )
                else:
                    if st.button(
                        "Ingest",
                        type="primary",
                        key=f"ingest_{inbox_file.sha256_checksum}",
                    ):
                        try:
                            with st.spinner(
                                f"Extracting {inbox_file.filename}..."
                            ):
                                extraction, review_candidates = (
                                    ingest_inbox_file(
                                        inbox_file,
                                        latest_approved_rfx,
                                    )
                                )

                            vendor_name = (
                                extraction.vendor_name
                                or "Unknown vendor"
                            )

                            st.success(
                                f"Ingested {inbox_file.filename}. "
                                f"Vendor: {vendor_name}. "
                                f"Quotes: {len(extraction.quotes)}. "
                                f"Answers: "
                                f"{len(extraction.questionnaire_answers)}. "
                                f"Review tasks: "
                                f"{len(review_candidates)}."
                            )

                            #st.rerun()

                        except DuplicateDocumentError as error:
                            st.warning(str(error))

                        except (
                            IngestionError,
                            GeminiClientError,
                            FileNotFoundError,
                            ValueError,
                        ) as error:
                            st.error(f"Could not ingest the document: {error}")

                        except Exception as error:
                            st.error(
                                "Unexpected ingestion error. "
                                f"Details: {error}"
                            )

    st.divider()
    st.subheader("Saved vendor submissions")

    saved_submissions = list_submissions_for_rfx(
        latest_approved_rfx["rfx_id"]
    )

    if saved_submissions:
        submission_rows = []

        for submission in saved_submissions:
            filenames = ", ".join(
                document["filename"]
                for document in submission["source_documents"]
            )

            submission_rows.append(
                {
                    "Submission ID": submission["submission_id"],
                    "Vendor": submission["vendor_name"],
                    "Status": submission["status"],
                    "Received at": submission["received_at"],
                    "Source document(s)": filenames,
                }
            )

        st.dataframe(
            pd.DataFrame(submission_rows),
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("No vendor documents have been ingested for this RFx.")
# -------------------------------------------------------------------
# Needs Review tab - Step 20
# -------------------------------------------------------------------
if active_tab == "Needs Review":
    st.subheader("Needs Review")

    latest_approved_rfx = selected_approved_rfx

    if latest_approved_rfx is None:
        st.warning(
            "Approve an RFx and ingest vendor documents before reviewing tasks."
        )
    else:
        st.caption(
            f"Review tasks for RFx: {latest_approved_rfx['rfx_id']}"
        )

        status_mapping = {
            "Open and clarification needed": [
                "open",
                "clarification_needed",
            ],
            "Open only": ["open"],
            "Clarification needed only": ["clarification_needed"],
            "Resolved": ["resolved"],
            "Excluded": ["excluded"],
            "All": [
                "open",
                "clarification_needed",
                "resolved",
                "excluded",
            ],
        }

        # Load all tasks once to provide a stable vendor filter. The selected
        # status and vendor are then both applied to the visible task list.
        all_review_tasks = list_review_tasks(
            rfx_id=latest_approved_rfx["rfx_id"],
            statuses=status_mapping["All"],
        )
        available_vendors = sorted(
            {
                task["vendor_name"]
                for task in all_review_tasks
                if task.get("vendor_name")
            }
        )

        status_column, vendor_column = st.columns(2)

        with status_column:
            task_status_filter = st.selectbox(
                "Show tasks",
                options=[
                    "Open and clarification needed",
                    "Open only",
                    "Clarification needed only",
                    "Resolved",
                    "Excluded",
                    "All",
                ],
                key="needs_review_status_filter",
            )

        with vendor_column:
            selected_vendor = st.selectbox(
                "Vendor",
                options=["All vendors", *available_vendors],
                key="needs_review_vendor_filter",
            )

        review_tasks = list_review_tasks(
            rfx_id=latest_approved_rfx["rfx_id"],
            statuses=status_mapping[task_status_filter],
        )

        if selected_vendor != "All vendors":
            review_tasks = [
                task
                for task in review_tasks
                if task.get("vendor_name") == selected_vendor
            ]

        if not review_tasks:
            st.success("No review tasks match the selected filter.")
        else:
            st.metric(
                "Review tasks shown",
                len(review_tasks),
            )

            export_rows = [
                {
                    "Task ID": task["task_id"],
                    "Vendor": task["vendor_name"],
                    "Issue type": task["issue_type"],
                    "Severity": task["severity"],
                    "Affected item": task["affected_item_id"],
                    "Source file": task["source_filename"],
                    "Status": task["status"],
                    "Created at": task["created_at"],
                }
                for task in review_tasks
            ]

            st.download_button(
                "Download Needs Review CSV",
                data=pd.DataFrame(export_rows).to_csv(index=False),
                file_name="needs_review.csv",
                mime="text/csv",
            )

            for task in review_tasks:
                title = (
                    f"{task['severity']} - {task['issue_type']} - "
                    f"{task['vendor_name']}"
                )

                if task["affected_item_id"]:
                    title += f" - {task['affected_item_id']}"

                with st.expander(title, expanded=task["status"] == "open"):
                    left_column, right_column = st.columns(2)

                    with left_column:
                        st.write(f"**Task ID:** {task['task_id']}")
                        st.write(f"**Vendor:** {task['vendor_name']}")
                        st.write(
                            f"**Source file:** "
                            f"{task['source_filename'] or 'Not available'}"
                        )
                        st.write(f"**Issue type:** {task['issue_type']}")
                        st.write(f"**Severity:** {task['severity']}")
                        st.write(f"**Status:** {task['status']}")

                        if task["affected_item_id"]:
                            st.write(
                                f"**Affected item:** "
                                f"{task['affected_item_id']}"
                            )

                    with right_column:
                        st.markdown("**Evidence**")
                        st.info(task["evidence_text"])

                        st.markdown("**Current extracted value**")
                        st.code(
                            task["current_value"]
                            or "No usable extracted value"
                        )

                        st.markdown("**Suggested resolution**")
                        st.write(task["suggested_resolution"])

                    st.divider()

                    task_history = get_review_task_history(task["task_id"])

                    if task_history:
                        st.markdown("#### Decision history")
                        st.dataframe(
                            pd.DataFrame(task_history),
                            width="stretch",
                            hide_index=True,
                        )

                    if task["status"] in {"open", "clarification_needed"}:
                        st.markdown("#### Buyer decision")

                        action = st.radio(
                            "Action",
                            options=[
                                "Approve",
                                "Correct",
                                "Mark missing",
                                "Exclude",
                                "Clarification needed",
                            ],
                            horizontal=True,
                            key=f"action_{task['task_id']}",
                        )

                        corrected_value = None

                        if action == "Correct":
                            corrected_value = st.text_input(
                                "Corrected value",
                                placeholder=(
                                    "Example: INR 4,300 per 100 sheets"
                                ),
                                key=f"corrected_value_{task['task_id']}",
                            )

                        resolution_reason = st.text_area(
                            "Resolution reason",
                            placeholder=(
                                "Explain why this decision was made. "
                                "The original extraction remains preserved."
                            ),
                            key=f"reason_{task['task_id']}",
                        )

                        if st.button(
                            "Save decision",
                            type="primary",
                            key=f"save_{task['task_id']}",
                        ):
                            try:
                                resolve_review_task(
                                    task_id=task["task_id"],
                                    action=action,
                                    corrected_value=corrected_value,
                                    resolution_reason=resolution_reason,
                                )

                                st.success(
                                    f"Review task updated: {action}"
                                )
                                st.rerun()

                            except Exception as error:
                                st.error(
                                    f"Could not save the decision: {error}"
                                )

                    else:
                        st.success(
                            "This task has already been resolved. "
                            "Its original extraction and decision history "
                            "remain available above."
                        )
# -------------------------------------------------------------------
# Comparison Matrix tab - Step 21
# -------------------------------------------------------------------
if active_tab == "Comparison Matrix":
    st.subheader("Comparison Matrix")

    latest_approved_rfx = selected_approved_rfx

    if latest_approved_rfx is None:
        st.warning(
            "Approve an RFx and ingest vendor responses first."
        )
    else:
        st.caption(
            f"Comparison for approved RFx: "
            f"{latest_approved_rfx['rfx_id']}"
        )

        render_comparison_filters(latest_approved_rfx["rfx_id"])

        if st.button("Normalize all existing quotes", type="primary"):
            try:
                normalized_count = normalize_all_quotes_for_rfx(
                    latest_approved_rfx["rfx_id"]
                )

                st.success(
                    f"Created {normalized_count} new normalized quote record(s)."
                )
                st.rerun()

            except Exception as error:
                st.error(f"Normalization failed: {error}")

        matrix_rows = get_comparison_matrix(
            latest_approved_rfx["rfx_id"]
        )

        if not matrix_rows:
            st.info(
                "No normalized quotes found. Ingest vendor documents, then "
                "select Normalize all existing quotes."
            )
        else:
            matrix_df = pd.DataFrame(matrix_rows)

            selected_vendors = st.session_state.get(
                "comparison_filter_vendors",
                [],
            )
            quality_filter = st.session_state.get(
                "comparison_filter_award_eligible",
                False,
            )
            review_filter = st.session_state.get(
                "comparison_filter_review_required",
                False,
            )
            moisture_filter = st.session_state.get(
                "comparison_filter_moisture_resistant",
                False,
            )
            printable_filter = st.session_state.get(
                "comparison_filter_printable",
                False,
            )

            filtered_df = matrix_df.copy()

            if selected_vendors:
                filtered_df = filtered_df[
                    filtered_df["vendor_name"].isin(selected_vendors)
                ]

            if quality_filter:
                filtered_df = filtered_df[
                    filtered_df["award_eligible"] == True
                ]

            if review_filter:
                filtered_df = filtered_df[
                    filtered_df["review_required"] == True
                ]

            if moisture_filter:
                filtered_df = filtered_df[
                    filtered_df["moisture_resistant"] == "Yes"
                ]

            if printable_filter:
                filtered_df = filtered_df[
                    filtered_df["print_requirement"]
                    .fillna("")
                    .str.lower()
                    .ne("none")
                ]

            filtered_df["Normalized INR/sheet"] = filtered_df[
                "normalized_inr_per_sheet"
            ].apply(
                lambda value: (
                    f"₹{value:,.2f}"
                    if pd.notna(value)
                    else "Not comparable"
                )
            )

            filtered_df["Raw quote"] = filtered_df.apply(
                lambda row: (
                    f"{row['raw_currency'] or ''} "
                    f"{row['raw_price'] or ''} "
                    f"{row['raw_unit'] or ''}"
                ).strip(),
                axis=1,
            )

            display_columns = [
                "item_id",
                "description",
                "ply",
                "flute",
                "outer_gsm",
                "dimensions_mm",
                "annual_quantity_sheets",
                "vendor_name",
                "Normalized INR/sheet",
                "Raw quote",
                "confidence",
                "freight_status",
                "comparable",
                "review_required",
                "award_eligible",
            ]

            st.dataframe(
                filtered_df[display_columns],
                width="stretch",
                hide_index=True,
            )

            st.download_button(
                "Download comparison_matrix.csv",
                data=filtered_df.to_csv(index=False),
                file_name="comparison_matrix.csv",
                mime="text/csv",
            )

            st.markdown("#### Evidence and normalization details")

            for _, row in filtered_df.iterrows():
                label = (
                    f"{row['item_id']} | {row['vendor_name']} | "
                    f"{row['Normalized INR/sheet']}"
                )

                with st.expander(label):
                    st.write(
                        f"**Source file:** "
                        f"{row['source_filename'] or 'Not available'}"
                    )
                    st.write(
                        f"**Page/section:** "
                        f"{row['page_or_section'] or 'Not available'}"
                    )
                    st.write(f"**Evidence:** {row['evidence_text']}")
                    st.write(
                        f"**Normalization formula:** "
                        f"{row['normalization_formula'] or 'Not available'}"
                    )
                    st.write(
                        "**Review reasons:** "
                        + " | ".join(row["normalization_reasons"])
                    )
# -------------------------------------------------------------------
# Analyst Copilot tab - Step 22
# -------------------------------------------------------------------
if active_tab == "Analyst Copilot":
    st.subheader("Analyst Copilot")

    latest_approved_rfx = selected_approved_rfx

    if latest_approved_rfx is None:
        st.warning("Approve an RFx before using the Analyst Copilot.")
    else:
        st.caption(
            "Gemini interprets the question. Python and SQLite calculate "
            "the answer using approved functions."
        )

        buyer_question = st.text_area(
            "Ask a procurement question",
            placeholder=(
                "Compare qualified vendor prices for CS-015.\n"
                "Why is PrimeBoard excluded for CS-015?\n"
                "Which vendor responses have missing quotes?"
            ),
            height=120,
        )

        if st.button(
            "Ask Analyst Copilot",
            type="primary",
            disabled=not buyer_question.strip(),
        ):
            try:
                matrix_rows = get_comparison_matrix(
                    latest_approved_rfx["rfx_id"]
                )

                item_ids = sorted(
                    {
                        row["item_id"]
                        for row in matrix_rows
                    }
                )

                vendor_names = sorted(
                    {
                        row["vendor_name"]
                        for row in matrix_rows
                        if row["vendor_name"] != "No quote"
                    }
                )

                with st.spinner("Interpreting the question..."):
                    intent = classify_buyer_question(
                        buyer_question=buyer_question,
                        item_ids=item_ids,
                        vendor_names=vendor_names,
                        api_key=st.secrets["GEMINI_API_KEY"],
                        model=st.secrets["GEMINI_MODEL"],
                    )

                result = execute_query_intent(
                    rfx_id=latest_approved_rfx["rfx_id"],
                    intent=intent,
                )

                render_analyst_answer(intent, result)

            except GeminiClientError as error:
                st.error(f"Gemini could not interpret the question: {error}")

            except Exception as error:
                st.error(f"Could not answer the question: {error}")

# -------------------------------------------------------------------
# Award Recommendation tab - Step 23
# -------------------------------------------------------------------
if active_tab == "Award Recommendation":
    st.subheader("Award Recommendation")

    latest_approved_rfx = selected_approved_rfx

    if latest_approved_rfx is None:
        st.warning(
            "Approve an RFx, ingest vendor responses, and normalize quotes first."
        )
    else:
        st.caption(
            "Recommendation uses deterministic Python logic only. "
            "It does not award line items. The buyer selects suppliers and "
            "confirms awards explicitly."
        )

        if st.button(
            "Calculate Award Recommendation",
            type="primary",
        ):
            try:
                award_result = calculate_award_recommendation(
                    latest_approved_rfx["rfx_id"]
                )

                award_run_id = save_award_run(award_result)

                st.success(
                    f"Recommendation saved: {award_run_id}. "
                    "Select suppliers below and confirm the awards."
                )
                st.rerun()

            except Exception as error:
                st.error(f"Award calculation failed: {error}")

        latest_award_run = get_latest_award_run(
            latest_approved_rfx["rfx_id"]
        )

        if latest_award_run:
            st.metric(
                "Recommended / confirmed spend",
                f"₹{latest_award_run['total_awarded_spend']:,.2f}",
            )

            summary_left, summary_middle, summary_right = st.columns(3)

            awarded_count = sum(
                decision["decision_status"] == "awarded"
                for decision in latest_award_run["decisions"]
            )

            recommended_count = sum(
                decision["decision_status"] == "recommended"
                for decision in latest_award_run["decisions"]
            )

            tie_count = sum(
                decision["decision_status"]
                == "tie_requires_buyer_decision"
                for decision in latest_award_run["decisions"]
            )

            unawarded_count = sum(
                decision["decision_status"] == "unawarded"
                for decision in latest_award_run["decisions"]
            )

            summary_left.metric("Buyer-confirmed awards", awarded_count)
            summary_middle.metric("Recommendations", recommended_count)
            summary_right.metric(
                "Tied / unawarded lines", tie_count + unawarded_count
            )

            # Keep every decision in Grid 2, including awards already saved.
            # This makes an approved buyer award visible after reruns and reloads.
            award_grid_decisions = list(latest_award_run["decisions"])

            eligible_choices = get_award_eligible_vendors_by_item(
                latest_approved_rfx["rfx_id"]
            )

            decisions_df = pd.DataFrame(latest_award_run["decisions"])

            # Grid 1 shows every ingested supplier quote for the item, rather
            # than only the winning recommendation. This gives the buyer a
            # transparent comparison before making an override in Grid 2.
            comparison_rows = get_comparison_matrix(
                latest_approved_rfx["rfx_id"]
            )
            all_vendor_quotes_by_item: dict[str, list[str]] = {}
            for quote in comparison_rows:
                item_id = quote.get("item_id")
                vendor_name = quote.get("vendor_name", "Unknown vendor")
                price = quote.get("normalized_inr_per_sheet")
                eligibility = (
                    "eligible" if quote.get("award_eligible") else "not eligible"
                )

                if price is None:
                    quote_text = f"{vendor_name}: no comparable quote"
                else:
                    quote_text = f"{vendor_name}: INR {float(price):,.2f}/sheet"

                if eligibility != "eligible":
                    quote_text += f" ({eligibility})"

                all_vendor_quotes_by_item.setdefault(item_id, []).append(quote_text)

            decisions_df["all_vendor_quotes"] = decisions_df["item_id"].map(
                lambda item_id: " | ".join(
                    all_vendor_quotes_by_item.get(item_id, [])
                )
                or "No vendor quotes ingested"
            )

            if award_grid_decisions:
                # Grid 2: this is the only editable award grid.
                st.markdown("#### Buyer award grid")
                st.caption(
                    "Select a supplier and tick Confirm this line only for the "
                    "lines you want to award now. Existing awarded lines remain "
                    "visible. The live preview below updates the price and annual "
                    "cost when you choose an eligible vendor."
                )

                pending_df = pd.DataFrame(award_grid_decisions).copy()
                pending_df["eligible_vendors"] = pending_df["item_id"].map(
                    lambda item_id: ", ".join(
                        choice["vendor_name"]
                        for choice in eligible_choices.get(item_id, [])
                    )
                )
                # Persisted awards arrive from SQLite with their selected
                # supplier. Pre-fill those rows so they never vanish from Grid 2.
                pending_df["buyer_selected_vendor"] = pending_df.apply(
                    lambda row: (
                        row["winner_vendor_name"]
                        if row["decision_status"] == "awarded"
                        else ""
                    ),
                    axis=1,
                )
                pending_df["confirm_award"] = False

                vendor_options = [""] + sorted(
                    {
                        choice["vendor_name"]
                        for choices in eligible_choices.values()
                        for choice in choices
                    }
                )
                grid_key = (
                    f"award_selection_grid_{latest_award_run['award_run_id']}_"
                    f"{'_'.join(pending_df['item_id'].tolist())}"
                )
                edited_pending_df = st.data_editor(
                    pending_df,
                    width="stretch",
                    hide_index=True,
                    key=grid_key,
                    disabled=[
                        "item_id",
                        "description",
                        "annual_quantity_sheets",
                        "winner_vendor_name",
                        "selected_price_inr_per_sheet",
                        "annual_line_cost",
                        "decision_status",
                        "rationale",
                        "eligible_vendors",
                    ],
                    column_config={
                        "eligible_vendors": st.column_config.TextColumn(
                            "Eligible vendors for this line",
                        ),
                        "buyer_selected_vendor": (
                            st.column_config.SelectboxColumn(
                                "Buyer-selected vendor",
                                options=vendor_options,
                                required=False,
                            )
                        ),
                        "confirm_award": st.column_config.CheckboxColumn(
                            "Confirm this line",
                            help=(
                                "Only checked rows with a selected eligible "
                                "vendor will be awarded."
                            ),
                        ),
                    },
                )

                # data_editor keeps calculated columns read-only. Render a
                # companion grid using its current selections so price and line
                # cost change immediately whenever the buyer selects a vendor.
                preview_df = edited_pending_df.copy()
                preview_prices: list[float | None] = []
                preview_costs: list[float | None] = []
                preview_statuses: list[str] = []

                for _, row in preview_df.iterrows():
                    selected_vendor = row["buyer_selected_vendor"]
                    matching_quote = next(
                        (
                            choice
                            for choice in eligible_choices.get(row["item_id"], [])
                            if choice["vendor_name"] == selected_vendor
                        ),
                        None,
                    )

                    if matching_quote is not None:
                        selected_price = float(
                            matching_quote["price_inr_per_sheet"]
                        )
                        preview_prices.append(selected_price)
                        preview_costs.append(
                            selected_price
                            * float(row["annual_quantity_sheets"])
                        )
                        preview_statuses.append(
                            "Saved award"
                            if row["decision_status"] == "awarded"
                            else "Ready to confirm"
                        )
                    elif selected_vendor:
                        preview_prices.append(None)
                        preview_costs.append(None)
                        preview_statuses.append(
                            "Selected vendor is not eligible for this line"
                        )
                    else:
                        preview_prices.append(row["selected_price_inr_per_sheet"])
                        preview_costs.append(row["annual_line_cost"])
                        preview_statuses.append("No buyer selection yet")

                preview_df["buyer_selected_price_inr_per_sheet"] = preview_prices
                preview_df["buyer_selected_annual_line_cost"] = preview_costs
                preview_df["buyer_selection_status"] = preview_statuses

                st.markdown("##### Live buyer-selection preview")
                st.caption(
                    "This grid is recalculated from the selected vendor. Awarded "
                    "rows remain here as saved records and are not changed again."
                )
                st.dataframe(
                    preview_df[
                        [
                            "item_id",
                            "description",
                            "annual_quantity_sheets",
                            "buyer_selected_vendor",
                            "buyer_selected_price_inr_per_sheet",
                            "buyer_selected_annual_line_cost",
                            "buyer_selection_status",
                        ]
                    ],
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "buyer_selected_price_inr_per_sheet": (
                            st.column_config.NumberColumn(
                                "Selected price (INR/sheet)", format="%.2f"
                            )
                        ),
                        "buyer_selected_annual_line_cost": (
                            st.column_config.NumberColumn(
                                "Selected annual line cost", format="%.2f"
                            )
                        ),
                    },
                )

                if st.button(
                    "Confirm selected awards",
                    type="primary",
                ):
                    selected_rows = edited_pending_df.loc[
                        (edited_pending_df["confirm_award"] == True)  # noqa: E712
                        & (edited_pending_df["decision_status"] != "awarded")
                    ]
                    selections = {
                        row["item_id"]: row["buyer_selected_vendor"]
                        for _, row in selected_rows.iterrows()
                        if row["buyer_selected_vendor"]
                    }
                    try:
                        confirmed_count = confirm_award_selections(
                            latest_award_run["award_run_id"], selections
                        )
                        st.success(
                            f"Confirmed {confirmed_count} buyer-selected "
                            "award(s)."
                        )
                        st.rerun()
                    except Exception as error:
                        st.error(f"Could not confirm awards: {error}")

            st.download_button(
                "Download award_recommendation.csv",
                data=decisions_df.to_csv(index=False),
                file_name="award_recommendation.csv",
                mime="text/csv",
            )

            st.markdown("#### Exclusion reasons")

            for decision in latest_award_run["decisions"]:
                with st.expander(
                    f"{decision['item_id']} - "
                    f"{decision['decision_status']}"
                ):
                    st.write(f"**Rationale:** {decision['rationale']}")

                    if decision["excluded_vendors"]:
                        st.dataframe(
                            pd.DataFrame(
                                decision["excluded_vendors"]
                            ),
                            width="stretch",
                            hide_index=True,
                        )

                    if decision["decision_status"] == (
                        "tie_requires_buyer_decision"
                    ):
                        st.warning(
                            "Tie detected. The buyer must select the supplier; "
                            "the app will not choose one automatically."
                        )
