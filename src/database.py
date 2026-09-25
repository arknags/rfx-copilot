import json

from decimal import Decimal
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from src.normalization import RawQuoteInput, normalize_quote


from sqlalchemy import (
    Boolean,
    and_,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    select,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
)

from src.extraction_schemas import (
    ReviewCandidate,
    VendorDocumentExtraction,
)
from src.ingestion import PreparedSourceDocument
from src.rfx_drafting import (
    AssumptionDecision,
    GeneratedRFxDraft,
)


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATABASE_FILE = PROJECT_DIR / "data" / "app.db"


class DatabaseError(RuntimeError):
    """Raised when a database operation cannot be completed."""


class DuplicateDocumentError(DatabaseError):
    """Raised when the same source document was already ingested."""


@dataclass(frozen=True)
class IngestionStarted:
    submission_id: str
    document_id: str


class Base(DeclarativeBase):
    pass


# -------------------------------------------------------------------
# Step 16: Approved RFx tables
# -------------------------------------------------------------------
class RFxRecord(Base):
    __tablename__ = "rfx"

    rfx_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    buyer_request: Mapped[str] = mapped_column(Text, nullable=False)
    scope: Mapped[str] = mapped_column(Text, nullable=False)
    commercial_terms: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(30), nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    approved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )

    line_items: Mapped[list["LineItemRecord"]] = relationship(
        back_populates="rfx",
        cascade="all, delete-orphan",
        order_by="LineItemRecord.id",
    )

    questionnaire_questions: Mapped[list["QuestionnaireQuestionRecord"]] = relationship(
        back_populates="rfx",
        cascade="all, delete-orphan",
        order_by="QuestionnaireQuestionRecord.id",
    )

    assumptions: Mapped[list["RFxAssumptionRecord"]] = relationship(
        back_populates="rfx",
        cascade="all, delete-orphan",
        order_by="RFxAssumptionRecord.id",
    )

    vendor_submissions: Mapped[list["VendorSubmissionRecord"]] = relationship(
        back_populates="rfx",
        cascade="all, delete-orphan",
        order_by="VendorSubmissionRecord.received_at",
    )

    review_tasks: Mapped[list["ReviewTaskRecord"]] = relationship(
        back_populates="rfx",
        cascade="all, delete-orphan",
        order_by="ReviewTaskRecord.created_at",
    )


class LineItemRecord(Base):
    __tablename__ = "line_items"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    rfx_id: Mapped[str] = mapped_column(
        ForeignKey("rfx.rfx_id"),
        nullable=False,
    )

    item_id: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    ply: Mapped[int] = mapped_column(Integer, nullable=False)
    flute: Mapped[str] = mapped_column(String(100), nullable=False)
    outer_gsm: Mapped[int] = mapped_column(Integer, nullable=False)
    dimensions_mm: Mapped[str] = mapped_column(String(100), nullable=False)
    print_requirement: Mapped[str | None] = mapped_column(String(255))
    moisture_resistant: Mapped[str] = mapped_column(String(10), nullable=False)
    annual_quantity_sheets: Mapped[int] = mapped_column(Integer, nullable=False)
    standard_unit: Mapped[str] = mapped_column(String(50), nullable=False)
    delivery_location: Mapped[str] = mapped_column(String(255), nullable=False)

    rfx: Mapped["RFxRecord"] = relationship(back_populates="line_items")


class QuestionnaireQuestionRecord(Base):
    __tablename__ = "questionnaire_questions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    rfx_id: Mapped[str] = mapped_column(
        ForeignKey("rfx.rfx_id"),
        nullable=False,
    )

    question_id: Mapped[str] = mapped_column(String(50), nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer_type: Mapped[str] = mapped_column(String(30), nullable=False)
    required: Mapped[bool] = mapped_column(Boolean, nullable=False)
    quality_gate: Mapped[bool] = mapped_column(Boolean, nullable=False)

    rfx: Mapped["RFxRecord"] = relationship(
        back_populates="questionnaire_questions"
    )


class RFxAssumptionRecord(Base):
    __tablename__ = "rfx_assumptions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    rfx_id: Mapped[str] = mapped_column(
        ForeignKey("rfx.rfx_id"),
        nullable=False,
    )

    assumption_text: Mapped[str] = mapped_column(Text, nullable=False)
    accepted: Mapped[bool] = mapped_column(Boolean, nullable=False)

    rfx: Mapped["RFxRecord"] = relationship(back_populates="assumptions")


# -------------------------------------------------------------------
# Step 18: Vendor ingestion and extraction tables
# -------------------------------------------------------------------
class VendorRecord(Base):
    __tablename__ = "vendors"

    vendor_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    submissions: Mapped[list["VendorSubmissionRecord"]] = relationship(
        back_populates="vendor",
        order_by="VendorSubmissionRecord.received_at",
    )


class VendorSubmissionRecord(Base):
    __tablename__ = "vendor_submissions"

    submission_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    rfx_id: Mapped[str] = mapped_column(
        ForeignKey("rfx.rfx_id"),
        nullable=False,
    )

    # This is initially null. Gemini identifies the vendor after extraction.
    vendor_id: Mapped[str | None] = mapped_column(
        ForeignKey("vendors.vendor_id"),
        nullable=True,
    )

    received_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(String(30), nullable=False)

    rfx: Mapped["RFxRecord"] = relationship(
        back_populates="vendor_submissions"
    )
    vendor: Mapped["VendorRecord | None"] = relationship(
        back_populates="submissions"
    )

    source_documents: Mapped[list["SourceDocumentRecord"]] = relationship(
        back_populates="submission",
        cascade="all, delete-orphan",
        order_by="SourceDocumentRecord.created_at",
    )

    extracted_quotes: Mapped[list["ExtractedQuoteRecord"]] = relationship(
        back_populates="submission",
        cascade="all, delete-orphan",
        order_by="ExtractedQuoteRecord.id",
    )

    questionnaire_answers: Mapped[list["QuestionnaireAnswerRecord"]] = relationship(
        back_populates="submission",
        cascade="all, delete-orphan",
        order_by="QuestionnaireAnswerRecord.id",
    )

    review_tasks: Mapped[list["ReviewTaskRecord"]] = relationship(
        back_populates="submission",
        cascade="all, delete-orphan",
        order_by="ReviewTaskRecord.created_at",
    )


class SourceDocumentRecord(Base):
    __tablename__ = "source_documents"

    document_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    submission_id: Mapped[str] = mapped_column(
        ForeignKey("vendor_submissions.submission_id"),
        nullable=False,
    )

    original_filename: Mapped[str] = mapped_column(String(500), nullable=False)
    original_path: Mapped[str] = mapped_column(Text, nullable=False)
    processed_path: Mapped[str] = mapped_column(Text, nullable=False)
    file_type: Mapped[str] = mapped_column(String(50), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    sha256_checksum: Mapped[str] = mapped_column(String(64), nullable=False)

    # Deterministic baseline text extracted by Python libraries.
    extracted_text: Mapped[str] = mapped_column(Text, nullable=False)

    # Raw Gemini structured output, retained as extraction evidence.
    extraction_json: Mapped[str | None] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    submission: Mapped["VendorSubmissionRecord"] = relationship(
        back_populates="source_documents"
    )

    extracted_quotes: Mapped[list["ExtractedQuoteRecord"]] = relationship(
        back_populates="source_document",
        cascade="all, delete-orphan",
        order_by="ExtractedQuoteRecord.id",
    )

    questionnaire_answers: Mapped[list["QuestionnaireAnswerRecord"]] = relationship(
        back_populates="source_document",
        cascade="all, delete-orphan",
        order_by="QuestionnaireAnswerRecord.id",
    )

    review_tasks: Mapped[list["ReviewTaskRecord"]] = relationship(
        back_populates="source_document",
        cascade="all, delete-orphan",
        order_by="ReviewTaskRecord.created_at",
    )


class ExtractedQuoteRecord(Base):
    __tablename__ = "extracted_quotes"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    submission_id: Mapped[str] = mapped_column(
        ForeignKey("vendor_submissions.submission_id"),
        nullable=False,
    )
    document_id: Mapped[str] = mapped_column(
        ForeignKey("source_documents.document_id"),
        nullable=False,
    )

    canonical_item_id: Mapped[str | None] = mapped_column(String(100))
    raw_description: Mapped[str] = mapped_column(Text, nullable=False)
    raw_price: Mapped[float | None] = mapped_column(Float)
    raw_unit: Mapped[str | None] = mapped_column(String(255))
    currency: Mapped[str | None] = mapped_column(String(10))
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    evidence_text: Mapped[str] = mapped_column(Text, nullable=False)
    page_or_section: Mapped[str | None] = mapped_column(String(255))
    extraction_note: Mapped[str | None] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    submission: Mapped["VendorSubmissionRecord"] = relationship(
        back_populates="extracted_quotes"
    )
    source_document: Mapped["SourceDocumentRecord"] = relationship(
        back_populates="extracted_quotes"
    )

class NormalizedQuoteRecord(Base):
    __tablename__ = "normalized_quotes"

    normalized_quote_id: Mapped[str] = mapped_column(
        String(40),
        primary_key=True,
    )

    extracted_quote_id: Mapped[int] = mapped_column(
        ForeignKey("extracted_quotes.id"),
        nullable=False,
        unique=True,
    )

    rfx_id: Mapped[str] = mapped_column(
        ForeignKey("rfx.rfx_id"),
        nullable=False,
    )

    submission_id: Mapped[str] = mapped_column(
        ForeignKey("vendor_submissions.submission_id"),
        nullable=False,
    )

    canonical_item_id: Mapped[str | None] = mapped_column(String(100))

    raw_price: Mapped[float | None] = mapped_column(Float)
    raw_unit: Mapped[str | None] = mapped_column(String(255))
    raw_currency: Mapped[str | None] = mapped_column(String(10))

    inr_price_before_unit_conversion: Mapped[float | None] = mapped_column(Float)
    normalized_inr_per_sheet: Mapped[float | None] = mapped_column(Float)

    currency_formula: Mapped[str | None] = mapped_column(Text)
    unit_formula: Mapped[str | None] = mapped_column(Text)
    normalization_formula: Mapped[str | None] = mapped_column(Text)

    comparable: Mapped[bool] = mapped_column(Boolean, nullable=False)
    review_required: Mapped[bool] = mapped_column(Boolean, nullable=False)
    award_eligible: Mapped[bool] = mapped_column(Boolean, nullable=False)

    freight_status: Mapped[str] = mapped_column(String(30), nullable=False)
    item_match_method: Mapped[str] = mapped_column(String(100), nullable=False)
    normalization_reasons: Mapped[str] = mapped_column(Text, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

class AwardRunRecord(Base):
    __tablename__ = "award_runs"

    award_run_id: Mapped[str] = mapped_column(
        String(40),
        primary_key=True,
    )

    rfx_id: Mapped[str] = mapped_column(
        ForeignKey("rfx.rfx_id"),
        nullable=False,
    )

    run_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    filters_used: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    total_awarded_spend: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )


class AwardLineDecisionRecord(Base):
    __tablename__ = "award_line_decisions"

    award_line_decision_id: Mapped[str] = mapped_column(
        String(40),
        primary_key=True,
    )

    award_run_id: Mapped[str] = mapped_column(
        ForeignKey("award_runs.award_run_id"),
        nullable=False,
    )

    item_id: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    description: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    annual_quantity_sheets: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    winner_vendor_name: Mapped[str | None] = mapped_column(
        String(255)
    )

    selected_price_inr_per_sheet: Mapped[float | None] = mapped_column(
        Float
    )

    annual_line_cost: Mapped[float | None] = mapped_column(Float)

    decision_status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    rationale: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    excluded_vendors_json: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

class QuestionnaireAnswerRecord(Base):
    __tablename__ = "questionnaire_answers"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    submission_id: Mapped[str] = mapped_column(
        ForeignKey("vendor_submissions.submission_id"),
        nullable=False,
    )
    document_id: Mapped[str] = mapped_column(
        ForeignKey("source_documents.document_id"),
        nullable=False,
    )

    question_id: Mapped[str | None] = mapped_column(String(50))
    raw_answer: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_answer: Mapped[str | None] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    evidence_text: Mapped[str] = mapped_column(Text, nullable=False)
    page_or_section: Mapped[str | None] = mapped_column(String(255))

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    submission: Mapped["VendorSubmissionRecord"] = relationship(
        back_populates="questionnaire_answers"
    )
    source_document: Mapped["SourceDocumentRecord"] = relationship(
        back_populates="questionnaire_answers"
    )




class ReviewTaskRecord(Base):
    __tablename__ = "review_tasks"

    task_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    rfx_id: Mapped[str] = mapped_column(
        ForeignKey("rfx.rfx_id"),
        nullable=False,
    )
    submission_id: Mapped[str | None] = mapped_column(
        ForeignKey("vendor_submissions.submission_id")
    )
    document_id: Mapped[str | None] = mapped_column(
        ForeignKey("source_documents.document_id")
    )

    issue_type: Mapped[str] = mapped_column(String(100), nullable=False)
    severity: Mapped[str] = mapped_column(String(10), nullable=False)
    affected_item_id: Mapped[str | None] = mapped_column(String(100))
    evidence_text: Mapped[str] = mapped_column(Text, nullable=False)
    current_value: Mapped[str | None] = mapped_column(Text)
    suggested_resolution: Mapped[str] = mapped_column(Text, nullable=False)

    status: Mapped[str] = mapped_column(String(30), nullable=False)
    buyer_resolution: Mapped[str | None] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    resolved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )

    rfx: Mapped["RFxRecord"] = relationship(back_populates="review_tasks")
    submission: Mapped["VendorSubmissionRecord | None"] = relationship(
        back_populates="review_tasks"
    )
    source_document: Mapped["SourceDocumentRecord | None"] = relationship(
        back_populates="review_tasks"
    )
    actions: Mapped[list["ReviewTaskActionRecord"]] = relationship(
        back_populates="review_task",
        cascade="all, delete-orphan",
        order_by="ReviewTaskActionRecord.created_at",
    )

class ReviewTaskActionRecord(Base):
    """
    Immutable audit history of buyer review decisions.

    This table never overwrites Gemini extraction evidence.
    """

    __tablename__ = "review_task_actions"

    action_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    task_id: Mapped[str] = mapped_column(
        ForeignKey("review_tasks.task_id"),
        nullable=False,
    )

    action: Mapped[str] = mapped_column(String(50), nullable=False)
    corrected_value: Mapped[str | None] = mapped_column(Text)
    resolution_reason: Mapped[str] = mapped_column(Text, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    review_task: Mapped["ReviewTaskRecord"] = relationship(
        back_populates="actions"
    )

# -------------------------------------------------------------------
# Database configuration
# -------------------------------------------------------------------
DATABASE_FILE.parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    f"sqlite:///{DATABASE_FILE}",
    connect_args={"check_same_thread": False},
)


def initialize_database() -> None:
    """Create missing SQLite tables."""

    Base.metadata.create_all(engine)


# -------------------------------------------------------------------
# Step 16: Approved RFx functions
# -------------------------------------------------------------------
def save_approved_rfx(
    *,
    draft: GeneratedRFxDraft,
    buyer_request: str,
    assumption_decisions: list[AssumptionDecision],
) -> str:
    """Save one buyer-approved RFx and its child records."""

    rfx_id = f"RFX-{uuid4().hex[:12].upper()}"
    now = datetime.now(timezone.utc)

    with Session(engine) as session:
        try:
            session.add(
                RFxRecord(
                    rfx_id=rfx_id,
                    title=draft.title,
                    buyer_request=buyer_request,
                    scope=draft.scope,
                    commercial_terms=draft.commercial_terms,
                    status="approved",
                    created_at=now,
                    approved_at=now,
                )
            )

            for item in draft.line_items:
                session.add(
                    LineItemRecord(
                        rfx_id=rfx_id,
                        **item.model_dump(),
                    )
                )

            for question in draft.questionnaire:
                session.add(
                    QuestionnaireQuestionRecord(
                        rfx_id=rfx_id,
                        **question.model_dump(),
                    )
                )

            for assumption in assumption_decisions:
                session.add(
                    RFxAssumptionRecord(
                        rfx_id=rfx_id,
                        assumption_text=assumption.assumption_text,
                        accepted=assumption.accepted,
                    )
                )

            session.commit()
            return rfx_id

        except Exception:
            session.rollback()
            raise


def get_rfx(rfx_id: str) -> dict | None:
    """Return one RFx and its approved line items, questions, and assumptions."""

    with Session(engine) as session:
        rfx = session.get(RFxRecord, rfx_id)

        if rfx is None:
            return None

        return {
            "rfx_id": rfx.rfx_id,
            "title": rfx.title,
            "buyer_request": rfx.buyer_request,
            "scope": rfx.scope,
            "commercial_terms": rfx.commercial_terms,
            "status": rfx.status,
            "created_at": rfx.created_at,
            "approved_at": rfx.approved_at,
            "line_items": [
                {
                    "item_id": item.item_id,
                    "description": item.description,
                    "ply": item.ply,
                    "flute": item.flute,
                    "outer_gsm": item.outer_gsm,
                    "dimensions_mm": item.dimensions_mm,
                    "print_requirement": item.print_requirement,
                    "moisture_resistant": item.moisture_resistant,
                    "annual_quantity_sheets": item.annual_quantity_sheets,
                    "standard_unit": item.standard_unit,
                    "delivery_location": item.delivery_location,
                }
                for item in rfx.line_items
            ],
            "questionnaire": [
                {
                    "question_id": question.question_id,
                    "question": question.question,
                    "answer_type": question.answer_type,
                    "required": question.required,
                    "quality_gate": question.quality_gate,
                }
                for question in rfx.questionnaire_questions
            ],
            "assumptions": [
                {
                    "assumption_text": assumption.assumption_text,
                    "accepted": assumption.accepted,
                }
                for assumption in rfx.assumptions
            ],
        }


def get_latest_approved_rfx() -> dict | None:
    """Return the newest approved RFx."""

    with Session(engine) as session:
        rfx_id = session.scalar(
            select(RFxRecord.rfx_id)
            .where(RFxRecord.status == "approved")
            .order_by(RFxRecord.approved_at.desc())
            .limit(1)
        )

    return get_rfx(rfx_id) if rfx_id else None


def list_approved_rfx() -> list[dict]:
    """Return all approved RFxs, newest approval first."""

    with Session(engine) as session:
        rfx_ids = session.scalars(
            select(RFxRecord.rfx_id)
            .where(RFxRecord.status == "approved")
            .order_by(RFxRecord.approved_at.desc())
        ).all()

    approved_rfxs = []

    for rfx_id in rfx_ids:
        rfx = get_rfx(rfx_id)

        if rfx is not None:
            approved_rfxs.append(rfx)

    return approved_rfxs


# -------------------------------------------------------------------
# Step 18: Ingestion functions
# -------------------------------------------------------------------
def get_document_by_checksum(
    *,
    rfx_id: str,
    sha256_checksum: str,
) -> dict | None:
    """Return a previously ingested document with this checksum for this RFx."""

    with Session(engine) as session:
        document = session.scalar(
            select(SourceDocumentRecord)
            .join(VendorSubmissionRecord)
            .where(
                VendorSubmissionRecord.rfx_id == rfx_id,
                SourceDocumentRecord.sha256_checksum == sha256_checksum,
            )
        )

        if document is None:
            return None

        return {
            "document_id": document.document_id,
            "submission_id": document.submission_id,
            "original_filename": document.original_filename,
            "status": document.submission.status,
        }


def start_document_ingestion(
    *,
    rfx_id: str,
    prepared_document: PreparedSourceDocument,
) -> IngestionStarted:
    """
    Create a submission and source-document record before Gemini extraction.

    The document remains traceable even if Gemini extraction later fails.
    """

    with Session(engine) as session:
        try:
            rfx = session.get(RFxRecord, rfx_id)

            if rfx is None:
                raise DatabaseError(f"RFx does not exist: {rfx_id}")

            if rfx.status != "approved":
                raise DatabaseError(
                    "Only an approved RFx can receive vendor submissions."
                )

            existing_document = session.scalar(
                select(SourceDocumentRecord)
                .join(VendorSubmissionRecord)
                .where(
                    VendorSubmissionRecord.rfx_id == rfx_id,
                    SourceDocumentRecord.sha256_checksum
                    == prepared_document.sha256_checksum,
                )
            )

            if existing_document is not None:
                raise DuplicateDocumentError(
                    f"{prepared_document.filename} was already ingested "
                    f"for RFx {rfx_id}."
                )

            now = datetime.now(timezone.utc)
            submission_id = f"SUB-{uuid4().hex[:12].upper()}"
            document_id = f"DOC-{uuid4().hex[:12].upper()}"

            session.add(
                VendorSubmissionRecord(
                    submission_id=submission_id,
                    rfx_id=rfx_id,
                    vendor_id=None,
                    received_at=now,
                    status="extracting",
                )
            )

            session.add(
                SourceDocumentRecord(
                    document_id=document_id,
                    submission_id=submission_id,
                    original_filename=prepared_document.filename,
                    original_path=str(prepared_document.original_path),
                    processed_path=str(prepared_document.processed_path),
                    file_type=prepared_document.extension,
                    file_size_bytes=prepared_document.file_size_bytes,
                    sha256_checksum=prepared_document.sha256_checksum,
                    extracted_text=prepared_document.baseline_content,
                    extraction_json=None,
                    created_at=now,
                )
            )

            session.commit()

            return IngestionStarted(
                submission_id=submission_id,
                document_id=document_id,
            )

        except Exception:
            session.rollback()
            raise


def _get_or_create_vendor(
    session: Session,
    vendor_name: str,
) -> VendorRecord:
    """Use an existing vendor record or create one."""

    vendor = session.scalar(
        select(VendorRecord).where(VendorRecord.name == vendor_name)
    )

    if vendor is not None:
        return vendor

    vendor = VendorRecord(
        vendor_id=f"VND-{uuid4().hex[:12].upper()}",
        name=vendor_name,
        created_at=datetime.now(timezone.utc),
    )

    session.add(vendor)
    return vendor


def save_document_extraction(
    *,
    rfx_id: str,
    submission_id: str,
    document_id: str,
    extraction: VendorDocumentExtraction,
    review_candidates: list[ReviewCandidate],
) -> None:
    """
    Save Gemini extraction results, evidence, and generated review tasks.

    This function does not overwrite prior extraction evidence. A submission
    can be saved only once while its status is 'extracting'.
    """

    with Session(engine) as session:
        try:
            submission = session.get(VendorSubmissionRecord, submission_id)
            document = session.get(SourceDocumentRecord, document_id)

            if submission is None:
                raise DatabaseError(f"Submission does not exist: {submission_id}")

            if document is None:
                raise DatabaseError(f"Document does not exist: {document_id}")

            if submission.rfx_id != rfx_id:
                raise DatabaseError("Submission does not belong to this RFx.")

            if document.submission_id != submission_id:
                raise DatabaseError(
                    "Document does not belong to this vendor submission."
                )

            if submission.status != "extracting":
                raise DatabaseError(
                    f"Cannot save extraction when submission status is "
                    f"'{submission.status}'."
                )

            if extraction.vendor_name:
                vendor = _get_or_create_vendor(
                    session,
                    extraction.vendor_name.strip(),
                )
                submission.vendor_id = vendor.vendor_id

            document.extraction_json = extraction.model_dump_json(indent=2)

            now = datetime.now(timezone.utc)

            for quote in extraction.quotes:
                session.add(
                    ExtractedQuoteRecord(
                        submission_id=submission_id,
                        document_id=document_id,
                        canonical_item_id=quote.canonical_item_id,
                        raw_description=quote.raw_description,
                        raw_price=quote.raw_price,
                        raw_unit=quote.raw_unit,
                        currency=quote.currency,
                        confidence=quote.confidence,
                        evidence_text=quote.evidence_text,
                        page_or_section=quote.page_or_section,
                        extraction_note=quote.extraction_note,
                        created_at=now,
                    )
                )

            for answer in extraction.questionnaire_answers:
                session.add(
                    QuestionnaireAnswerRecord(
                        submission_id=submission_id,
                        document_id=document_id,
                        question_id=answer.question_id,
                        raw_answer=answer.raw_answer,
                        normalized_answer=answer.normalized_answer,
                        confidence=answer.confidence,
                        evidence_text=answer.evidence_text,
                        page_or_section=answer.page_or_section,
                        created_at=now,
                    )
                )

            for task in review_candidates:
                session.add(
                    ReviewTaskRecord(
                        task_id=f"REV-{uuid4().hex[:12].upper()}",
                        rfx_id=rfx_id,
                        submission_id=submission_id,
                        document_id=document_id,
                        issue_type=task.issue_type,
                        severity=task.severity,
                        affected_item_id=task.affected_item_id,
                        evidence_text=task.evidence_text,
                        current_value=task.current_value,
                        suggested_resolution=task.suggested_resolution,
                        status="open",
                        buyer_resolution=None,
                        created_at=now,
                        resolved_at=None,
                    )
                )

            submission.status = (
                "needs_review" if review_candidates else "extracted"
            )

            session.commit()

        except Exception:
            session.rollback()
            raise


def mark_submission_failed(
    *,
    submission_id: str,
    reason: str,
) -> None:
    """Preserve the document and record a failed extraction status."""

    with Session(engine) as session:
        try:
            submission = session.get(VendorSubmissionRecord, submission_id)

            if submission is None:
                raise DatabaseError(f"Submission does not exist: {submission_id}")

            submission.status = "failed"

            session.add(
                ReviewTaskRecord(
                    task_id=f"REV-{uuid4().hex[:12].upper()}",
                    rfx_id=submission.rfx_id,
                    submission_id=submission_id,
                    document_id=None,
                    issue_type="AMBIGUOUS_COMMERCIAL_TERM",
                    severity="Red",
                    affected_item_id=None,
                    evidence_text=f"Document extraction failed: {reason}",
                    current_value=None,
                    suggested_resolution=(
                        "Review the source document and retry extraction "
                        "after correcting the issue."
                    ),
                    status="open",
                    buyer_resolution=None,
                    created_at=datetime.now(timezone.utc),
                    resolved_at=None,
                )
            )

            session.commit()

        except Exception:
            session.rollback()
            raise


def list_submissions_for_rfx(rfx_id: str) -> list[dict]:
    """Return vendor-submission rows for the Inbox screen."""

    with Session(engine) as session:
        submissions = session.scalars(
            select(VendorSubmissionRecord)
            .where(VendorSubmissionRecord.rfx_id == rfx_id)
            .order_by(VendorSubmissionRecord.received_at.desc())
        ).all()

        return [
            {
                "submission_id": submission.submission_id,
                "vendor_name": (
                    submission.vendor.name
                    if submission.vendor is not None
                    else "Unknown"
                ),
                "status": submission.status,
                "received_at": submission.received_at,
                "source_documents": [
                    {
                        "document_id": document.document_id,
                        "filename": document.original_filename,
                        "file_type": document.file_type,
                        "checksum": document.sha256_checksum,
                    }
                    for document in submission.source_documents
                ],
            }
            for submission in submissions
        ]
def _review_task_to_dict(task: ReviewTaskRecord) -> dict:
    """Convert one review task and its evidence into UI-friendly data."""

    vendor_name = "Unknown"

    if task.submission is not None and task.submission.vendor is not None:
        vendor_name = task.submission.vendor.name

    source_filename = None

    if task.source_document is not None:
        source_filename = task.source_document.original_filename

    return {
        "task_id": task.task_id,
        "rfx_id": task.rfx_id,
        "submission_id": task.submission_id,
        "document_id": task.document_id,
        "vendor_name": vendor_name,
        "source_filename": source_filename,
        "issue_type": task.issue_type,
        "severity": task.severity,
        "affected_item_id": task.affected_item_id,
        "evidence_text": task.evidence_text,
        "current_value": task.current_value,
        "suggested_resolution": task.suggested_resolution,
        "status": task.status,
        "buyer_resolution": task.buyer_resolution,
        "created_at": task.created_at,
        "resolved_at": task.resolved_at,
    }


def list_review_tasks(
    *,
    rfx_id: str,
    statuses: list[str] | None = None,
) -> list[dict]:
    """
    Return review tasks for the Needs Review screen.

    Default: open and clarification-needed tasks only.
    """

    if statuses is None:
        statuses = ["open", "clarification_needed"]

    with Session(engine) as session:
        tasks = session.scalars(
            select(ReviewTaskRecord)
            .where(
                ReviewTaskRecord.rfx_id == rfx_id,
                ReviewTaskRecord.status.in_(statuses),
            )
            .order_by(
                ReviewTaskRecord.severity.desc(),
                ReviewTaskRecord.created_at.asc(),
            )
        ).all()

        return [_review_task_to_dict(task) for task in tasks]


def get_review_task_history(task_id: str) -> list[dict]:
    """Return the immutable buyer-action history for one review task."""

    with Session(engine) as session:
        actions = session.scalars(
            select(ReviewTaskActionRecord)
            .where(ReviewTaskActionRecord.task_id == task_id)
            .order_by(ReviewTaskActionRecord.created_at.asc())
        ).all()

        return [
            {
                "action_id": action.action_id,
                "action": action.action,
                "corrected_value": action.corrected_value,
                "resolution_reason": action.resolution_reason,
                "created_at": action.created_at,
            }
            for action in actions
        ]


def resolve_review_task(
    *,
    task_id: str,
    action: str,
    resolution_reason: str,
    corrected_value: str | None = None,
) -> None:
    """
    Save a buyer decision without overwriting original extracted data.

    Valid actions:
      Approve
      Correct
      Mark missing
      Exclude
      Clarification needed
    """

    allowed_actions = {
        "Approve",
        "Correct",
        "Mark missing",
        "Exclude",
        "Clarification needed",
    }

    if action not in allowed_actions:
        raise DatabaseError(f"Unsupported review action: {action}")

    if not resolution_reason.strip():
        raise DatabaseError("A resolution reason is required.")

    if action == "Correct" and not (corrected_value or "").strip():
        raise DatabaseError(
            "Enter the corrected value before saving a correction."
        )

    status_by_action = {
        "Approve": "resolved",
        "Correct": "resolved",
        "Mark missing": "resolved",
        "Exclude": "excluded",
        "Clarification needed": "clarification_needed",
    }

    with Session(engine) as session:
        try:
            task = session.get(ReviewTaskRecord, task_id)

            if task is None:
                raise DatabaseError(f"Review task does not exist: {task_id}")

            now = datetime.now(timezone.utc)

            session.add(
                ReviewTaskActionRecord(
                    action_id=f"ACT-{uuid4().hex[:12].upper()}",
                    task_id=task_id,
                    action=action,
                    corrected_value=(
                        corrected_value.strip()
                        if corrected_value
                        else None
                    ),
                    resolution_reason=resolution_reason.strip(),
                    created_at=now,
                )
            )

            task.status = status_by_action[action]
            task.buyer_resolution = resolution_reason.strip()

            if action in {"Approve", "Correct", "Mark missing", "Exclude"}:
                task.resolved_at = now
            else:
                task.resolved_at = None

            session.commit()

        except Exception:
            session.rollback()
            raise
def normalize_submission_quotes(submission_id: str) -> int:
    """
    Normalize all extracted quotes for one vendor submission.

    Returns the number of normalized quote records created.
    """

    with Session(engine) as session:
        try:
            submission = session.get(VendorSubmissionRecord, submission_id)

            if submission is None:
                raise DatabaseError(
                    f"Submission does not exist: {submission_id}"
                )

            rfx = session.get(RFxRecord, submission.rfx_id)

            if rfx is None:
                raise DatabaseError(
                    f"RFx does not exist: {submission.rfx_id}"
                )

            canonical_line_items = [
                {
                    "item_id": item.item_id,
                    "description": item.description,
                    "ply": item.ply,
                    "flute": item.flute,
                    "outer_gsm": item.outer_gsm,
                    "dimensions_mm": item.dimensions_mm,
                    "print_requirement": item.print_requirement,
                    "moisture_resistant": item.moisture_resistant,
                    "annual_quantity_sheets": item.annual_quantity_sheets,
                    "standard_unit": item.standard_unit,
                    "delivery_location": item.delivery_location,
                }
                for item in rfx.line_items
            ]

            extracted_quotes = session.scalars(
                select(ExtractedQuoteRecord).where(
                    ExtractedQuoteRecord.submission_id == submission_id
                )
            ).all()

            source_document = session.scalar(
                select(SourceDocumentRecord)
                .where(
                    SourceDocumentRecord.submission_id == submission_id
                )
                .order_by(SourceDocumentRecord.created_at.asc())
                .limit(1)
            )

            freight_status = "unknown"

            if source_document and source_document.extraction_json:
                try:
                    extraction_payload = json.loads(
                        source_document.extraction_json
                    )
                    freight_status = extraction_payload.get(
                        "freight_status",
                        "unknown",
                    )
                except json.JSONDecodeError:
                    freight_status = "unknown"

            if freight_status not in {
                "included",
                "extra",
                "unknown",
                "not_stated",
            }:
                freight_status = "unknown"

            count = 0

            for extracted_quote in extracted_quotes:
                existing_normalization = session.scalar(
                    select(NormalizedQuoteRecord).where(
                        NormalizedQuoteRecord.extracted_quote_id
                        == extracted_quote.id
                    )
                )

                if existing_normalization is not None:
                    continue

                raw_quote = RawQuoteInput(
                    canonical_item_id=extracted_quote.canonical_item_id,
                    raw_description=extracted_quote.raw_description,
                    raw_price=extracted_quote.raw_price,
                    raw_unit=extracted_quote.raw_unit,
                    currency=extracted_quote.currency,
                    confidence=extracted_quote.confidence,
                    freight_status=freight_status,
                )

                normalized = normalize_quote(
                    quote=raw_quote,
                    canonical_line_items=canonical_line_items,
                )

                session.add(
                    NormalizedQuoteRecord(
                        normalized_quote_id=(
                            f"NORM-{uuid4().hex[:12].upper()}"
                        ),
                        extracted_quote_id=extracted_quote.id,
                        rfx_id=submission.rfx_id,
                        submission_id=submission_id,
                        canonical_item_id=normalized.matched_item_id,
                        raw_price=(
                            float(normalized.raw_price)
                            if normalized.raw_price is not None
                            else None
                        ),
                        raw_unit=normalized.raw_unit,
                        raw_currency=normalized.raw_currency,
                        inr_price_before_unit_conversion=(
                            float(normalized.inr_price_before_unit_conversion)
                            if normalized.inr_price_before_unit_conversion
                            is not None
                            else None
                        ),
                        normalized_inr_per_sheet=(
                            float(normalized.normalized_inr_per_sheet)
                            if normalized.normalized_inr_per_sheet
                            is not None
                            else None
                        ),
                        currency_formula=normalized.currency_formula,
                        unit_formula=normalized.unit_formula,
                        normalization_formula=normalized.normalization_formula,
                        comparable=normalized.comparable,
                        review_required=normalized.review_required,
                        award_eligible=normalized.award_eligible,
                        freight_status=raw_quote.freight_status,
                        item_match_method=normalized.item_match_method,
                        normalization_reasons=json.dumps(
                            normalized.reasons
                        ),
                        created_at=datetime.now(timezone.utc),
                    )
                )

                count += 1

            session.commit()
            return count

        except Exception:
            session.rollback()
            raise


def normalize_all_quotes_for_rfx(rfx_id: str) -> int:
    """Normalize every extracted vendor quote for one approved RFx."""

    with Session(engine) as session:
        submission_ids = session.scalars(
            select(VendorSubmissionRecord.submission_id).where(
                VendorSubmissionRecord.rfx_id == rfx_id
            )
        ).all()

    total_count = 0

    for submission_id in submission_ids:
        total_count += normalize_submission_quotes(submission_id)

    return total_count


def get_comparison_matrix(rfx_id: str) -> list[dict]:
    """
    Return one flattened row per buyer item per vendor quote.

    test2.py will pivot this data into the visual comparison matrix.
    """

    with Session(engine) as session:
        rows = session.execute(
            select(
                LineItemRecord,
                NormalizedQuoteRecord,
                VendorSubmissionRecord,
                VendorRecord,
                ExtractedQuoteRecord,
                SourceDocumentRecord,
            )
            .outerjoin(
                NormalizedQuoteRecord,
                and_(
                    LineItemRecord.item_id
                    == NormalizedQuoteRecord.canonical_item_id,
                    NormalizedQuoteRecord.rfx_id == rfx_id,
                ),
            )
            .outerjoin(
                VendorSubmissionRecord,
                VendorSubmissionRecord.submission_id
                == NormalizedQuoteRecord.submission_id,
            )
            .outerjoin(
                VendorRecord,
                VendorRecord.vendor_id
                == VendorSubmissionRecord.vendor_id,
            )
            .outerjoin(
                ExtractedQuoteRecord,
                ExtractedQuoteRecord.id
                == NormalizedQuoteRecord.extracted_quote_id,
            )
            .outerjoin(
                SourceDocumentRecord,
                SourceDocumentRecord.document_id
                == ExtractedQuoteRecord.document_id,
            )
            .where(LineItemRecord.rfx_id == rfx_id)
            .order_by(
                LineItemRecord.item_id,
                VendorRecord.name,
            )
        ).all()

        output = []

        for (
            line_item,
            normalized_quote,
            submission,
            vendor,
            extracted_quote,
            source_document,
        ) in rows:
            output.append(
                {
                    "item_id": line_item.item_id,
                    "description": line_item.description,
                    "ply": line_item.ply,
                    "flute": line_item.flute,
                    "outer_gsm": line_item.outer_gsm,
                    "dimensions_mm": line_item.dimensions_mm,
                    "print_requirement": line_item.print_requirement,
                    "moisture_resistant": line_item.moisture_resistant,
                    "annual_quantity_sheets": (
                        line_item.annual_quantity_sheets
                    ),
                    "vendor_name": (
                        vendor.name if vendor is not None else "No quote"
                    ),
                    "normalized_inr_per_sheet": (
                        normalized_quote.normalized_inr_per_sheet
                        if normalized_quote is not None
                        else None
                    ),
                    "raw_price": (
                        normalized_quote.raw_price
                        if normalized_quote is not None
                        else None
                    ),
                    "raw_unit": (
                        normalized_quote.raw_unit
                        if normalized_quote is not None
                        else None
                    ),
                    "raw_currency": (
                        normalized_quote.raw_currency
                        if normalized_quote is not None
                        else None
                    ),
                    "confidence": (
                        extracted_quote.confidence
                        if extracted_quote is not None
                        else None
                    ),
                    "freight_status": (
                        normalized_quote.freight_status
                        if normalized_quote is not None
                        else None
                    ),
                    "comparable": (
                        normalized_quote.comparable
                        if normalized_quote is not None
                        else False
                    ),
                    "review_required": (
                        normalized_quote.review_required
                        if normalized_quote is not None
                        else True
                    ),
                    "award_eligible": (
                        normalized_quote.award_eligible
                        if normalized_quote is not None
                        else False
                    ),
                    "evidence_text": (
                        extracted_quote.evidence_text
                        if extracted_quote is not None
                        else None
                    ),
                    "source_filename": (
                        source_document.original_filename
                        if source_document is not None
                        else None
                    ),
                    "page_or_section": (
                        extracted_quote.page_or_section
                        if extracted_quote is not None
                        else None
                    ),
                    "normalization_formula": (
                        normalized_quote.normalization_formula
                        if normalized_quote is not None
                        else None
                    ),
                    "normalization_reasons": (
                        json.loads(normalized_quote.normalization_reasons)
                        if normalized_quote is not None
                        else ["No quote available"]
                    ),
                }
            )

        return output
def get_vendor_qualification(
    *,
    rfx_id: str,
    vendor_name: str | None = None,
) -> list[dict]:
    """
    Determine qualification from Q01-Q04 answers only.

    A vendor is qualified only when all four answers are Yes.
    """

    required_quality_questions = {"Q01", "Q02", "Q03", "Q04"}

    with Session(engine) as session:
        submissions = session.scalars(
            select(VendorSubmissionRecord)
            .where(VendorSubmissionRecord.rfx_id == rfx_id)
            .order_by(VendorSubmissionRecord.received_at.desc())
        ).all()

        results = []

        for submission in submissions:
            current_vendor_name = (
                submission.vendor.name
                if submission.vendor is not None
                else "Unknown"
            )

            if vendor_name and current_vendor_name.lower() != vendor_name.lower():
                continue

            answers = session.scalars(
                select(QuestionnaireAnswerRecord).where(
                    QuestionnaireAnswerRecord.submission_id
                    == submission.submission_id
                )
            ).all()

            answers_by_question = {
                answer.question_id: (answer.normalized_answer or "").strip().lower()
                for answer in answers
                if answer.question_id
            }

            missing_questions = [
                question_id
                for question_id in required_quality_questions
                if question_id not in answers_by_question
            ]

            failed_questions = [
                question_id
                for question_id in required_quality_questions
                if question_id in answers_by_question
                and answers_by_question[question_id] != "yes"
            ]

            qualified = not missing_questions and not failed_questions

            results.append(
                {
                    "vendor_name": current_vendor_name,
                    "submission_id": submission.submission_id,
                    "qualified": qualified,
                    "quality_answers": {
                        question_id: answers_by_question.get(question_id, "Missing")
                        for question_id in sorted(required_quality_questions)
                    },
                    "missing_questions": missing_questions,
                    "failed_questions": failed_questions,
                }
            )

        return results


def get_missing_quotes(
    *,
    rfx_id: str,
    vendor_name: str | None = None,
) -> list[dict]:
    """Return missing or non-comparable quotes by vendor and line item."""

    matrix_rows = get_comparison_matrix(rfx_id)
    results = []

    for row in matrix_rows:
        if vendor_name and row["vendor_name"].lower() != vendor_name.lower():
            continue

        if (
            row["vendor_name"] == "No quote"
            or not row["comparable"]
            or row["normalized_inr_per_sheet"] is None
        ):
            results.append(
                {
                    "vendor_name": row["vendor_name"],
                    "item_id": row["item_id"],
                    "description": row["description"],
                    "reason": " | ".join(row["normalization_reasons"]),
                }
            )

    return results


def get_discount_evidence(
    *,
    rfx_id: str,
    vendor_name: str | None = None,
) -> list[dict]:
    """Return conditional-discount review tasks and their evidence."""

    with Session(engine) as session:
        tasks = session.scalars(
            select(ReviewTaskRecord)
            .where(
                ReviewTaskRecord.rfx_id == rfx_id,
                ReviewTaskRecord.issue_type == "CONDITIONAL_DISCOUNT",
            )
            .order_by(ReviewTaskRecord.created_at.desc())
        ).all()

        results = []

        for task in tasks:
            current_vendor_name = (
                task.submission.vendor.name
                if task.submission is not None
                and task.submission.vendor is not None
                else "Unknown"
            )

            if vendor_name and current_vendor_name.lower() != vendor_name.lower():
                continue

            results.append(
                {
                    "vendor_name": current_vendor_name,
                    "source_file": (
                        task.source_document.original_filename
                        if task.source_document is not None
                        else "Not available"
                    ),
                    "evidence": task.evidence_text,
                    "status": task.status,
                    "buyer_resolution": task.buyer_resolution,
                }
            )

        return results


def get_vendor_exclusion_reasons(
    *,
    rfx_id: str,
    item_id: str,
    vendor_name: str,
) -> list[str]:
    """Return deterministic reasons a vendor is not award-eligible."""

    matrix_rows = get_comparison_matrix(rfx_id)

    matching_rows = [
        row
        for row in matrix_rows
        if row["item_id"] == item_id
        and row["vendor_name"].lower() == vendor_name.lower()
    ]

    if not matching_rows:
        return ["No quote was found for this vendor and item."]

    row = matching_rows[0]
    reasons = list(row["normalization_reasons"])

    if not row["comparable"]:
        reasons.append("Quote is not comparable as INR per sheet.")

    if row["review_required"]:
        reasons.append("Quote has unresolved review requirements.")

    if not row["award_eligible"]:
        reasons.append("Quote is not currently award eligible.")

    return list(dict.fromkeys(reasons))


def get_review_summary(rfx_id: str) -> dict:
    """Return deterministic counts for the Needs Review queue."""

    with Session(engine) as session:
        tasks = session.scalars(
            select(ReviewTaskRecord).where(
                ReviewTaskRecord.rfx_id == rfx_id
            )
        ).all()

        return {
            "total": len(tasks),
            "open": sum(task.status == "open" for task in tasks),
            "clarification_needed": sum(
                task.status == "clarification_needed"
                for task in tasks
            ),
            "resolved": sum(task.status == "resolved" for task in tasks),
            "excluded": sum(task.status == "excluded" for task in tasks),
        }  

def save_award_run(award_result: dict) -> str:
    """Save one recommendation run; this does not award any line item."""

    award_run_id = f"AWD-{uuid4().hex[:12].upper()}"
    now = datetime.now(timezone.utc)

    with Session(engine) as session:
        try:
            session.add(
                AwardRunRecord(
                    award_run_id=award_run_id,
                    rfx_id=award_result["rfx_id"],
                    run_timestamp=now,
                    filters_used=json.dumps(
                        {
                            "quality_gate": "Q01-Q04 must all be Yes",
                            "comparison_basis": "INR per sheet, excluding GST",
                            "freight_required": "included",
                            "review_status": "no unresolved review required",
                            "tie_rule": "buyer decision required",
                        }
                    ),
                    total_awarded_spend=award_result["total_awarded_spend"],
                )
            )

            for decision in award_result["decisions"]:
                session.add(
                    AwardLineDecisionRecord(
                        award_line_decision_id=(
                            f"AWDLINE-{uuid4().hex[:12].upper()}"
                        ),
                        award_run_id=award_run_id,
                        item_id=decision["item_id"],
                        description=decision["description"],
                        annual_quantity_sheets=(
                            decision["annual_quantity_sheets"]
                        ),
                        winner_vendor_name=decision["winner_vendor_name"],
                        selected_price_inr_per_sheet=(
                            decision["selected_price_inr_per_sheet"]
                        ),
                        annual_line_cost=decision["annual_line_cost"],
                        decision_status=decision["decision_status"],
                        rationale=decision["rationale"],
                        excluded_vendors_json=json.dumps(
                            decision["excluded_vendors"]
                        ),
                    )
                )

            session.commit()
            return award_run_id

        except Exception:
            session.rollback()
            raise


def get_latest_award_run(rfx_id: str) -> dict | None:
    """Return the most recently saved award run for an RFx."""

    with Session(engine) as session:
        award_run = session.scalar(
            select(AwardRunRecord)
            .where(AwardRunRecord.rfx_id == rfx_id)
            .order_by(AwardRunRecord.run_timestamp.desc())
            .limit(1)
        )

        if award_run is None:
            return None

        decisions = session.scalars(
            select(AwardLineDecisionRecord)
            .where(
                AwardLineDecisionRecord.award_run_id
                == award_run.award_run_id
            )
            .order_by(AwardLineDecisionRecord.item_id)
        ).all()

        return {
            "award_run_id": award_run.award_run_id,
            "rfx_id": award_run.rfx_id,
            "run_timestamp": award_run.run_timestamp,
            "filters_used": json.loads(award_run.filters_used),
            "total_awarded_spend": award_run.total_awarded_spend,
            "decisions": [
                {
                    "item_id": decision.item_id,
                    "description": decision.description,
                    "annual_quantity_sheets": (
                        decision.annual_quantity_sheets
                    ),
                    "winner_vendor_name": decision.winner_vendor_name,
                    "selected_price_inr_per_sheet": (
                        decision.selected_price_inr_per_sheet
                    ),
                    "annual_line_cost": decision.annual_line_cost,
                    "decision_status": decision.decision_status,
                    "rationale": decision.rationale,
                    "excluded_vendors": json.loads(
                        decision.excluded_vendors_json
                    ),
                }
                for decision in decisions
            ],
        }


def get_award_eligible_vendors_by_item(rfx_id: str) -> dict[str, list[dict]]:
    """Return current eligible vendor choices for each RFx line item."""

    choices: dict[str, list[dict]] = {}

    for row in get_comparison_matrix(rfx_id):
        vendor_name = row["vendor_name"]
        price = row["normalized_inr_per_sheet"]

        if (
            vendor_name == "No quote"
            or not row["award_eligible"]
            or price is None
        ):
            continue

        item_choices = choices.setdefault(row["item_id"], [])
        existing = next(
            (
                choice
                for choice in item_choices
                if choice["vendor_name"] == vendor_name
            ),
            None,
        )

        if existing is None or price < existing["price_inr_per_sheet"]:
            if existing is not None:
                item_choices.remove(existing)
            item_choices.append(
                {
                    "vendor_name": vendor_name,
                    "price_inr_per_sheet": price,
                }
            )

    for item_choices in choices.values():
        item_choices.sort(
            key=lambda choice: (
                Decimal(str(choice["price_inr_per_sheet"])),
                choice["vendor_name"],
            )
        )

    return choices


def confirm_award_selections(
    award_run_id: str,
    selections: dict[str, str],
) -> int:
    """Persist only the buyer-selected supplier awards for a recommendation."""

    with Session(engine) as session:
        try:
            award_run = session.get(AwardRunRecord, award_run_id)
            if award_run is None:
                raise DatabaseError("Award recommendation was not found.")

            eligible_choices = get_award_eligible_vendors_by_item(
                award_run.rfx_id
            )
            decisions = session.scalars(
                select(AwardLineDecisionRecord).where(
                    AwardLineDecisionRecord.award_run_id == award_run_id
                )
            ).all()

            if not selections:
                raise DatabaseError(
                    "Select at least one supplier and mark its line for "
                    "confirmation."
                )

            decisions_by_item = {
                decision.item_id: decision for decision in decisions
            }
            confirmed_count = 0

            for item_id, selected_vendor in selections.items():
                decision = decisions_by_item.get(item_id)
                if decision is None:
                    raise DatabaseError(
                        f"Award line was not found: {item_id}."
                    )

                if decision.decision_status not in {
                    "recommended",
                    "tie_requires_buyer_decision",
                }:
                    raise DatabaseError(
                        f"{item_id} is not awaiting buyer confirmation."
                    )

                selected_choice = next(
                    (
                        choice
                        for choice in eligible_choices.get(
                            item_id,
                            []
                        )
                        if choice["vendor_name"] == selected_vendor
                    ),
                    None,
                )

                if selected_choice is None:
                    raise DatabaseError(
                        f"{selected_vendor} is not currently eligible for "
                        f"{decision.item_id}. Refresh the recommendation and "
                        "select an eligible vendor."
                    )

                price = Decimal(
                    str(selected_choice["price_inr_per_sheet"])
                )
                annual_line_cost = price * Decimal(
                    str(decision.annual_quantity_sheets)
                )

                decision.winner_vendor_name = selected_vendor
                decision.selected_price_inr_per_sheet = float(price)
                decision.annual_line_cost = float(annual_line_cost)
                decision.decision_status = "awarded"
                decision.rationale = (
                    "Buyer confirmed this eligible quote. "
                    "Comparison basis: INR per sheet, excluding GST."
                )
                confirmed_count += 1

            total_awarded_spend = sum(
                (
                    Decimal(str(decision.annual_line_cost))
                    for decision in decisions
                    if decision.decision_status == "awarded"
                    and decision.annual_line_cost is not None
                ),
                Decimal("0"),
            )

            award_run.total_awarded_spend = float(total_awarded_spend)
            session.commit()
            return confirmed_count

        except Exception:
            session.rollback()
            raise
