# RFx Copilot — Corrugated Sheets

A free-tier local prototype for the Aerchain take-home assignment. It will draft an RFx, ingest five heterogeneous supplier responses, normalize commercial terms, surface uncertainty with evidence, and answer procurement questions in natural language.

## Current milestone

The canonical RFx dataset is complete: 30 corrugated-sheet line items, quality questionnaire, commercial terms, and five vendor-response scenarios. The next milestone builds the AI-assisted RFx drafting experience and persists the approved RFx into SQLite.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The project will use a Gemini Developer API key for real structured extraction and real query interpretation. Save it only in `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "AIzaSyDwqnWk3SCDSh0vhGzn9Rd2WuL1r9TUtfo"
```

Do not add real supplier documents or API keys to source control.

## Design rules

- Canonical prices are INR per sheet, excluding GST.
- Every extracted value will retain raw evidence, confidence, and review state.
- Quality qualification is deterministic: questions Q01–Q04 must all be `Yes`.
- The analyst may interpret questions, but Python/SQLite will calculate award decisions.
- A red/amber review state must be visible before a buyer relies on an uncertain value.

