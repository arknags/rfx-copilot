from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class GeminiClientError(RuntimeError):
    """Raised when Gemini cannot return a valid structured result."""


def _validate_configuration(api_key: str, model: str) -> None:
    if not api_key or not api_key.strip():
        raise GeminiClientError("GEMINI_API_KEY is missing.")

    if not model or not model.strip():
        raise GeminiClientError("GEMINI_MODEL is missing.")


def _parse_response(response_text: str | None, response_model: type[T]) -> T:
    if not response_text:
        raise GeminiClientError("Gemini returned an empty response.")

    try:
        return response_model.model_validate_json(response_text)
    except Exception as error:
        raise GeminiClientError(
            f"Gemini returned JSON that does not match the expected schema: {error}"
        ) from error


def _gemini_response_schema(response_model: type[T]) -> dict[str, Any]:
    """Remove JSON Schema keywords unsupported by this Gemini SDK/API path.

    Pydantic's ``extra='forbid'`` produces ``additionalProperties: false``.
    Keep that strictness when parsing the returned JSON, but omit the keyword
    from the schema sent to Gemini because its response-schema subset rejects it.
    """

    def remove_unsupported_keywords(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: remove_unsupported_keywords(item)
                for key, item in value.items()
                if key not in {"additionalProperties", "additional_properties"}
            }

        if isinstance(value, list):
            return [remove_unsupported_keywords(item) for item in value]

        return value

    return remove_unsupported_keywords(response_model.model_json_schema())


def generate_structured(
    *,
    api_key: str,
    model: str,
    prompt: str,
    response_model: type[T],
) -> T:
    """Generate a Pydantic-validated structured response from text."""

    _validate_configuration(api_key, model)

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=_gemini_response_schema(response_model),
            ),
        )

        return _parse_response(response.text, response_model)

    except GeminiClientError:
        raise
    except Exception as error:
        raise GeminiClientError(
            f"Gemini request failed: {error}"
        ) from error


def generate_structured_from_document(
    *,
    api_key: str,
    model: str,
    prompt: str,
    response_model: type[T],
    file_path: Path,
    mime_type: str,
) -> T:
    """
    Generate structured output using a small local PDF or image as input.

    Use this for PDF and JPG/JPEG source files in Step 18.
    The original file is encoded and sent with the prompt.
    """

    _validate_configuration(api_key, model)

    if not file_path.exists():
        raise GeminiClientError(f"Document does not exist: {file_path}")

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model=model,
            contents=[
                types.Part.from_bytes(
                    data=file_path.read_bytes(),
                    mime_type=mime_type,
                ),
                prompt,
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=_gemini_response_schema(response_model),
            ),
        )

        return _parse_response(response.text, response_model)

    except GeminiClientError:
        raise
    except Exception as error:
        raise GeminiClientError(
            f"Gemini document extraction failed: {error}"
        ) from error
