"""
Shared, low-level LLM call helpers.

All LLM calls in the Legal GraphRAG system go through this module:
- clause extraction
- text-to-Cypher
- router
- auditor
- synthesizer

Uses Hugging Face InferenceClient.
"""

from __future__ import annotations

import json
import os
import re

from huggingface_hub import InferenceClient

from .tracing import traceable

import os
import json
import re

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from .tracing import trace_huggingface_call

load_dotenv()
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HF_TOKEN = os.getenv("HF_TOKEN")

# Choose the Hugging Face model you want to use.
# Example:
MODEL_NAME = "openai/gpt-oss-120b:groq"

if not HF_TOKEN:
    raise RuntimeError(
        "HF_TOKEN environment variable is not set. "
        "Create a Hugging Face token with inference permissions."
    )


# ---------------------------------------------------------------------------
# Hugging Face client
# ---------------------------------------------------------------------------

_client = InferenceClient(
    api_key=HF_TOKEN
)


# ---------------------------------------------------------------------------
# JSON LLM call
# ---------------------------------------------------------------------------

@traceable(name="llm.call_json", run_type="llm")
def call_json(
    system: str,
    user: str,
    max_tokens: int = 2000,
) -> dict | list:
    """
    Call the Hugging Face model expecting JSON output.

    Returns:
        Parsed dict or list.
    """

    response = _client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": system,
            },
            {
                "role": "user",
                "content": user,
            },
        ],
        max_tokens=max_tokens,
        temperature=0.0,
    )

    raw = response.choices[0].message.content or ""

    cleaned = raw.strip()

    # Remove markdown code fences if the model still returns them.
    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned,
    )

    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)

    except json.JSONDecodeError:

        # Last-resort recovery:
        # find the first JSON object or array.
        match = re.search(
            r"(\{.*\}|\[.*\])",
            cleaned,
            flags=re.DOTALL,
        )

        if match:
            return json.loads(match.group(1))

        raise ValueError(
            "Model returned invalid JSON.\n\n"
            f"Raw response:\n{raw}"
        )


# ---------------------------------------------------------------------------
# Plain-text LLM call
# ---------------------------------------------------------------------------

@traceable(name="llm.call_text", run_type="llm")
def call_text(
    system: str,
    user: str,
    max_tokens: int = 1500,
) -> str:
    """
    Call the Hugging Face model expecting plain text.
    """

    response = _client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": system,
            },
            {
                "role": "user",
                "content": user,
            },
        ],
        max_tokens=max_tokens,
        temperature=0.2,
    )

    return (
        response.choices[0].message.content or ""
    ).strip()


# """
# Shared, low-level LLM call helpers.

# Centralized here (rather than duplicated per module) so every part of the
# system — clause extraction, text-to-Cypher, the router, the auditor, the
# synthesizer — goes through the same client construction and the same
# defensive JSON parsing, instead of five slightly-different copies drifting
# apart over time.
# """

# from __future__ import annotations

# import json
# import os
# import re

# import anthropic

# from .tracing import traceable, wrap_anthropic_client

# # MODEL_NAME = "claude-sonnet-5"  # adjust to whatever model your API key has access to

# # wrap_anthropic_client is a no-op passthrough unless LangSmith tracing is
# # enabled (see tracing.configure_langsmith) — when enabled, every call below
# # is logged as its own LLM run with prompt/completion/token counts, in
# # addition to the @traceable span each call is made from.
# _client = wrap_anthropic_client(anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY")))


# @traceable(name="llm.call_json", run_type="llm")
# def call_json(system: str, user: str, max_tokens: int = 2000) -> dict | list:
#     """Call the model expecting JSON-only output, and parse it defensively."""
#     response = _client.messages.create(
#         model=MODEL_NAME,
#         max_tokens=max_tokens,
#         system=system,
#         messages=[{"role": "user", "content": user}],
#     )
#     raw = "".join(block.text for block in response.content if block.type == "text")
#     cleaned = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
#     try:
#         return json.loads(cleaned)
#     except json.JSONDecodeError:
#         # Last-resort recovery: grab the first {...} or [...] block in the response.
#         match = re.search(r"(\{.*\}|\[.*\])", cleaned, flags=re.DOTALL)
#         if match:
#             return json.loads(match.group(1))
#         raise


# @traceable(name="llm.call_text", run_type="llm")
# def call_text(system: str, user: str, max_tokens: int = 1500) -> str:
#     """Call the model expecting a plain-text (non-JSON) response."""
#     response = _client.messages.create(
#         model=MODEL_NAME,
#         max_tokens=max_tokens,
#         system=system,
#         messages=[{"role": "user", "content": user}],
#     )
#     return "".join(block.text for block in response.content if block.type == "text").strip()
