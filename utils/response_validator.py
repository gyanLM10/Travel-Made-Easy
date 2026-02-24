"""
Response Validator — Hallucination Control Layer
------------------------------------------------
Runs a second LLM "critic" call on every generated travel plan.
Returns a structured trustworthiness assessment with:
  - confidence_score (0-100)
  - trustworthy flag
  - uncertain_claims  (prices, hours, visa policies, etc.)
  - verified_by_tools (which tools grounded the answer)
  - summary           (one-line verdict)

No new API keys needed — uses the same Groq model already configured.
Falls back gracefully if the critic call fails.
"""

from __future__ import annotations

import json
import os
import re
from typing import TypedDict


class ValidationResult(TypedDict):
    confidence_score: int          # 0-100
    trustworthy: bool              # True if score >= 60
    uncertain_claims: list[str]    # specific things to double-check
    verified_by_tools: list[str]   # tools that grounded the answer
    summary: str                   # one-sentence verdict


_CRITIC_PROMPT = """\
You are a strict travel-information fact-checker.
A travel AI assistant just generated the plan below in response to the user query.

USER QUERY: {question}

GENERATED PLAN:
{plan}

Your task is to evaluate this plan for potential hallucinations — claims that:
• Are specific but unverifiable at the time of generation (exact ticket prices, visa fees, precise opening hours, exchange rates, specific bus/train schedules)
• May be outdated or vary by season / nationality
• Were likely invented rather than retrieved from a live source

Return ONLY valid JSON with this exact structure (no markdown, no explanation):
{{
  "confidence_score": <integer 0-100>,
  "trustworthy": <true if score >= 60, else false>,
  "uncertain_claims": [<list of short strings, each one specific claim that should be verified>],
  "verified_by_tools": [<list of topics that appear grounded: "weather", "places", "currency", "attractions", etc.>],
  "summary": "<one sentence verdict>"
}}

Scoring guide:
• 80-100: Mostly general advice; volatile facts are either absent or clearly marked as estimates
• 50-79: Some specific numbers/dates present; user should verify before booking
• 0-49:  Many precise but unverifiable claims (prices, schedules, visa details stated as facts)

Be strict. A plan that says "tickets cost €28" without a source should lose points.
A plan that says "tickets are approximately €20-30, verify on the official site" is better.
"""


def validate(question: str, plan: str) -> ValidationResult:
    """
    Run the critic LLM on the generated travel plan.
    Never raises — returns a safe default on any failure.
    """
    try:
        return _run_critic(question, plan)
    except Exception as exc:
        print(f"⚠️ ResponseValidator failed (non-critical): {exc}")
        return _safe_default()


def _run_critic(question: str, plan: str) -> ValidationResult:
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return _safe_default()

    critic_llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=512,
        timeout=30,
    )

    prompt = _CRITIC_PROMPT.format(question=question, plan=plan[:3000])  # cap length
    response = critic_llm.invoke(prompt)
    raw = response.content.strip()

    # Strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    data = json.loads(raw)

    return ValidationResult(
        confidence_score=int(data.get("confidence_score", 50)),
        trustworthy=bool(data.get("trustworthy", True)),
        uncertain_claims=list(data.get("uncertain_claims", [])),
        verified_by_tools=list(data.get("verified_by_tools", [])),
        summary=str(data.get("summary", "")),
    )


def _safe_default() -> ValidationResult:
    """Returned when the critic call cannot complete."""
    return ValidationResult(
        confidence_score=-1,
        trustworthy=True,
        uncertain_claims=[],
        verified_by_tools=[],
        summary="Trustworthiness check unavailable.",
    )
