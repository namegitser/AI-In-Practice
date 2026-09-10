#!/usr/bin/env python3
"""Lab 2 — the configurations under test.

Each variant is a callable `str -> dict`. `grid.py` runs them all through the
same harness, so the only thing that differs between rows of your table is the
thing you intended to differ.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip.llm import structured  # noqa: E402
from labs.lab1.extract import (  # noqa: E402
    SYSTEM_PROMPT, TicketRecord, apply_business_rules, extract_deterministic,
)

# ---------------------------------------------------------------------------
# A1 — six edge-case examples from the dev set.
# ---------------------------------------------------------------------------
FEW_SHOT_IDS: list[str] = [
    "T0054",  # teaches: billing/complaint boundary with no policy number and anger; useful because prose alone leaves that edge ambiguous.
    "T0238",  # teaches: ignore quoted historical replies when extracting policy_number and PII; quoted history is stale and should not be treated as live input.
    "T0183",  # teaches: an ombudsman threat plus Platinum plan should land as complaint / urgency 5, not a generic policy-change request.
    "T0200",  # teaches: Hinglish claims ticket with a policy number but no explicit product; language should be hi-en and product stays unknown.
    "T0021",  # teaches: a technical failure in Hinglish with urgency 5 and no policy number; the model must avoid inventing a policy or downgrading urgency.
    "T0097",  # teaches: forwarded email with a personal email address and repeated billing issue; the ticket is billing, not complaint-only, and contains_pii must stay true.
]


def load_examples(ids: list[str]) -> list[dict]:
    rows = [json.loads(l) for l in
            (ROOT / "data/eval/extraction_dev.jsonl").open(encoding="utf-8")]
    by_id = {r["id"]: r for r in rows}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise KeyError(f"unknown example ids: {missing}")
    return [by_id[i] for i in ids]


def _reasoning_for(case: dict) -> str:
    ticket = case["input"]
    expected = case["expected"]
    if expected["category"] == "complaint":
        return f"This is a complaint because the customer reports dissatisfaction or an ombudsman threat; urgency={expected['urgency']} follows the stated delay or threat, and no policy number is present in the live message."
    if expected["category"] == "policy_change":
        return f"The ticket asks to add or change a policy attribute, so category=policy_change; policy_number is null in the live ticket when only quoted history contains a reference, and the asked action is not a claim or billing dispute."
    if expected["category"] == "technical":
        return f"The issue is a portal/app failure rather than a billing or claims matter; the ticket is Hinglish/urgent and the model should not hallucinate a policy number."
    if expected["category"] == "billing":
        return f"This is a billing issue because the customer reports a double debit/refund request; the repeated payment problem makes urgency high and the presence of a personal email makes contains_pii true."
    if expected["category"] == "claims":
        return f"This is a claims question because it asks about claim settlement, reimbursement, or policy coverage; the policy number is present in the live body and product is unknown unless explicitly named."
    return f"This is a general information request because the ticket asks a coverage or account question without an active transaction or dispute; sentiment and urgency are taken directly from the ticket wording."


def few_shot_block(ids: list[str], *, include_reasoning: bool = False) -> str:
    """Render dev-set examples as explicit few-shot demonstrations."""
    examples: list[str] = []
    for case in load_examples(ids):
        out = dict(case["expected"])
        if include_reasoning:
            out = {"reasoning": _reasoning_for(case), **out}
        examples.append(
            f"Example {case['id']}\n"
            f"Ticket:\n{case['input']}\n\n"
            f"Output JSON:\n{json.dumps(out, ensure_ascii=False, indent=2)}"
        )
    return "\n\n".join(examples)


def _normalise_record(result: dict | BaseModel, ticket: str) -> dict:
    if hasattr(result, "model_dump"):
        out = result.model_dump()
    elif isinstance(result, dict):
        out = dict(result)
    else:
        out = {}
    det = extract_deterministic(ticket)
    out.update(det)
    return apply_business_rules(out, ticket)


def _run_structured(
    ticket: str,
    *,
    schema: type[BaseModel],
    tier: str,
    temperature: float | None = None,
    prompt: str | None = None,
) -> dict:
    try:
        if prompt is None:
            result = structured(ticket, schema=schema, system=SYSTEM_PROMPT,
                                tier=tier, temperature=temperature)
        else:
            result = structured(prompt, schema=schema, system=SYSTEM_PROMPT,
                                tier=tier, temperature=temperature)
    except Exception as exc:  # noqa: BLE001
        result = {
            "evidence": "",
            "category": "information",
            "urgency": 1,
            "sentiment": "neutral",
            "product": "unknown",
            "language": "en",
            "policy_number": None,
            "contains_pii": False,
            "needs_human_review": True,
            "review_reason": f"Structured parsing failure: {exc}",
        }
    return _normalise_record(result, ticket)


# ---------------------------------------------------------------------------
# The variants
# ---------------------------------------------------------------------------
def zero_shot(ticket: str, tier: str = "SMALL") -> dict:
    """Baseline: Lab 1 extractor, then deterministic post-processing."""
    return _run_structured(ticket, schema=TicketRecord, tier=tier)


def few_shot(ticket: str, tier: str = "SMALL") -> dict:
    """Zero-shot plus six hand-picked examples from the dev set."""
    prompt = (
        f"{few_shot_block(FEW_SHOT_IDS)}\n\n"
        f"Now extract the following ticket. Return only a JSON object.\n\n"
        f"Ticket:\n{ticket}"
    )
    return _run_structured(ticket, schema=TicketRecord, tier=tier, prompt=prompt)


class TicketRecordReasoned(BaseModel):
    """Reasoning-first schema: reasoning conditions the answer instead of rationalising it."""

    reasoning: str = Field(
        max_length=400,
        description="Brief reasoning grounded in the ticket text: why this category, urgency, sentiment, and language are correct."
    )
    evidence: str = Field(
        max_length=200,
        description="The span of the ticket that determined the category, quoted verbatim.",
    )
    category: str = Field(
        description="billing: premium payments, double debits, invoices, payment receipts, or billing refunds; claims: claim status, filing new claims, cashless hospitalisation, post-hospital reimbursement; policy_change: updating contact info (email/phone/address), adding beneficiaries, plan changes; technical: login errors, app crashes, portal downtime, password/OTP verification failures; complaint: dissatisfaction with Aurora service, delays, staff behaviour, or ombudsman threats; information: general policy coverage questions and inquiries with no active transaction."
    )
    urgency: int = Field(
        ge=1,
        le=5,
        description="Urgency scale 1-5: 1 = general informational query with no time limit; 2 = routine request with standard processing timeline; 3 = active issue with customer waiting on follow-up; 4 = delay >7 days, ombudsman threat, or deadline tomorrow; 5 = emergency in progress, active hospital admission, or critical medical crisis.",
    )
    sentiment: str = Field(
        description="Customer emotional state: 'angry' for threats/shouting, 'frustrated' for delays/annoyance, 'neutral' for factual queries, 'satisfied' for positive feedback."
    )
    product: str = Field(
        description="The Aurora insurance plan tier mentioned. Use 'unknown' when no specific tier is named."
    )
    language: str = Field(
        description="'en' if written entirely in English; 'hi-en' if code-mixed with Hindi words or phrasing (Hinglish)."
    )
    policy_number: str | None = Field(
        default=None,
        description="Policy number in exact format 'AUR-' followed by 7 digits. Must be null when no policy number appears. Never invent or reformat one.",
    )
    contains_pii: bool = Field(
        default=False,
        description="True if the ticket contains a phone number or email address. Customer names alone do not count.",
    )


def few_shot_reasoned(ticket: str, tier: str = "SMALL") -> dict:
    """Few-shot with a reasoning-first schema to see whether the field itself conditions the answer."""
    prompt = (
        f"{few_shot_block(FEW_SHOT_IDS, include_reasoning=True)}\n\n"
        f"Now extract the following ticket. Return only a JSON object.\n\n"
        f"Ticket:\n{ticket}"
    )
    return _run_structured(ticket, schema=TicketRecordReasoned, tier=tier, prompt=prompt)


def _records_differ(left: dict, right: dict) -> bool:
    ignore = {"needs_human_review", "review_reason", "_path"}
    return {k: v for k, v in left.items() if k not in ignore} != {
        k: v for k, v in right.items() if k not in ignore
    }


def cascade(ticket: str) -> dict:
    """Small model first, then escalate on validation failure or low agreement."""
    small = _run_structured(ticket, schema=TicketRecord, tier="SMALL", temperature=0.0)
    small["_path"] = "small"

    if small.get("needs_human_review") or not small.get("evidence", "").strip():
        large = _run_structured(ticket, schema=TicketRecord, tier="MAIN", temperature=0.0)
        large["_path"] = "large"
        return large

    small_2 = _run_structured(ticket, schema=TicketRecord, tier="SMALL", temperature=0.7)
    if _records_differ(small, small_2):
        large = _run_structured(ticket, schema=TicketRecord, tier="MAIN", temperature=0.0)
        large["_path"] = "large"
        return large

    return small


VARIANTS = {
    "zero_shot": lambda t: zero_shot(t, "SMALL"),
    "zero_shot_main": lambda t: zero_shot(t, "MAIN"),
    "few_shot": lambda t: few_shot(t, "SMALL"),
    "few_shot_main": lambda t: few_shot(t, "MAIN"),
    "few_shot_reasoned": lambda t: few_shot_reasoned(t, "SMALL"),
    "few_shot_reasoned_main": lambda t: few_shot_reasoned(t, "MAIN"),
    "cascade": cascade,
}
