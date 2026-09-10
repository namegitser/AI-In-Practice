#!/usr/bin/env python3
"""Lab 1, Parts B and C — the extractor you actually ship.

Complete the TODOs. `run_eval.py` imports `extract_b` and `extract_c` from
here, so keep those two function names.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aip.guards import _PII_PATTERNS  # noqa: E402
from aip.llm import StructuredOutputError, structured  # noqa: E402

CATEGORIES = Literal[
    "billing",
    "claims",
    "policy_change",
    "technical",
    "complaint",
    "information",
]


# ===========================================================================
# PART B — the schema
# ===========================================================================
class TicketRecord(BaseModel):
    """The contract. Everything the model is allowed to say, and nothing else."""

    # Placed evidence first so the model generates supporting text before committing to classifications (Chain-of-Thought effect).
    evidence: str = Field(
        max_length=200,
        description="The span of the ticket that determined the category, quoted verbatim.",
    )

    category: CATEGORIES = Field(
        description="billing: premium payments, double debits, invoices, payment receipts, or billing refunds; "
                    "claims: claim status, filing new claims, cashless hospitalisation, post-hospital reimbursement; "
                    "policy_change: updating contact info (email/phone/address), adding beneficiaries, plan changes; "
                    "technical: login errors, app crashes, portal downtime, password/OTP verification failures; "
                    "complaint: dissatisfaction with Aurora service, delays, staff behaviour, or ombudsman threats; "
                    "information: general policy coverage questions and inquiries with no active transaction."
    )

    urgency: int = Field(
        ge=1,
        le=5,
        description="Urgency scale 1-5: "
                    "1 = general informational query with no time limit; "
                    "2 = routine request with standard processing timeline; "
                    "3 = active issue with customer waiting on follow-up; "
                    "4 = delay >7 days, ombudsman threat, or deadline tomorrow; "
                    "5 = emergency in progress, active hospital admission, or critical medical crisis."
    )

    sentiment: Literal["angry", "frustrated", "neutral", "satisfied"] = Field(
        description="Customer emotional state: 'angry' for threats/shouting, 'frustrated' for delays/annoyance, "
                    "'neutral' for factual queries, 'satisfied' for positive feedback."
    )

    product: Literal["bronze", "silver", "gold", "platinum", "unknown"] = Field(
        description="The Aurora insurance plan tier mentioned. Use 'unknown' when no specific tier is named."
    )

    language: Literal["en", "hi-en"] = Field(
        description="'en' if written entirely in English; 'hi-en' if code-mixed with Hindi words or phrasing (Hinglish)."
    )

    policy_number: str | None = Field(
        default=None,
        description="Policy number in exact format 'AUR-' followed by 7 digits (e.g. AUR-1234567). "
                    "Must be null when no policy number appears. Never invent or reformat one."
    )

    contains_pii: bool = Field(
        default=False,
        description="True if the ticket contains a phone number or email address. Customer names alone do not count."
    )

    # Set by our code, never by the model.
    needs_human_review: bool = False
    review_reason: str = ""

    @field_validator("policy_number", mode="before")
    @classmethod
    def _policy_format(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = str(v).strip()
        if v.lower() in {"", "null", "none", "n/a"}:
            return None
        match = re.search(r"AUR-\d{7}", v)
        return match.group(0) if match else None


SYSTEM_PROMPT = """\
You are an operations ticket extraction engine for Aurora Health Insurance.
Extract ticket metadata strictly following the supplied JSON schema.

Rules:
1. Evidence: Always extract the exact verbatim quote justifying the category first.
2. Restraint: Never hallucinate or infer policy numbers. Return null if not matching AUR-<7 digits>.
3. Complaints: Service failures, unresolved delays, and company disputes must be classified as 'complaint'.
4. Language: Code-mixed Hindi-English (e.g., 'jaldi karo') must be tagged as 'hi-en'.
"""


def extract_b(ticket: str) -> TicketRecord:
    """Part B: the model decides everything."""
    try:
        # Pass ticket as positional, schema and system as keyword arguments
        return structured(ticket, schema=TicketRecord, system=SYSTEM_PROMPT)
    except StructuredOutputError as exc:
        return TicketRecord(
            evidence="",
            category="information",
            urgency=1,
            sentiment="neutral",
            product="unknown",
            language="en",
            policy_number=None,
            contains_pii=False,
            needs_human_review=True,
            review_reason=f"Structured parsing failure: {exc}",
        )


# ===========================================================================
# PART C — move the deterministic work out of the model
# ===========================================================================
POLICY_RE = re.compile(r"\bAUR-\d{7}\b")

# The quoted-reply marker. Everything after this is history, not the current
# message. Part C3 asks you to decide what that means for policy extraction.
QUOTE_MARKER = re.compile(r"^\s*>", re.MULTILINE)


def extract_deterministic(ticket: str) -> dict:
    """Return {'policy_number', 'contains_pii'} without a model call."""
    # Rule for C3: We extract policy number ONLY from the live message body
    # (before any quoted reply marked with '>'). Policy numbers appearing in
    # historical/quoted threads are stale and should not be extracted.
    live_body = QUOTE_MARKER.split(ticket, maxsplit=1)[0]
    match = POLICY_RE.search(live_body)
    policy_number = match.group(0) if match else None

    # Check for phone numbers and personal email addresses (ignoring Aurora internal support emails)
    has_phone = bool(_PII_PATTERNS["PHONE_IN"].search(ticket))
    emails = _PII_PATTERNS["EMAIL"].findall(ticket)
    has_personal_email = any(
        not em.lower().endswith("@aurorahealth.example") for em in emails
    )
    has_pii = has_phone or has_personal_email

    return {
        "policy_number": policy_number,
        "contains_pii": has_pii,
    }


def apply_business_rules(rec_fields: dict, ticket: str) -> dict:
    """Compute `escalate` in code: escalate = urgency >= 4 or 'ombudsman' appears in ticket."""
    urgency = rec_fields.get("urgency", 1)
    has_ombudsman = "ombudsman" in ticket.lower()
    rec_fields["escalate"] = bool(urgency >= 4 or has_ombudsman)
    return rec_fields


class TicketRecordC(BaseModel):
    """The reduced schema the model sees in Part C."""

    # Placed evidence first so the model generates supporting text before committing to classifications.
    evidence: str = Field(
        max_length=200,
        description="The span of the ticket that determined the category, quoted verbatim.",
    )

    category: CATEGORIES = Field(
        description="billing: premium payments, double debits, invoices, 80D tax cert, or refunds; "
                    "claims: claim status, filing new claims, cashless hospitalisation, reimbursement; "
                    "policy_change: contact info update (email/phone/address), adding beneficiaries, plan port/changes; "
                    "technical: login errors, app crashes, portal downtime, password/OTP verification failures; "
                    "complaint: dissatisfaction with Aurora conduct/service, delays, staff behaviour, or ombudsman threats; "
                    "information: general policy coverage questions with no active transaction."
    )

    urgency: int = Field(
        ge=1,
        le=5,
        description="Urgency scale 1-5: "
                    "1 = general informational query answerable without opening customer record; "
                    "2 = routine request requiring Aurora to lookup account or act on standard in-flight transaction; "
                    "3 = active issue stuck or delayed where customer is waiting on follow-up; "
                    "4 = repeated failure ('third time'), money/access at risk now, or threat of escalation; "
                    "5 = medical emergency in progress, ICU admission, claim denial reversal, or active filing with Ombudsman."
    )

    sentiment: Literal["angry", "frustrated", "neutral", "satisfied"] = Field(
        description="Customer emotional state: 'angry' for threats/shouting, 'frustrated' for delays/prior failures, "
                    "'neutral' for factual/first-time queries, 'satisfied' for positive feedback."
    )

    product: Literal["bronze", "silver", "gold", "platinum", "unknown"] = Field(
        description="The Aurora insurance plan tier explicitly named in the ticket. Use 'unknown' if no specific tier is named."
    )

    language: Literal["en", "hi-en"] = Field(
        description="'en' if written entirely in English; 'hi-en' if code-mixed with Hindi words or phrasing (Hinglish)."
    )


def extract_c(ticket: str) -> dict:
    """Part C: model for judgement, code for everything else."""
    det_fields = extract_deterministic(ticket)
    try:
        model_rec = structured(ticket, schema=TicketRecordC, system=SYSTEM_PROMPT)
        result = model_rec.model_dump()
    except StructuredOutputError as exc:
        result = {
            "evidence": "",
            "category": "information",
            "urgency": 1,
            "sentiment": "neutral",
            "product": "unknown",
            "language": "en",
            "needs_human_review": True,
            "review_reason": f"Structured parsing failure: {exc}",
        }
    except Exception as exc:
        result = {
            "evidence": "",
            "category": "information",
            "urgency": 1,
            "sentiment": "neutral",
            "product": "unknown",
            "language": "en",
            "needs_human_review": True,
            "review_reason": f"Unexpected error: {exc}",
        }
    result.update(det_fields)
    return apply_business_rules(result, ticket)


if __name__ == "__main__":
    import json

    root = Path(__file__).resolve().parents[2]
    sample = json.loads(
        (root / "data/eval/extraction_dev.jsonl").open(encoding="utf-8").readline()
    )
    print("--- ticket ---")
    print(sample["input"][:600])
    print("\n--- gold ---")
    print(sample["expected"])
    print("\n--- yours ---")
    print(extract_c(sample["input"]))