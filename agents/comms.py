"""Comms Agent: customer SMS per affected sector and a Markdown incident report.

The LLM writes the words; code checks them (SMS <= 160 characters, no internal
ids such as T3 or S3) and falls back to a template if the LLM is unavailable
or its output still breaks the rules after one retry.
"""

from __future__ import annotations

import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from core import llm as llm_mod

SMS_MAX = 160
INTERNAL_ID = re.compile(r"\b(?:[TSC]\d+)\b")

# Customer-facing area names; customers never see sector ids.
AREA_NAMES = {
    "S1": "Central Market",
    "S2": "Railway Station",
    "S3": "City Mall",
    "S4": "Tech Park",
    "S5": "University",
    "S6": "Airport Road",
}


class SectorSMS(BaseModel):
    sector_id: str
    text: str = Field(description="Customer SMS, plain language, max 160 characters")


class CommsDraft(BaseModel):
    sms: list[SectorSMS]
    report_markdown: str = Field(description="Incident report in Markdown")


SYSTEM_PROMPT = """You are the Comms Agent of a mobile network operator.
Write:
1. One SMS per affected sector for customers in that area: plain, calm language,
   at most 160 characters, name the area (never internal ids like T3 or S3), say
   whether service is restored, and apologise briefly.
2. An internal incident report in Markdown with sections: Summary, Root cause,
   Actions taken, Replans, Time to resolve, Status. Internal ids are fine here."""


def sms_problems(drafts: list[SectorSMS], sectors: list[str]) -> list[str]:
    problems = []
    got = {d.sector_id for d in drafts}
    for s in sectors:
        if s not in got:
            problems.append(f"missing SMS for {s}")
    for d in drafts:
        if len(d.text) > SMS_MAX:
            problems.append(f"SMS for {d.sector_id} is {len(d.text)} characters (max {SMS_MAX})")
        if INTERNAL_ID.search(d.text):
            problems.append(f"SMS for {d.sector_id} contains an internal id")
    return problems


def template_sms(sector: str, restored: bool) -> str:
    area = AREA_NAMES.get(sector, "your area")
    if restored:
        return (
            f"Service in the {area} area has been restored. Sorry for the disruption. - Network Ops"
        )
    return (
        f"We are fixing a network fault affecting the {area} area. "
        "Sorry for the disruption. - Network Ops"
    )


def template_report(facts: dict) -> str:
    lines = [
        "## Summary",
        f"{facts['incident_text']}",
        "",
        "## Root cause",
        ", ".join(f"{n}: {t}" for n, t in facts["faults"].items()) or "unknown",
        "",
        "## Actions taken",
        *[f"{i}. {s}" for i, s in enumerate(facts["actions"], 1)],
        "",
        "## Replans",
        *([f"- {r}" for r in facts["replans"]] or ["None"]),
        "",
        "## Time to resolve",
        f"{facts['duration_s']:.0f} s (wall-clock time of the agent run, simulated network)",
        "",
        "## Status",
        facts["status"],
    ]
    return "\n".join(lines)


def write_comms(facts: dict, writer=None) -> dict:
    """facts: incident_text, faults {node: type}, affected_sectors, restored,
    actions [str], replans [str], duration_s, status.

    Returns {"sms": [{sector_id, area, text}], "report": markdown, "source": "llm"|"template"}.
    """
    sectors = facts["affected_sectors"]
    msgs = [
        SystemMessage(SYSTEM_PROMPT),
        HumanMessage(
            "Facts (the only information you may use):\n"
            + "\n".join(f"- {k}: {v}" for k, v in facts.items())
            + "\nArea names: "
            + ", ".join(f"{s} = {AREA_NAMES.get(s, s)}" for s in sectors)
        ),
    ]
    draft, error = None, None
    try:
        writer = writer or llm_mod.structured(CommsDraft)
        for _ in range(2):
            draft = writer.invoke(msgs)
            problems = sms_problems(draft.sms, sectors)
            if not problems:
                break
            msgs += [
                AIMessage(draft.model_dump_json()),
                HumanMessage(
                    "Fix these problems and return everything again: " + "; ".join(problems)
                ),
            ]
    except Exception as e:  # LLM down or rate limited: fall back to templates
        draft, error = None, f"{type(e).__name__}: {str(e)[:200]}"

    restored = facts.get("restored", False)
    if draft is None:
        sms = {s: template_sms(s, restored) for s in sectors}
        report, source = template_report(facts), "template"
    else:
        good = {d.sector_id: d.text for d in draft.sms if not sms_problems([d], [])}
        sms = {s: good.get(s) or template_sms(s, restored) for s in sectors}
        report, source = draft.report_markdown, "llm"
    return {
        "sms": [{"sector_id": s, "area": AREA_NAMES.get(s, s), "text": t} for s, t in sms.items()],
        "report": report,
        "source": source,
        "error": error,
    }
