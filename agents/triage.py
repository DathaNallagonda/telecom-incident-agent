"""Triage Agent: reads messy incident text, investigates with tools, returns a TriageResult.

Two stages:
1. Investigate: a LangGraph ReAct agent calls the network tools (max 6 steps).
2. Report: a structured-output call turns the findings into a TriageResult,
   which is validated against the network. On a validation error the error
   is sent back once; after 2 failures a clear error is returned.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from pydantic import ValidationError

from agents.schemas import TriageResult
from agents.tools import make_tools
from core import llm as llm_mod
from core.simulator import NetworkState

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from langgraph.prebuilt import create_react_agent

MAX_STEPS = 6
MAX_REPORT_ATTEMPTS = 2

SYSTEM_PROMPT = """You are the Triage Agent in a telecom network operations centre.
An incident report has arrived. It may be vague or written by a customer.

The network has a core router C0, towers T1-T6, and sectors S1-S6 (tower Tn serves sector Sn).

Investigate with your tools, then summarise your findings:
- Start with get_alarms.
- For each alarmed node, run_diagnostics to learn the fault type, and
  get_sectors_for_node to learn which sectors lost service.
- Never guess ids: only use ids the tools return.
- Be efficient: you have at most {max_steps} tool-calling steps.

End with a short plain-text summary: faulty nodes and their fault types
(power, hardware or link), affected sectors, links that are down, and a priority
(P1 = a sector with no backup tower or several sectors down, P2 = one sector down
with a backup available, P3 = degraded but serving)."""

REPORT_PROMPT = """Convert the investigation below into the TriageResult JSON.
Use only ids that appear in the tool results. fault_types needs exactly one entry
per id in faulty_nodes. A cut backhaul link of tower Tn is the link Tn-C0.
confidence is between 0 and 1. reasoning is one or two short sentences."""


@dataclass
class TriageOutcome:
    result: TriageResult | None
    error: str | None = None
    transcript: list[dict] = field(default_factory=list)  # tool calls, for the UI
    model: str = ""


def _transcript(messages: list[BaseMessage]) -> tuple[str, list[dict]]:
    """Plain-text record of the investigation plus a list of tool calls."""
    lines, calls = [], []
    for m in messages:
        if isinstance(m, AIMessage):
            for tc in m.tool_calls:
                lines.append(f"CALL {tc['name']}({json.dumps(tc['args'])})")
                calls.append({"tool": tc["name"], "args": tc["args"]})
            if m.text:
                lines.append(f"AGENT: {m.text}")
        elif isinstance(m, ToolMessage):
            lines.append(f"RESULT {m.name}: {m.content}")
            if calls:
                calls[-1]["result"] = m.content
    return "\n".join(lines), calls


def investigate(model: BaseChatModel, incident_text: str, state: NetworkState) -> list[BaseMessage]:
    agent = create_react_agent(
        model, make_tools(state), prompt=SYSTEM_PROMPT.format(max_steps=MAX_STEPS)
    )
    messages: list[BaseMessage] = []
    try:
        for values in agent.stream(
            {"messages": [HumanMessage(incident_text)]},
            {"recursion_limit": 2 * MAX_STEPS + 1},
            stream_mode="values",
        ):
            messages = values["messages"]
    except GraphRecursionError:
        pass  # step limit reached: report on what was found so far
    return messages


def report(
    incident_text: str,
    transcript: str,
    state: NetworkState,
    structured=None,
) -> tuple[TriageResult | None, str | None]:
    structured = structured or llm_mod.structured(TriageResult)
    msgs: list[BaseMessage] = [
        SystemMessage(REPORT_PROMPT),
        HumanMessage(f"Incident report:\n{incident_text}\n\nInvestigation:\n{transcript}"),
    ]
    error = None
    for _ in range(MAX_REPORT_ATTEMPTS):
        data = None
        try:
            raw = structured.invoke(msgs)
            data = raw.model_dump() if hasattr(raw, "model_dump") else raw
            return TriageResult.model_validate(data, context={"network": state}), None
        except ValidationError as e:
            error = "; ".join(err["msg"] for err in e.errors())
            msgs += [
                AIMessage(json.dumps(data) if data is not None else "(invalid output)"),
                HumanMessage(f"That output failed validation: {error}. Return corrected JSON."),
            ]
        except Exception as e:  # LLM/network failure after fallbacks
            error = f"{type(e).__name__}: {str(e)[:200]}"
    return None, f"triage failed after {MAX_REPORT_ATTEMPTS} attempts: {error}"


def run_triage(incident_text: str, state: NetworkState) -> TriageOutcome:
    """Investigate with the first model that works, then produce a validated TriageResult."""
    try:
        models = llm_mod.chat_models()
    except llm_mod.LLMUnavailableError as e:
        return TriageOutcome(None, str(e))

    messages, used, last_error = [], "", None
    for model in models:
        try:
            messages = investigate(model, incident_text, state)
            used = getattr(model, "model", None) or getattr(model, "model_name", "")
            break
        except Exception as e:  # provider down: try the next model
            last_error = f"{type(e).__name__}: {str(e)[:200]}"
    if not messages:
        return TriageOutcome(None, f"investigation failed on every model: {last_error}")

    text, calls = _transcript(messages)
    result, error = report(incident_text, text, state)
    return TriageOutcome(result, error, calls, str(used))
