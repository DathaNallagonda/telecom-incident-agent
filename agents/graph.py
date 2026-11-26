"""Supervisor graph: routes state between the agents.

    triage -> plan -> execute -> communicate -> END
       \\-> END (with an error) if triage failed

execute and communicate are stubs until Day 3 (Operator and Comms agents).
A MemorySaver checkpointer lets the graph pause (for approvals) and resume.
"""

from __future__ import annotations

import uuid
from typing import Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agents.planner_agent import plan_with_verification
from agents.schemas import TriageResult
from agents.triage import run_triage
from core.simulator import NetworkState, from_dict, to_dict


class IncidentState(TypedDict, total=False):
    incident_text: str
    network: dict  # serialised NetworkState (see core.simulator.to_dict)
    goal_mode: str
    triage: dict | None
    triage_calls: list[dict]
    plan: dict | None
    execution_log: list[dict]
    pending_approval: dict | None
    report: str | None
    errors: list[str]


def triage_node(state: IncidentState) -> dict[str, Any]:
    outcome = run_triage(state["incident_text"], from_dict(state["network"]))
    if outcome.result is None:
        return {"triage": None, "triage_calls": outcome.transcript, "errors": [outcome.error]}
    return {"triage": outcome.result.model_dump(), "triage_calls": outcome.transcript}


def plan_node(state: IncidentState) -> dict[str, Any]:
    network: NetworkState = from_dict(state["network"])
    triage = TriageResult.model_validate(state["triage"], context={"network": network})
    result = plan_with_verification(triage, network, state.get("goal_mode", "default"))
    errors = [] if result.valid else [f"planning failed: {result.message}"]
    return {"plan": result.to_dict(), "errors": state.get("errors", []) + errors}


def execute_node(state: IncidentState) -> dict[str, Any]:
    # Day 3: the Operator Agent executes step by step with approvals and replanning.
    return {"execution_log": [], "pending_approval": None}


def communicate_node(state: IncidentState) -> dict[str, Any]:
    # Day 3: the Comms Agent writes customer SMS and the incident report.
    return {"report": None}


def after_triage(state: IncidentState) -> str:
    return "plan" if state.get("triage") else END


def after_plan(state: IncidentState) -> str:
    plan = state.get("plan") or {}
    return "execute" if plan.get("valid") else END


def build_graph(checkpointer=None):
    g = StateGraph(IncidentState)
    g.add_node("triage", triage_node)
    g.add_node("plan", plan_node)
    g.add_node("execute", execute_node)
    g.add_node("communicate", communicate_node)
    g.add_edge(START, "triage")
    g.add_conditional_edges("triage", after_triage, ["plan", END])
    g.add_conditional_edges("plan", after_plan, ["execute", END])
    g.add_edge("execute", "communicate")
    g.add_edge("communicate", END)
    return g.compile(checkpointer=checkpointer or MemorySaver())


def run_incident(
    incident_text: str, network: NetworkState, goal_mode: str = "default", graph=None
) -> tuple[IncidentState, str]:
    """Run the whole graph once; returns the final state and the thread id."""
    graph = graph or build_graph()
    thread_id = str(uuid.uuid4())
    final = graph.invoke(
        {
            "incident_text": incident_text,
            "network": to_dict(network),
            "goal_mode": goal_mode,
            "errors": [],
        },
        {"configurable": {"thread_id": thread_id}},
    )
    return final, thread_id
