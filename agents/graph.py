"""Supervisor graph: routes state between the agents.

    START -> triage -> plan -> execute --(next step)--> execute
               |         ^       |  \\--(step failed / blocked)--> plan (replan)
               v         |       v
              END        |   communicate -> END
                         \\-- (plan failed) -> END

execute runs ONE step per visit. A risky step pauses the graph with
interrupt(); resume it with Command(resume={"approved": bool, "by": name}).
A MemorySaver checkpointer keeps the paused state per thread id.
"""

from __future__ import annotations

import random
import time
import uuid
from typing import Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from agents.comms import write_comms
from agents.operator import MAX_REPLANS, execute_step, fmt, is_risky, replan
from agents.planner_agent import plan_with_verification
from agents.schemas import TriageResult
from agents.triage import run_triage
from core.simulator import NetworkState, from_dict, goals_met, to_dict
from core.spec import goal_facts

RECURSION_LIMIT = 150


class IncidentState(TypedDict, total=False):
    incident_text: str
    network: dict  # live network, serialised (core.simulator.to_dict)
    goal_mode: str
    goal_facts: list[list[str]]
    inject_failure: bool  # make the first restart fail (replanning demo)
    auto_approve: bool  # approve risky steps without pausing (evaluation)
    started_at: float
    triage: dict | None
    triage_calls: list[dict]
    plan: dict | None  # current plan; source "replan" after a replan
    first_plan: dict | None  # the verify-and-fix result, kept for the UI
    plan_version: int
    step_index: int
    replan_reason: str | None
    replans: list[dict]
    restart_failed: bool
    execution_log: list[dict]
    status: str  # running | completed | rejected | failed
    comms: dict | None
    report: str | None
    errors: list[str]


# --- nodes -------------------------------------------------------------------
def triage_node(state: IncidentState) -> dict[str, Any]:
    outcome = run_triage(state["incident_text"], from_dict(state["network"]))
    if outcome.result is None:
        return {
            "triage": None,
            "triage_calls": outcome.transcript,
            "status": "failed",
            "errors": state.get("errors", []) + [outcome.error],
        }
    return {"triage": outcome.result.model_dump(), "triage_calls": outcome.transcript}


def plan_node(state: IncidentState) -> dict[str, Any]:
    network: NetworkState = from_dict(state["network"])
    if state.get("replan_reason"):  # Operator asked for a new plan from the live state
        replans = state.get("replans", [])
        if len(replans) >= MAX_REPLANS:
            return {
                "status": "failed",
                "replan_reason": None,
                "errors": state.get("errors", []) + [f"gave up after {MAX_REPLANS} replans"],
            }
        new = replan(network, state.get("goal_mode", "default"))
        entry = {"reason": state["replan_reason"], "new_plan": new.plan}
        if new.status != "solved":
            return {
                "status": "failed",
                "replan_reason": None,
                "replans": replans + [entry],
                "errors": state.get("errors", []) + [f"replanning failed: {new.message}"],
            }
        return {
            "plan": {
                "steps": new.plan,
                "source": "replan",
                "valid": True,
                "rounds_used": 0,
                "validation_log": [],
                "message": new.message,
            },
            "plan_version": state.get("plan_version", 0) + 1,
            "step_index": 0,
            "replan_reason": None,
            "replans": replans + [entry],
        }

    triage = TriageResult.model_validate(state["triage"], context={"network": network})
    result = plan_with_verification(triage, network, state.get("goal_mode", "default")).to_dict()
    update: dict[str, Any] = {
        "plan": result,
        "first_plan": result,
        "plan_version": 0,
        "step_index": 0,
    }
    if not result["valid"]:
        update["status"] = "failed"
        update["errors"] = state.get("errors", []) + [f"planning failed: {result['message']}"]
    return update


def execute_node(state: IncidentState) -> dict[str, Any]:
    steps = state["plan"]["steps"]
    i = state.get("step_index", 0)
    step = steps[i]
    version = state.get("plan_version", 0)
    log = list(state.get("execution_log", []))

    approved_by = None
    if is_risky(step):
        if state.get("auto_approve"):
            approved_by = "auto"
        else:
            decision = interrupt(
                {
                    "step": step,
                    "index": i + 1,
                    "plan_version": version,
                    "question": f"Approve risky step {fmt(step)}?",
                }
            )
            if not decision.get("approved"):
                log.append(
                    {
                        "index": i + 1,
                        "action": step["action"],
                        "args": step["args"],
                        "result": "rejected",
                        "message": "rejected by operator",
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "approved_by": None,
                        "plan_version": version,
                    }
                )
                return {"execution_log": log, "status": "rejected"}
            approved_by = decision.get("by", "operator")

    network = from_dict(state["network"])
    fail_now = bool(
        state.get("inject_failure")
        and step["action"] == "restart_node"
        and not state.get("restart_failed")
    )
    rec = execute_step(
        network,
        step,
        i + 1,
        approved_by,
        plan_version=version,
        restart_failure_prob=1.0 if fail_now else 0.0,
        rng=random.Random(0),
    )
    update: dict[str, Any] = {
        "network": to_dict(network),
        "execution_log": log + [rec.to_dict()],
    }
    if rec.result == "ok":
        update["step_index"] = i + 1
    else:
        update["replan_reason"] = f"{fmt(step)}: {rec.message}"
        if fail_now:
            update["restart_failed"] = True
    return update


def communicate_node(state: IncidentState) -> dict[str, Any]:
    network = from_dict(state["network"])
    goals = [tuple(g) for g in state.get("goal_facts", [])]
    unmet = goals_met(network, goals)
    status = state.get("status") or "running"
    if status not in ("rejected", "failed"):
        status = "completed" if not unmet else "failed"
    triage = state.get("triage") or {}
    faults = {f["node_id"]: f["fault_type"] for f in triage.get("fault_types", [])}
    facts = {
        "incident_text": state["incident_text"],
        "faults": faults,
        "affected_sectors": triage.get("affected_sectors", []),
        "restored": status == "completed",
        "actions": [
            f"{e['action']}({', '.join(e['args'])}) -> {e['result']}"
            + (f" [approved by {e['approved_by']}]" if e.get("approved_by") else "")
            for e in state.get("execution_log", [])
        ],
        "replans": [r["reason"] for r in state.get("replans", [])],
        "duration_s": time.time() - state.get("started_at", time.time()),
        "status": status
        + (
            f"; unmet goals: {', '.join(g[0] + '(' + ', '.join(g[1:]) + ')' for g in unmet)}"
            if unmet
            else ""
        ),
    }
    comms = write_comms(facts)
    return {"status": status, "comms": comms, "report": comms["report"]}


# --- routing -----------------------------------------------------------------
def after_triage(state: IncidentState) -> str:
    return "plan" if state.get("triage") else END


def after_plan(state: IncidentState) -> str:
    if state.get("status") == "failed":
        # a failed first plan ends the run; a failed replan still reports to customers
        return "communicate" if state.get("execution_log") else END
    return "execute"


def after_execute(state: IncidentState) -> str:
    if state.get("status") == "rejected":
        return "communicate"
    if state.get("replan_reason"):
        return "plan"
    if state.get("step_index", 0) >= len(state["plan"]["steps"]):
        return "communicate"
    return "execute"


def build_graph(checkpointer=None):
    g = StateGraph(IncidentState)
    g.add_node("triage", triage_node)
    g.add_node("plan", plan_node)
    g.add_node("execute", execute_node)
    g.add_node("communicate", communicate_node)
    g.add_edge(START, "triage")
    g.add_conditional_edges("triage", after_triage, ["plan", END])
    g.add_conditional_edges("plan", after_plan, ["execute", "communicate", END])
    g.add_conditional_edges("execute", after_execute, ["execute", "plan", "communicate"])
    g.add_edge("communicate", END)
    return g.compile(checkpointer=checkpointer or MemorySaver())


def initial_state(
    incident_text: str,
    network: NetworkState,
    goal_mode: str = "default",
    *,
    inject_failure: bool = False,
    auto_approve: bool = False,
) -> IncidentState:
    return {
        "incident_text": incident_text,
        "network": to_dict(network),
        "goal_mode": goal_mode,
        "goal_facts": [list(g) for g in goal_facts(network.to_facts(), goal_mode)],
        "inject_failure": inject_failure,
        "auto_approve": auto_approve,
        "started_at": time.time(),
        "execution_log": [],
        "replans": [],
        "status": "running",
        "errors": [],
    }


def config_for(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}


def pending_interrupt(graph, thread_id: str) -> dict | None:
    """The approval request the graph is paused on, if any."""
    snapshot = graph.get_state(config_for(thread_id))
    for task in snapshot.tasks:
        if task.interrupts:
            return task.interrupts[0].value
    return None


def resume(graph, thread_id: str, approved: bool, by: str = "operator") -> IncidentState:
    return graph.invoke(Command(resume={"approved": approved, "by": by}), config_for(thread_id))


def run_incident(
    incident_text: str,
    network: NetworkState,
    goal_mode: str = "default",
    graph=None,
    *,
    inject_failure: bool = False,
    auto_approve: bool = True,
) -> tuple[IncidentState, str]:
    """Start the graph; returns the state (paused or final) and the thread id."""
    graph = graph or build_graph()
    thread_id = str(uuid.uuid4())
    final = graph.invoke(
        initial_state(
            incident_text,
            network,
            goal_mode,
            inject_failure=inject_failure,
            auto_approve=auto_approve,
        ),
        config_for(thread_id),
    )
    return final, thread_id
