"""Planner Agent: the LLM-Modulo verify-and-fix loop.

1. The LLM proposes a JSON plan from the triage result and the action list.
2. The verifier (unified-planning PlanValidator) checks every step; on failure it
   returns the exact step and broken condition.
3. The LLM fixes the plan using that feedback, up to MAX_FIX_ROUNDS times.
4. Still invalid -> pyperplan solves the problem directly (planner_fallback).

The LLM never writes PDDL: it fills a strict JSON schema, and code builds the
formal problem.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from agents.schemas import PlanDraft, TriageResult
from core import llm as llm_mod
from core.planner import solve, validate
from core.problem_builder import belief_state, build_problem
from core.simulator import NetworkState
from core.spec import RISKY, describe_actions

MAX_FIX_ROUNDS = 3
Source = Literal["llm_verified", "planner_fallback", "llm_only", "failed"]

SYSTEM_PROMPT = """You are the Planner Agent for a telecom network operations centre.
Write a repair plan as JSON: a list of steps, each {{"action": name, "args": [ids]}}.

Actions (arguments in this order; every listed condition must hold before the step):
{actions}

Rules:
- Use only these actions and only ids from the network description.
- A step's conditions must already be true from the start state or earlier steps.
- Goal: every affected sector has service (sector_served) and its customers are
  notified (customers_notified).
- Prefer the shortest plan. Risky actions ({risky}) will need human approval."""


def network_description(belief: NetworkState) -> str:
    lines = ["Nodes:"]
    for n in belief.nodes.values():
        status = [
            "up" if n.up else "DOWN",
            "ALARM" if n.alarm else "no alarm",
            "power_ok" if n.power_ok else "power_lost",
            "hw_fault" if n.hardware_fault else "hw_ok",
            "backhaul_ok" if belief.backhaul_ok(n.id) else "link_cut",
        ]
        lines.append(f"- {n.id} ({n.type}): {', '.join(status)}")
    lines.append("Sectors:")
    for s in belief.sectors.values():
        backups = ", ".join(s.backup_towers) or "none"
        flags = ["AFFECTED" if s.affected else "ok", "served" if s.served else "NOT served"]
        lines.append(f"- {s.id}: served by {s.served_by}; backups: {backups}; {', '.join(flags)}")
    return "\n".join(lines)


def first_messages(triage: TriageResult, belief: NetworkState) -> list[BaseMessage]:
    system = SYSTEM_PROMPT.format(actions=describe_actions(), risky=", ".join(sorted(RISKY)))
    human = (
        f"Triage result:\n{triage.model_dump_json(indent=1)}\n\n"
        f"Current network state:\n{network_description(belief)}\n\n"
        "Return the plan."
    )
    return [SystemMessage(system), HumanMessage(human)]


@dataclass
class RoundLog:
    round: int
    steps: list[dict]
    valid: bool
    reason: str = ""


@dataclass
class PlanResult:
    steps: list[dict]
    source: Source
    valid: bool
    rounds_used: int  # LLM proposals made (1 = first try)
    validation_log: list[RoundLog] = field(default_factory=list)
    message: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _draft_steps(draft) -> list[dict]:
    if hasattr(draft, "model_dump"):
        draft = draft.model_dump()
    return [{"action": s["action"], "args": list(s["args"])} for s in draft["steps"]]


def make_problem(state: NetworkState, triage: TriageResult, goals="default"):
    faults = {f.node_id: f.fault_type for f in triage.fault_types}
    belief = belief_state(state, faults, triage.affected_sectors)
    return belief, build_problem(belief, goals)


def plan_with_verification(
    triage: TriageResult,
    state: NetworkState,
    goals="default",
    proposer: Runnable | None = None,
    max_fix_rounds: int = MAX_FIX_ROUNDS,
) -> PlanResult:
    """LLM proposes, verifier critiques, LLM fixes; pyperplan as the fallback."""
    belief, problem = make_problem(state, triage, goals)
    log: list[RoundLog] = []
    try:
        proposer = proposer or llm_mod.structured(PlanDraft)
    except llm_mod.LLMUnavailableError as e:
        proposer, log = None, [RoundLog(0, [], False, str(e))]

    if proposer is not None:
        messages = first_messages(triage, belief)
        for rnd in range(1, max_fix_rounds + 2):  # first proposal + fix rounds
            try:
                steps = _draft_steps(proposer.invoke(messages))
            except Exception as e:
                reason = f"LLM output could not be used: {type(e).__name__}: {str(e)[:200]}"
                log.append(RoundLog(rnd, [], False, reason))
                messages.append(HumanMessage(f"{reason}. Return the plan as valid JSON."))
                continue
            report = validate(problem, steps)
            log.append(RoundLog(rnd, steps, report.valid, report.reason))
            if report.valid:
                return PlanResult(steps, "llm_verified", True, rnd, log, "LLM plan verified")
            messages += [
                AIMessage(json.dumps({"steps": steps})),
                HumanMessage(
                    f"The verifier rejected this plan: {report.reason}. "
                    "Return a corrected, complete plan."
                ),
            ]

    fallback = solve(problem)
    rounds = sum(1 for r in log if r.round > 0)
    if fallback.status == "solved":
        return PlanResult(
            fallback.plan,
            "planner_fallback",
            True,
            rounds,
            log,
            f"LLM plan still invalid after {rounds} round(s); pyperplan found a "
            f"{len(fallback.plan)}-step plan",
        )
    return PlanResult([], "failed", False, rounds, log, fallback.message)


def baseline_plan(
    triage: TriageResult, state: NetworkState, goals="default", proposer: Runnable | None = None
) -> PlanResult:
    """LLM-only baseline: the same first prompt, no feedback, no fallback.

    The plan is validated afterwards only to score it, never to fix it.
    """
    belief, problem = make_problem(state, triage, goals)
    proposer = proposer or llm_mod.structured(PlanDraft)
    try:
        steps = _draft_steps(proposer.invoke(first_messages(triage, belief)))
    except Exception as e:
        reason = f"LLM output could not be used: {type(e).__name__}: {str(e)[:200]}"
        return PlanResult([], "llm_only", False, 1, [RoundLog(1, [], False, reason)], reason)
    report = validate(problem, steps)
    return PlanResult(
        steps,
        "llm_only",
        report.valid,
        1,
        [RoundLog(1, steps, report.valid, report.reason)],
        "valid" if report.valid else report.reason,
    )
