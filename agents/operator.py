"""Operator Agent: executes a verified plan on the network, one step at a time.

- Before each step, preconditions are re-checked against the live network.
- Risky steps (restart_node, reroute_traffic) need human approval.
- If a step fails (e.g. a restart fails) or its preconditions no longer hold,
  the operator replans from the current state with pyperplan, up to MAX_REPLANS.
- Every step is logged: action, args, result, time and who approved it.

`execute_step` is the single-step core used by the LangGraph execute node
(which pauses for approval with interrupt()) and by `execute_plan`, the
non-interactive loop the evaluation uses.
"""

from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass, field
from typing import Literal

from core.planner import solve
from core.problem_builder import build_problem
from core.simulator import NetworkState, apply_action, check_preconditions, goals_met
from core.spec import RISKY, Fact

MAX_REPLANS = 3

Outcome = Literal["ok", "failed", "blocked", "rejected"]


def is_risky(step: dict) -> bool:
    return step["action"] in RISKY


def fmt(step: dict) -> str:
    return f"{step['action']}({', '.join(step['args'])})"


@dataclass
class StepRecord:
    index: int  # position in the plan being executed (1-based)
    action: str
    args: list[str]
    result: Outcome
    message: str
    timestamp: str
    approved_by: str | None = None  # "operator", "auto" or None for safe steps
    plan_version: int = 0  # 0 = original plan, n = after n replans

    def to_dict(self) -> dict:
        return asdict(self)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def execute_step(
    network: NetworkState,
    step: dict,
    index: int,
    approved_by: str | None,
    *,
    plan_version: int = 0,
    restart_failure_prob: float = 0.0,
    rng: random.Random | None = None,
) -> StepRecord:
    """Re-check preconditions, then apply one step to the live network in place.

    The caller must already have obtained approval for risky steps.
    """
    args = tuple(step["args"])
    violated = check_preconditions(network, step["action"], args)
    if violated:
        return StepRecord(
            index,
            step["action"],
            list(args),
            "blocked",
            "preconditions no longer hold: " + ", ".join(violated),
            _now(),
            approved_by,
            plan_version,
        )
    res = apply_action(
        network, step["action"], args, restart_failure_prob=restart_failure_prob, rng=rng
    )
    return StepRecord(
        index,
        step["action"],
        list(args),
        "ok" if res.ok else "failed",
        res.message,
        _now(),
        approved_by,
        plan_version,
    )


def replan(network: NetworkState, goals: str | list[Fact] = "default"):
    """New plan from the current live state (the failure has revealed the true fault)."""
    return solve(build_problem(network, goals))


@dataclass
class ExecutionResult:
    log: list[StepRecord]
    status: Literal["completed", "rejected", "failed"]
    replans: list[dict] = field(default_factory=list)  # {reason, new_plan}
    goals_unmet: list[str] = field(default_factory=list)
    attempted_invalid: int = 0  # steps run although their preconditions failed

    @property
    def goal_reached(self) -> bool:
        return not self.goals_unmet


def execute_plan(
    network: NetworkState,
    steps: list[dict],
    goal_facts: list[Fact],
    *,
    goals: str | list[Fact] = "default",
    approve=lambda step: True,
    allow_replan: bool = True,
    check_first: bool = True,
    fail_first_restart: bool = False,
    seed: int = 0,
) -> ExecutionResult:
    """Run a plan to the end without pausing (used by the evaluation).

    check_first=False models an unverified agent: it tries every step, and a
    step whose preconditions fail counts as an unsafe/invalid action attempted.
    fail_first_restart=True makes the first restart_node fail (replanning test).
    """
    rng = random.Random(seed)
    log: list[StepRecord] = []
    replans: list[dict] = []
    attempted_invalid = 0
    restart_failed_once = False
    plan, version, i = list(steps), 0, 0
    while i < len(plan):
        step = plan[i]
        approved_by = None
        if is_risky(step):
            if not approve(step):
                log.append(
                    StepRecord(
                        i + 1,
                        step["action"],
                        step["args"],
                        "rejected",
                        "rejected by operator",
                        _now(),
                        None,
                        version,
                    )
                )
                return ExecutionResult(log, "rejected", replans, _unmet(network, goal_facts))
            approved_by = "auto"
        prob = 0.0
        if fail_first_restart and step["action"] == "restart_node" and not restart_failed_once:
            prob = 1.0
        rec = execute_step(
            network,
            step,
            i + 1,
            approved_by,
            plan_version=version,
            restart_failure_prob=prob,
            rng=rng,
        )
        if prob and rec.result == "failed":
            restart_failed_once = True
        if rec.result == "blocked" and not check_first:
            attempted_invalid += 1
            rec.message = "attempted with unmet preconditions (no effect): " + rec.message
        log.append(rec)
        if rec.result == "ok" or (rec.result == "blocked" and not check_first):
            i += 1
            continue
        # failed or blocked: replan from the live state
        if not allow_replan or len(replans) >= MAX_REPLANS:
            return ExecutionResult(
                log, "failed", replans, _unmet(network, goal_facts), attempted_invalid
            )
        new = replan(network, goals)
        replans.append({"reason": f"{fmt(step)}: {rec.message}", "new_plan": new.plan})
        if new.status != "solved":
            return ExecutionResult(
                log, "failed", replans, _unmet(network, goal_facts), attempted_invalid
            )
        plan, version, i = new.plan, version + 1, 0
    unmet = _unmet(network, goal_facts)
    return ExecutionResult(
        log, "completed" if not unmet else "failed", replans, unmet, attempted_invalid
    )


def _unmet(network: NetworkState, goal_facts: list[Fact]) -> list[str]:
    return [f"{g[0]}({', '.join(g[1:])})" for g in goals_met(network, goal_facts)]
