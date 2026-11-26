"""Planner tool: solve with pyperplan and validate any plan with PlanValidator."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from unified_planning.engines import PlanGenerationResultStatus, ValidationResultStatus
from unified_planning.plans import ActionInstance, SequentialPlan
from unified_planning.shortcuts import OneshotPlanner, PlanValidator, Problem, SequentialSimulator

from core.spec import ACTIONS

Step = dict  # {"action": str, "args": [str, ...]}

SOLVED = {
    PlanGenerationResultStatus.SOLVED_SATISFICING,
    PlanGenerationResultStatus.SOLVED_OPTIMALLY,
}
UNSOLVABLE = {
    PlanGenerationResultStatus.UNSOLVABLE_PROVEN,
    PlanGenerationResultStatus.UNSOLVABLE_INCOMPLETELY,
}


@dataclass
class SolveResult:
    status: str  # "solved", "unsolvable" or "error"
    plan: list[Step]
    solve_ms: float
    message: str = ""


@dataclass
class ValidationReport:
    valid: bool
    failed_step: int | None = None  # 1-based index of the first failing step
    reason: str = ""
    unmet_goals: list[str] = field(default_factory=list)


def _to_steps(plan: SequentialPlan) -> list[Step]:
    return [
        {"action": ai.action.name, "args": [str(p) for p in ai.actual_parameters]}
        for ai in plan.actions
    ]


def solve(problem: Problem, optimal: bool = True) -> SolveResult:
    """Find a plan with pyperplan (optimal A* search by default)."""
    engine = "pyperplan-opt" if optimal else "pyperplan"
    start = time.perf_counter()
    try:
        with OneshotPlanner(name=engine) as planner:
            result = planner.solve(problem)
    except Exception as e:  # engine crash should not take the app down
        return SolveResult("error", [], (time.perf_counter() - start) * 1000, f"planner error: {e}")
    ms = (time.perf_counter() - start) * 1000
    if result.status in SOLVED:
        steps = _to_steps(result.plan)
        return SolveResult("solved", steps, ms, f"found a {len(steps)}-step plan")
    if result.status in UNSOLVABLE:
        return SolveResult(
            "unsolvable",
            [],
            ms,
            "no plan can reach the goals from this state",
        )
    return SolveResult("error", [], ms, f"planner returned {result.status.name}")


def _instances(
    problem: Problem, steps: list[Step]
) -> tuple[list[ActionInstance], ValidationReport | None]:
    """Turn {action, args} dicts into action instances; report malformed steps."""
    out = []
    for i, step in enumerate(steps, start=1):
        name = step.get("action")
        args = [str(a) for a in step.get("args", [])]
        if name not in ACTIONS:
            return out, ValidationReport(False, i, f"step {i}: unknown action {name!r}")
        spec = ACTIONS[name]
        if len(args) != len(spec.params):
            return out, ValidationReport(
                False,
                i,
                f"step {i}: {name} takes {len(spec.params)} argument(s) "
                f"({', '.join(spec.param_names)}), got {len(args)}",
            )
        objs = []
        for (pname, ptype), a in zip(spec.params, args, strict=True):
            if not problem.has_object(a) or problem.object(a).type.name != ptype:
                return out, ValidationReport(
                    False, i, f"step {i}: {a!r} is not a known {ptype.lower()} (parameter {pname})"
                )
            objs.append(problem.object(a))
        out.append(ActionInstance(problem.action(name), objs))
    return out, None


def _fmt_step(step: Step) -> str:
    return f"{step['action']}({', '.join(step['args'])})"


def validate(problem: Problem, steps: list[Step]) -> ValidationReport:
    """Check a plan step by step.

    Returns valid, or the first failing step with the exact broken condition,
    or the goals still unmet at the end.
    """
    instances, error = _instances(problem, steps)
    if error:
        return error

    with PlanValidator(problem_kind=problem.kind) as validator:
        result = validator.validate(problem, SequentialPlan(instances))
    if result.status == ValidationResultStatus.VALID:
        return ValidationReport(True)

    # Replay with the simulator to name the failing step and condition precisely.
    with SequentialSimulator(problem) as sim:
        state = sim.get_initial_state()
        for i, ai in enumerate(instances, start=1):
            unmet, _reason = sim.get_unsatisfied_conditions(state, ai)
            if unmet:
                conds = ", ".join(str(c) for c in unmet)
                return ValidationReport(
                    False, i, f"step {i}: {_fmt_step(steps[i - 1])} needs {conds}"
                )
            state = sim.apply(state, ai)
        missing = [str(g) for g in sim.get_unsatisfied_goals(state)]
    return ValidationReport(
        False,
        None,
        "all steps can run, but these goals are not reached: " + ", ".join(missing),
        missing,
    )
