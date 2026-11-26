"""unified-planning domain, built from the shared spec in core/spec.py."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from unified_planning.io import PDDLWriter
from unified_planning.shortcuts import (
    BoolType,
    Fluent,
    InstantaneousAction,
    Problem,
    UserType,
    get_environment,
)

from core.spec import ACTIONS, FLUENTS, NODE, RISKY, SECTOR

get_environment().credits_stream = None  # silence the engine banner

Node = UserType(NODE)
Sector = UserType(SECTOR)
TYPES = {NODE: Node, SECTOR: Sector}

__all__ = ["Node", "Sector", "RISKY", "fluents", "actions", "domain_problem", "export_pddl"]


@lru_cache(maxsize=1)
def fluents() -> dict[str, Fluent]:
    return {
        name: Fluent(name, BoolType(), **{f"p{i}": TYPES[t] for i, t in enumerate(types)})
        for name, types in FLUENTS.items()
    }


@lru_cache(maxsize=1)
def actions() -> dict[str, InstantaneousAction]:
    fl = fluents()
    out = {}
    for spec in ACTIONS.values():
        act = InstantaneousAction(spec.name, **{p: TYPES[t] for p, t in spec.params})
        params = {p.name: p for p in act.parameters}

        def lit(literal, params=params):
            return fl[literal[0]](*(params[a] for a in literal[1:]))

        for p in spec.pre:
            act.add_precondition(lit(p))
        for e in spec.add:
            act.add_effect(lit(e), True)
        for e in spec.delete:
            act.add_effect(lit(e), False)
        out[spec.name] = act
    return out


def domain_problem(name: str = "telecom") -> Problem:
    """An empty problem holding only the domain (fluents + actions)."""
    problem = Problem(name)
    for f in fluents().values():
        problem.add_fluent(f, default_initial_value=False)
    for a in actions().values():
        problem.add_action(a)
    return problem


def export_pddl(problem: Problem, out_dir: str | Path) -> tuple[Path, Path]:
    """Write domain.pddl and problem.pddl for inspection."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    writer = PDDLWriter(problem)
    domain_path, problem_path = out / "domain.pddl", out / "problem.pddl"
    writer.write_domain(str(domain_path))
    writer.write_problem(str(problem_path))
    return domain_path, problem_path
