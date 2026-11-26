"""Turn a NetworkState (+ goals) into a unified-planning Problem."""

from __future__ import annotations

from unified_planning.shortcuts import Object, Problem

from core.domain import Node, Sector, domain_problem, fluents
from core.simulator import NetworkState
from core.spec import FLUENTS, GOAL_MODES, NODE, Fact, goal_facts


class UnknownIdError(ValueError):
    """Raised when a goal or fact names a node or sector that does not exist."""


def _validate_fact(state: NetworkState, fact: Fact) -> None:
    name, *args = fact
    if name not in FLUENTS:
        raise UnknownIdError(f"unknown fact {name!r}")
    types = FLUENTS[name]
    if len(args) != len(types):
        raise UnknownIdError(f"{name} takes {len(types)} argument(s), got {len(args)}")
    for value, t in zip(args, types, strict=True):
        pool = state.nodes if t == NODE else state.sectors
        if value not in pool:
            raise UnknownIdError(
                f"unknown {t.lower()} id {value!r} in {name}({', '.join(args)}); "
                f"known: {', '.join(sorted(pool))}"
            )


def build_problem(
    state: NetworkState,
    goals: str | list[Fact] = "default",
    name: str = "incident",
) -> Problem:
    """Create objects, initial values from the state, and goals.

    goals: "default" (affected sectors served + notified), "full_repair"
    (also every alarmed node up with backhaul), or an explicit list of facts.
    """
    facts = state.to_facts()
    if isinstance(goals, str):
        if goals not in GOAL_MODES:
            raise ValueError(f"unknown goal mode {goals!r}; expected one of {GOAL_MODES}")
        goal_list = goal_facts(facts, goals)
    else:
        goal_list = [tuple(g) for g in goals]
        for g in goal_list:
            _validate_fact(state, g)

    problem: Problem = domain_problem(name)
    objects = {n: Object(n, Node) for n in state.nodes}
    objects |= {s: Object(s, Sector) for s in state.sectors}
    problem.add_objects(objects.values())

    fl = fluents()

    def expr(fact: Fact):
        return fl[fact[0]](*(objects[a] for a in fact[1:]))

    for f in facts:
        problem.set_initial_value(expr(f), True)
    for g in goal_list:
        problem.add_goal(expr(g))
    return problem


def belief_state(
    state: NetworkState, faults: dict[str, str], affected_sectors: list[str]
) -> NetworkState:
    """The network as triage believes it is: the planning problem is built from this.

    Observable facts (alarms, up/down, links) come from the live state. Hidden
    facts (what kind of fault each node has, which sectors are affected) come
    from triage, so a triage mistake shows up as a plan that fails on the real
    network, which is exactly what the evaluation should measure.

    faults: node id -> "power" | "hardware" | "link" | "unknown"
    """
    belief = state.copy()
    for node_id, kind in faults.items():
        if node_id not in belief.nodes:
            raise UnknownIdError(f"unknown node id {node_id!r}")
        n = belief.nodes[node_id]
        n.power_ok = kind != "power"
        n.hardware_fault = kind == "hardware"
    for s in belief.sectors.values():
        s.affected = s.id in affected_sectors
    unknown = [s for s in affected_sectors if s not in belief.sectors]
    if unknown:
        raise UnknownIdError(f"unknown sector ids {unknown}")
    return belief
