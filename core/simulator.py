"""Simulated telecom network.

The state is stored as readable dataclasses (nodes, links, sectors) and is
converted to and from planning facts. Actions are applied through the shared
spec in core/spec.py, so the simulator and the planner always agree on what
each action needs and does.

Backhaul model: a node has backhaul when its direct link to the core router
is up. Neighbour links (e.g. T1-T2) exist for the graph view only.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import networkx as nx

from core.spec import ACTIONS, FLUENTS, Fact

CORE_ID = "C0"
FAILURE_KINDS = ("power", "hardware", "link")


@dataclass
class Node:
    id: str
    type: str  # "tower" or "router"
    up: bool = True
    power_ok: bool = True
    hardware_fault: bool = False
    alarm: bool = False
    diagnosed: bool = False
    technician_at: bool = False


@dataclass
class Link:
    a: str
    b: str
    up: bool = True


@dataclass
class Sector:
    id: str
    served_by: str
    backup_towers: list[str] = field(default_factory=list)
    served: bool = True
    notified: bool = False
    affected: bool = False


@dataclass
class NetworkState:
    nodes: dict[str, Node]
    links: list[Link]
    sectors: dict[str, Sector]

    # --- convenience views -------------------------------------------------
    @property
    def alarms(self) -> list[str]:
        return [n.id for n in self.nodes.values() if n.alarm]

    @property
    def faults(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for n in self.nodes.values():
            kinds = []
            if not n.power_ok:
                kinds.append("power")
            if n.hardware_fault:
                kinds.append("hardware")
            if n.id != CORE_ID and not self.backhaul_ok(n.id):
                kinds.append("link")
            if kinds:
                out[n.id] = kinds
        return out

    @property
    def technician_locations(self) -> list[str]:
        return [n.id for n in self.nodes.values() if n.technician_at]

    def link(self, a: str, b: str) -> Link | None:
        for lk in self.links:
            if {lk.a, lk.b} == {a, b}:
                return lk
        return None

    def backhaul_ok(self, node_id: str) -> bool:
        if node_id == CORE_ID:
            return True
        lk = self.link(node_id, CORE_ID)
        return bool(lk and lk.up)

    def sectors_of(self, node_id: str) -> list[Sector]:
        return [s for s in self.sectors.values() if s.served_by == node_id]

    def graph(self) -> nx.Graph:
        g = nx.Graph()
        for n in self.nodes.values():
            g.add_node(n.id, **asdict(n))
        for lk in self.links:
            g.add_edge(lk.a, lk.b, up=lk.up)
        return g

    def copy(self) -> NetworkState:
        return from_dict(to_dict(self))

    # --- facts -------------------------------------------------------------
    def to_facts(self) -> set[Fact]:
        facts: set[Fact] = set()
        for n in self.nodes.values():
            i = n.id
            if n.alarm:
                facts.add(("alarm", i))
            if n.diagnosed:
                facts.add(("diagnosed", i))
            if n.up:
                facts.add(("node_up", i))
            facts.add(("power_ok", i) if n.power_ok else ("power_lost", i))
            facts.add(("hw_fault", i) if n.hardware_fault else ("hw_ok", i))
            facts.add(("backhaul_ok", i) if self.backhaul_ok(i) else ("link_cut", i))
            if n.technician_at:
                facts.add(("technician_at", i))
        for s in self.sectors.values():
            facts.add(("serves", s.served_by, s.id))
            for b in s.backup_towers:
                facts.add(("can_backup", b, s.id))
            if s.affected:
                facts.add(("affected", s.id))
            if s.served:
                facts.add(("sector_served", s.id))
            if s.notified:
                facts.add(("customers_notified", s.id))
        return facts

    def set_facts(self, facts: set[Fact]) -> None:
        """Write dynamic facts back into the dataclasses (static facts are ignored)."""
        for n in self.nodes.values():
            i = n.id
            n.alarm = ("alarm", i) in facts
            n.diagnosed = ("diagnosed", i) in facts
            n.up = ("node_up", i) in facts
            n.power_ok = ("power_ok", i) in facts
            n.hardware_fault = ("hw_fault", i) in facts
            n.technician_at = ("technician_at", i) in facts
            if i != CORE_ID:
                lk = self.link(i, CORE_ID)
                if lk is not None:
                    lk.up = ("backhaul_ok", i) in facts
        for s in self.sectors.values():
            s.affected = ("affected", s.id) in facts
            s.served = ("sector_served", s.id) in facts
            s.notified = ("customers_notified", s.id) in facts


# --- sample network ----------------------------------------------------------
def sample_network() -> NetworkState:
    """1 core router, 6 towers, 8 links, 6 sectors (see the plan doc)."""
    g = nx.Graph()
    g.add_node(CORE_ID, type="router")
    towers = [f"T{i}" for i in range(1, 7)]
    g.add_nodes_from(towers, type="tower")
    g.add_edges_from((t, CORE_ID) for t in towers)  # 6 backhaul links
    g.add_edges_from([("T1", "T2"), ("T4", "T5")])  # 2 neighbour links
    backups = {
        "S1": ["T2"],
        "S2": ["T1"],
        "S3": [],
        "S4": ["T5"],
        "S5": ["T4", "T6"],
        "S6": ["T5"],
    }
    nodes = {n: Node(id=n, type=d["type"]) for n, d in g.nodes(data=True)}
    links = [Link(a, b) for a, b in g.edges()]
    sectors = {s: Sector(id=s, served_by=f"T{s[1:]}", backup_towers=b) for s, b in backups.items()}
    return NetworkState(nodes=nodes, links=links, sectors=sectors)


# --- persistence -------------------------------------------------------------
def to_dict(state: NetworkState) -> dict:
    return {
        "nodes": [asdict(n) for n in state.nodes.values()],
        "links": [asdict(lk) for lk in state.links],
        "sectors": [asdict(s) for s in state.sectors.values()],
    }


def from_dict(data: dict) -> NetworkState:
    return NetworkState(
        nodes={d["id"]: Node(**d) for d in data["nodes"]},
        links=[Link(**d) for d in data["links"]],
        sectors={d["id"]: Sector(**d) for d in data["sectors"]},
    )


def save_network(state: NetworkState, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(to_dict(state), indent=2), encoding="utf-8")


def load_network(path: str | Path) -> NetworkState:
    return from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# --- actions -----------------------------------------------------------------
@dataclass
class ActionResult:
    ok: bool
    action: str
    args: tuple[str, ...]
    message: str
    violated: list[str] = field(default_factory=list)


def _fmt(f: Fact) -> str:
    return f"{f[0]}({', '.join(f[1:])})"


def _check_ids(state: NetworkState, name: str, args: tuple[str, ...]) -> list[str]:
    spec = ACTIONS[name]
    errors = []
    for (pname, ptype), value in zip(spec.params, args, strict=True):
        pool = state.nodes if ptype == "Node" else state.sectors
        if value not in pool:
            errors.append(f"unknown {ptype.lower()} id {value!r} for parameter {pname}")
    return errors


def check_preconditions(state: NetworkState, action: str, args: tuple[str, ...]) -> list[str]:
    """Return the violated conditions (empty list = the action may run)."""
    if action not in ACTIONS:
        return [f"unknown action {action!r}"]
    args = tuple(args)
    try:
        pre, _, _ = ACTIONS[action].ground(args)
    except ValueError as e:
        return [str(e)]
    id_errors = _check_ids(state, action, args)
    if id_errors:
        return id_errors
    facts = state.to_facts()
    return [_fmt(p) for p in pre if p not in facts]


def apply_action(
    state: NetworkState,
    action: str,
    args: tuple[str, ...],
    *,
    restart_failure_prob: float = 0.0,
    rng: random.Random | None = None,
) -> ActionResult:
    """Apply one action to the state in place, exactly as its effects say.

    If preconditions do not hold, nothing changes and ok=False.
    random_failure_on_execute: with restart_failure_prob > 0 a restart_node
    can fail; the node then reveals a hardware fault and stays down, which
    forces the planner to replan.
    """
    args = tuple(args)
    violated = check_preconditions(state, action, args)
    if violated:
        return ActionResult(False, action, args, "preconditions not met", violated)

    if action == "restart_node" and restart_failure_prob > 0:
        rng = rng or random.Random()
        if rng.random() < restart_failure_prob:
            node = state.nodes[args[0]]
            node.hardware_fault = True
            node.up = False
            node.alarm = True
            return ActionResult(
                False, action, args, f"restart of {args[0]} failed: hardware fault detected"
            )

    _, add, delete = ACTIONS[action].ground(args)
    facts = state.to_facts()
    facts -= set(delete)
    facts |= set(add)
    state.set_facts(facts)
    return ActionResult(True, action, args, "ok")


def inject_failure(state: NetworkState, kind: str, node: str) -> NetworkState:
    """Break a node in place: power loss, hardware fault or backhaul link cut."""
    if kind not in FAILURE_KINDS:
        raise ValueError(f"unknown failure kind {kind!r}; expected one of {FAILURE_KINDS}")
    if node not in state.nodes:
        raise ValueError(f"unknown node id {node!r}")
    n = state.nodes[node]
    n.alarm = True
    n.diagnosed = False
    if kind == "power":
        n.power_ok = False
        n.up = False
    elif kind == "hardware":
        n.hardware_fault = True
        n.up = False
    else:  # link
        if node == CORE_ID:
            raise ValueError("the core router has no backhaul link to cut")
        state.link(node, CORE_ID).up = False
    for s in state.sectors_of(node):
        s.served = False
        s.affected = True
        s.notified = False
    return state


def goals_met(state: NetworkState, goals: list[Fact]) -> list[Fact]:
    """Return the goals that do NOT hold (empty list = all met)."""
    facts = state.to_facts()
    return [g for g in goals if g not in facts]


assert set(FLUENTS) >= {a for spec in ACTIONS.values() for a, *_ in spec.pre}
