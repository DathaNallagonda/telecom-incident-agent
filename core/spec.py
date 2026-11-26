"""Single source of truth for the planning domain.

Both the symbolic planner (core/domain.py) and the network simulator
(core/simulator.py) read these definitions, so they cannot drift apart.

All preconditions are positive (pyperplan is STRIPS-only); negations are
modelled with paired facts such as power_ok / power_lost.
"""

from __future__ import annotations

from dataclasses import dataclass

NODE = "Node"
SECTOR = "Sector"

# fluent name -> parameter types
FLUENTS: dict[str, tuple[str, ...]] = {
    "alarm": (NODE,),
    "diagnosed": (NODE,),
    "node_up": (NODE,),
    "power_ok": (NODE,),
    "power_lost": (NODE,),
    "hw_ok": (NODE,),
    "hw_fault": (NODE,),
    "backhaul_ok": (NODE,),
    "link_cut": (NODE,),
    "technician_at": (NODE,),
    "serves": (NODE, SECTOR),
    "can_backup": (NODE, SECTOR),
    "affected": (SECTOR,),
    "sector_served": (SECTOR,),
    "customers_notified": (SECTOR,),
}

# A fact is a tuple: (fluent_name, arg1, arg2, ...)
Fact = tuple[str, ...]
# A literal inside an action schema refers to parameter names, not object ids.
Literal = tuple[str, ...]


@dataclass(frozen=True)
class ActionSpec:
    name: str
    params: tuple[tuple[str, str], ...]  # (param_name, type)
    pre: tuple[Literal, ...]
    add: tuple[Literal, ...] = ()
    delete: tuple[Literal, ...] = ()
    risky: bool = False
    description: str = ""

    @property
    def param_names(self) -> tuple[str, ...]:
        return tuple(p for p, _ in self.params)

    def ground(self, args: tuple[str, ...]) -> tuple[list[Fact], list[Fact], list[Fact]]:
        """Return (preconditions, add effects, delete effects) as concrete facts."""
        if len(args) != len(self.params):
            raise ValueError(
                f"{self.name} takes {len(self.params)} argument(s) "
                f"({', '.join(self.param_names)}), got {len(args)}"
            )
        binding = dict(zip(self.param_names, args, strict=True))

        def sub(lits: tuple[Literal, ...]) -> list[Fact]:
            return [(lit[0], *(binding[a] for a in lit[1:])) for lit in lits]

        return sub(self.pre), sub(self.add), sub(self.delete)


N = (("n", NODE),)

ACTIONS: dict[str, ActionSpec] = {
    a.name: a
    for a in [
        ActionSpec(
            "diagnose",
            N,
            pre=(("alarm", "n"),),
            add=(("diagnosed", "n"),),
            description="Run remote diagnostics on an alarmed node",
        ),
        ActionSpec(
            "dispatch_technician",
            N,
            pre=(("diagnosed", "n"),),
            add=(("technician_at", "n"),),
            description="Send a field technician to a diagnosed node",
        ),
        ActionSpec(
            "restore_power",
            N,
            pre=(("technician_at", "n"), ("power_lost", "n")),
            add=(("power_ok", "n"),),
            delete=(("power_lost", "n"),),
            description="Technician restores power at the site",
        ),
        ActionSpec(
            "repair_hardware",
            N,
            pre=(("technician_at", "n"), ("hw_fault", "n")),
            add=(("hw_ok", "n"),),
            delete=(("hw_fault", "n"),),
            description="Technician replaces the faulty hardware",
        ),
        ActionSpec(
            "repair_link",
            N,
            pre=(("technician_at", "n"), ("link_cut", "n")),
            add=(("backhaul_ok", "n"),),
            delete=(("link_cut", "n"), ("alarm", "n")),
            description="Technician repairs the backhaul link to the core",
        ),
        ActionSpec(
            "restart_node",
            N,
            pre=(("diagnosed", "n"), ("power_ok", "n"), ("hw_ok", "n")),
            add=(("node_up", "n"),),
            delete=(("alarm", "n"),),
            risky=True,
            description="Restart the node (needs power and healthy hardware)",
        ),
        ActionSpec(
            "restore_service",
            (("s", SECTOR), ("n", NODE)),
            pre=(("serves", "n", "s"), ("node_up", "n"), ("backhaul_ok", "n")),
            add=(("sector_served", "s"),),
            description="Hand a sector back to its own tower once it is up",
        ),
        ActionSpec(
            "reroute_traffic",
            (("s", SECTOR), ("to", NODE)),
            pre=(
                ("affected", "s"),
                ("can_backup", "to", "s"),
                ("node_up", "to"),
                ("backhaul_ok", "to"),
            ),
            add=(("sector_served", "s"),),
            risky=True,
            description="Move a sector's traffic onto a healthy backup tower",
        ),
        ActionSpec(
            "notify_customers",
            (("s", SECTOR),),
            pre=(("affected", "s"),),
            add=(("customers_notified", "s"),),
            description="Send customers in the sector a service notice",
        ),
    ]
}

RISKY: frozenset[str] = frozenset(name for name, a in ACTIONS.items() if a.risky)

GOAL_MODES = ("default", "full_repair")


def goal_facts(facts: set[Fact], mode: str = "default") -> list[Fact]:
    """Goals derived from the current facts.

    default:     every affected sector is served and its customers notified.
    full_repair: also every alarmed node is up with working backhaul.
    """
    if mode not in GOAL_MODES:
        raise ValueError(f"unknown goal mode {mode!r}; expected one of {GOAL_MODES}")
    goals: list[Fact] = []
    for f in sorted(facts):
        if f[0] == "affected":
            goals += [("sector_served", f[1]), ("customers_notified", f[1])]
    if mode == "full_repair":
        for f in sorted(facts):
            if f[0] == "alarm":
                goals += [("node_up", f[1]), ("backhaul_ok", f[1])]
    return goals


def describe_actions() -> str:
    """Plain-language action list (used later in LLM prompts)."""
    lines = []
    for a in ACTIONS.values():
        params = ", ".join(a.param_names)
        pre = ", ".join(f"{p[0]}({', '.join(p[1:])})" for p in a.pre)
        eff = [f"{p[0]}({', '.join(p[1:])})" for p in a.add]
        eff += [f"not {p[0]}({', '.join(p[1:])})" for p in a.delete]
        risk = " [RISKY: needs human approval]" if a.risky else ""
        lines.append(f"- {a.name}({params}): needs {pre}; effects {', '.join(eff)}.{risk}")
    return "\n".join(lines)
