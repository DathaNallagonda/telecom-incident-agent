"""Hand-written Day 1 scenarios (see the plan doc's scenario table)."""

from __future__ import annotations

from core.simulator import NetworkState, inject_failure, sample_network

SCENARIOS = {
    "A": {"title": "T3 power loss", "kind": "power", "node": "T3"},
    "B": {"title": "T3 hardware fault", "kind": "hardware", "node": "T3"},
    "C": {"title": "T2 backhaul link cut", "kind": "link", "node": "T2"},
}


def scenario_state(key: str) -> NetworkState:
    sc = SCENARIOS[key]
    return inject_failure(sample_network(), sc["kind"], sc["node"])
