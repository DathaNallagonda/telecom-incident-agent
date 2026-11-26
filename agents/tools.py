"""Network tools the Triage Agent calls to investigate an incident.

Tools are built per incident with `make_tools(state)` so each run reads its
own network. They return short JSON strings, and unknown ids come back as an
error message rather than an exception, so the agent can correct itself.
"""

from __future__ import annotations

import json

from langchain_core.tools import BaseTool, tool

from core.simulator import CORE_ID, NetworkState


def _err(msg: str) -> str:
    return json.dumps({"error": msg})


def make_tools(state: NetworkState) -> list[BaseTool]:
    def node_or_error(node_id: str):
        node = state.nodes.get(node_id.strip().upper())
        return node, (
            None
            if node
            else _err(f"unknown node id {node_id!r}; known: {', '.join(sorted(state.nodes))}")
        )

    @tool
    def get_alarms() -> str:
        """List all nodes with an active alarm."""
        return json.dumps(
            [{"node_id": n.id, "type": n.type} for n in state.nodes.values() if n.alarm]
        )

    @tool
    def get_node_status(node_id: str) -> str:
        """Get a node's status: up/down, alarm and backhaul to the core."""
        node, err = node_or_error(node_id)
        if err:
            return err
        return json.dumps(
            {
                "node_id": node.id,
                "type": node.type,
                "up": node.up,
                "alarm": node.alarm,
                "backhaul_ok": state.backhaul_ok(node.id),
            }
        )

    @tool
    def get_link_status(a: str, b: str) -> str:
        """Get whether the link between two nodes is up or down."""
        lk = state.link(a.strip().upper(), b.strip().upper())
        if lk is None:
            return _err(f"no link between {a} and {b}")
        return json.dumps({"a": lk.a, "b": lk.b, "status": "up" if lk.up else "down"})

    @tool
    def get_sectors_for_node(node_id: str) -> str:
        """List the sectors a node serves, whether each has service, and its backup towers."""
        node, err = node_or_error(node_id)
        if err:
            return err
        return json.dumps(
            [
                {"sector_id": s.id, "has_service": s.served, "backup_towers": s.backup_towers}
                for s in state.sectors_of(node.id)
            ]
        )

    @tool
    def run_diagnostics(node_id: str) -> str:
        """Run remote diagnostics on a node; returns its fault types (power, hardware, link)."""
        node, err = node_or_error(node_id)
        if err:
            return err
        faults = []
        if not node.power_ok:
            faults.append("power")
        if node.hardware_fault:
            faults.append("hardware")
        if node.id != CORE_ID and not state.backhaul_ok(node.id):
            faults.append("link")
        return json.dumps({"node_id": node.id, "faults": faults or ["none"]})

    return [get_alarms, get_node_status, get_link_status, get_sectors_for_node, run_diagnostics]
