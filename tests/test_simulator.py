import random

import pytest

from core.simulator import (
    apply_action,
    check_preconditions,
    inject_failure,
    load_network,
    sample_network,
    save_network,
)
from core.spec import ACTIONS


def test_sample_network_shape():
    s = sample_network()
    assert sum(n.type == "router" for n in s.nodes.values()) == 1
    assert sum(n.type == "tower" for n in s.nodes.values()) == 6
    assert len(s.links) == 8
    assert len(s.sectors) == 6


def test_save_and_load_roundtrip(tmp_path):
    s = inject_failure(sample_network(), "power", "T3")
    path = tmp_path / "net.json"
    save_network(s, path)
    assert load_network(path).to_facts() == s.to_facts()


@pytest.mark.parametrize(
    "kind,node,expect",
    [
        ("power", "T3", {("power_lost", "T3"), ("affected", "S3")}),
        ("hardware", "T5", {("hw_fault", "T5"), ("affected", "S5")}),
        ("link", "T2", {("link_cut", "T2"), ("affected", "S2"), ("node_up", "T2")}),
    ],
)
def test_inject_failure(kind, node, expect):
    s = inject_failure(sample_network(), kind, node)
    facts = s.to_facts()
    assert expect <= facts
    assert ("alarm", node) in facts
    assert ("sector_served", f"S{node[1:]}") not in facts


def test_inject_failure_rejects_unknown():
    with pytest.raises(ValueError):
        inject_failure(sample_network(), "flood", "T1")
    with pytest.raises(ValueError):
        inject_failure(sample_network(), "power", "T9")


def _state_where_runnable(action):
    """A state in which the given action's preconditions hold."""
    s = sample_network()
    if action in ("restore_power",):
        inject_failure(s, "power", "T3")
    elif action in ("repair_hardware",):
        inject_failure(s, "hardware", "T3")
    elif action in ("repair_link", "reroute_traffic", "notify_customers", "restore_service"):
        inject_failure(s, "link", "T3" if action != "reroute_traffic" else "T2")
    else:
        inject_failure(s, "power", "T3")
        s.nodes["T3"].power_ok = True  # leave it diagnosable / restartable
    n = s.nodes["T3"]
    n.diagnosed = True
    n.technician_at = action in ("restore_power", "repair_hardware", "repair_link")
    if action == "restore_service":
        s.link("T3", "C0").up = True
    return s


ARGS = {
    "diagnose": ("T3",),
    "dispatch_technician": ("T3",),
    "restore_power": ("T3",),
    "repair_hardware": ("T3",),
    "repair_link": ("T3",),
    "restart_node": ("T3",),
    "restore_service": ("S3", "T3"),
    "reroute_traffic": ("S2", "T1"),
    "notify_customers": ("S3",),
}


@pytest.mark.parametrize("action", list(ACTIONS))
def test_each_action_changes_only_its_effects(action):
    args = ARGS[action]
    s = _state_where_runnable(action)
    assert check_preconditions(s, action, args) == []
    before = s.to_facts()
    result = apply_action(s, action, args)
    assert result.ok, result
    after = s.to_facts()
    _, add, delete = ACTIONS[action].ground(args)
    assert after == (before - set(delete)) | set(add)


def test_violated_preconditions_leave_state_untouched():
    s = inject_failure(sample_network(), "hardware", "T3")
    before = s.to_facts()
    r = apply_action(s, "restart_node", ("T3",))
    assert not r.ok
    assert set(r.violated) == {"diagnosed(T3)", "hw_ok(T3)"}
    assert s.to_facts() == before


def test_unknown_ids_and_actions():
    s = sample_network()
    assert check_preconditions(s, "reboot", ("T1",)) == ["unknown action 'reboot'"]
    assert "unknown node id" in check_preconditions(s, "diagnose", ("T9",))[0]
    assert "takes 1 argument" in check_preconditions(s, "diagnose", ("T1", "T2"))[0]


def test_restart_can_fail_on_execute():
    s = inject_failure(sample_network(), "power", "T3")
    for a in ("diagnose", "dispatch_technician", "restore_power"):
        assert apply_action(s, a, ("T3",)).ok
    r = apply_action(s, "restart_node", ("T3",), restart_failure_prob=1.0, rng=random.Random(0))
    assert not r.ok and "failed" in r.message
    assert s.nodes["T3"].hardware_fault and not s.nodes["T3"].up
