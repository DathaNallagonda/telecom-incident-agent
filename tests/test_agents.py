"""Day 2 agent tests. LLMs are replaced by scripted fakes: no network calls."""

import json

import pytest
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

import agents.graph as graph_mod
from agents.planner_agent import baseline_plan, plan_with_verification
from agents.schemas import PlanDraft, TriageResult
from agents.tools import make_tools
from agents.triage import report
from core.problem_builder import UnknownIdError, belief_state
from core.scenarios import scenario_state

GOOD_A = [
    ("diagnose", ["T3"]),
    ("dispatch_technician", ["T3"]),
    ("restore_power", ["T3"]),
    ("restart_node", ["T3"]),
    ("restore_service", ["S3", "T3"]),
    ("notify_customers", ["S3"]),
]
BAD_A = [("restart_node", ["T3"]), ("notify_customers", ["S3"])]


def draft(steps):
    return PlanDraft(steps=[{"action": a, "args": args} for a, args in steps])


def scripted(*outputs):
    """A fake LLM that returns the given outputs in order (exceptions are raised)."""
    queue = list(outputs)

    def respond(_messages):
        out = queue.pop(0)
        if isinstance(out, Exception):
            raise out
        return out

    return RunnableLambda(respond)


def triage_for(node="T3", kind="power", sectors=("S3",)):
    return TriageResult(
        faulty_nodes=[node],
        fault_types=[{"node_id": node, "fault_type": kind}],
        affected_sectors=list(sectors),
        cut_links=[],
        priority="P1",
        confidence=0.9,
        reasoning="test",
    )


# --- schemas -----------------------------------------------------------------
def test_triage_rejects_unknown_ids_with_network_context():
    data = triage_for().model_dump() | {"faulty_nodes": ["T9"]}
    data["fault_types"] = [{"node_id": "T9", "fault_type": "power"}]
    with pytest.raises(ValidationError, match="unknown node ids"):
        TriageResult.model_validate(data, context={"network": scenario_state("A")})


def test_triage_requires_fault_type_per_node():
    data = triage_for().model_dump() | {"fault_types": []}
    with pytest.raises(ValidationError, match="no entry for faulty nodes"):
        TriageResult.model_validate(data)


def test_triage_rejects_bad_enums_and_confidence():
    with pytest.raises(ValidationError):
        TriageResult.model_validate(triage_for().model_dump() | {"priority": "urgent"})
    with pytest.raises(ValidationError, match="confidence"):
        TriageResult.model_validate(triage_for().model_dump() | {"confidence": 7})


# --- tools -------------------------------------------------------------------
def test_tools_report_the_simulated_network():
    tools = {t.name: t for t in make_tools(scenario_state("A"))}
    assert json.loads(tools["get_alarms"].invoke({})) == [{"node_id": "T3", "type": "tower"}]
    assert json.loads(tools["run_diagnostics"].invoke({"node_id": "T3"}))["faults"] == ["power"]
    sectors = json.loads(tools["get_sectors_for_node"].invoke({"node_id": "T3"}))
    assert sectors == [{"sector_id": "S3", "has_service": False, "backup_towers": []}]
    assert "error" in json.loads(tools["get_node_status"].invoke({"node_id": "T9"}))
    link = json.loads(make_tools(scenario_state("C"))[2].invoke({"a": "T2", "b": "C0"}))
    assert link["status"] == "down"


# --- triage report stage -----------------------------------------------------
def test_report_retries_once_after_validation_error():
    bad = triage_for().model_dump() | {"affected_sectors": ["S9"]}
    result, error = report("x", "", scenario_state("A"), structured=scripted(bad, triage_for()))
    assert error is None and result.affected_sectors == ["S3"]


def test_report_gives_clear_error_after_two_failures():
    bad = triage_for().model_dump() | {"affected_sectors": ["S9"]}
    result, error = report("x", "", scenario_state("A"), structured=scripted(bad, bad))
    assert result is None and "triage failed after 2 attempts" in error


# --- belief state ------------------------------------------------------------
def test_belief_state_uses_triage_for_hidden_facts():
    truth = scenario_state("A")  # really a power fault
    belief = belief_state(truth, {"T3": "hardware"}, ["S3"])
    assert belief.nodes["T3"].hardware_fault and belief.nodes["T3"].power_ok
    assert not truth.nodes["T3"].hardware_fault  # the real state is untouched
    with pytest.raises(UnknownIdError):
        belief_state(truth, {"T9": "power"}, [])


# --- verify-and-fix loop -----------------------------------------------------
def test_loop_accepts_valid_first_plan():
    r = plan_with_verification(triage_for(), scenario_state("A"), proposer=scripted(draft(GOOD_A)))
    assert (r.source, r.valid, r.rounds_used) == ("llm_verified", True, 1)


def test_loop_feeds_back_the_failing_step_and_fixes_it():
    seen = []

    def respond(messages):
        seen.append(messages[-1].content)
        return draft(BAD_A) if len(seen) == 1 else draft(GOOD_A)

    r = plan_with_verification(triage_for(), scenario_state("A"), proposer=RunnableLambda(respond))
    assert (r.source, r.rounds_used) == ("llm_verified", 2)
    assert "step 1: restart_node(T3) needs diagnosed(T3)" in seen[1]


def test_loop_falls_back_to_pyperplan_after_three_fix_rounds():
    proposer = scripted(draft(BAD_A), RuntimeError("bad json"), draft(BAD_A), draft(BAD_A))
    r = plan_with_verification(triage_for(), scenario_state("A"), proposer=proposer)
    assert r.source == "planner_fallback" and r.valid
    assert r.rounds_used == 4 and len(r.validation_log) == 4
    assert len(r.steps) == 6


def test_baseline_is_scored_but_never_fixed():
    r = baseline_plan(triage_for(), scenario_state("A"), proposer=scripted(draft(BAD_A)))
    assert r.source == "llm_only" and not r.valid
    assert r.steps == [{"action": a, "args": args} for a, args in BAD_A]
    assert "needs" in r.message


# --- supervisor graph --------------------------------------------------------
class FakeOutcome:
    def __init__(self, result, error=None):
        self.result, self.error, self.transcript, self.model = result, error, [], "fake"


def test_graph_runs_triage_plan_execute_communicate(monkeypatch):
    monkeypatch.setattr(graph_mod, "run_triage", lambda text, net: FakeOutcome(triage_for()))
    real = graph_mod.plan_with_verification
    monkeypatch.setattr(
        graph_mod,
        "plan_with_verification",
        lambda t, n, g: real(t, n, g, proposer=scripted(draft(GOOD_A))),
    )
    final, thread = graph_mod.run_incident("no signal", scenario_state("A"))
    assert final["plan"]["source"] == "llm_verified"
    assert final["execution_log"] == [] and final["report"] is None  # Day 3 stubs
    assert final["errors"] == []
    assert thread


def test_graph_ends_with_error_when_triage_fails(monkeypatch):
    monkeypatch.setattr(graph_mod, "run_triage", lambda t, n: FakeOutcome(None, "triage failed"))
    final, _ = graph_mod.run_incident("???", scenario_state("A"))
    assert final["errors"] == ["triage failed"]
    assert "plan" not in final
