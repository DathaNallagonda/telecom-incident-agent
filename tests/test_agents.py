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


@pytest.fixture
def fake_llms(monkeypatch):
    """Triage, planner and comms without any LLM call."""
    monkeypatch.setattr(graph_mod, "run_triage", lambda text, net: FakeOutcome(triage_for()))
    real = graph_mod.plan_with_verification
    monkeypatch.setattr(
        graph_mod,
        "plan_with_verification",
        lambda t, n, g: real(t, n, g, proposer=scripted(draft(GOOD_A))),
    )
    monkeypatch.setattr(
        graph_mod, "write_comms", lambda facts: {"sms": [], "report": "# report", "source": "x"}
    )


def test_graph_runs_end_to_end_with_auto_approval(fake_llms):
    final, thread = graph_mod.run_incident("no signal", scenario_state("A"))
    assert final["first_plan"]["source"] == "llm_verified"
    assert final["status"] == "completed" and final["errors"] == []
    assert [e["result"] for e in final["execution_log"]] == ["ok"] * 6
    restart = next(e for e in final["execution_log"] if e["action"] == "restart_node")
    assert restart["approved_by"] == "auto"
    assert final["report"] == "# report"


def test_graph_pauses_for_approval_and_resumes(fake_llms):
    graph = graph_mod.build_graph()
    state, thread = graph_mod.run_incident(
        "no signal", scenario_state("A"), graph=graph, auto_approve=False
    )
    pending = graph_mod.pending_interrupt(graph, thread)
    assert pending["step"]["action"] == "restart_node" and pending["index"] == 4
    assert len(state["execution_log"]) == 3  # paused before the risky step
    final = graph_mod.resume(graph, thread, approved=True, by="alice")
    assert final["status"] == "completed"
    assert final["execution_log"][3]["approved_by"] == "alice"


def test_graph_stops_when_operator_rejects(fake_llms):
    graph = graph_mod.build_graph()
    _, thread = graph_mod.run_incident(
        "no signal", scenario_state("A"), graph=graph, auto_approve=False
    )
    final = graph_mod.resume(graph, thread, approved=False)
    assert final["status"] == "rejected"
    assert final["execution_log"][-1]["result"] == "rejected"
    assert final["report"] == "# report"  # customers are still told


def test_graph_replans_after_injected_restart_failure(fake_llms):
    final, _ = graph_mod.run_incident("no signal", scenario_state("A"), inject_failure=True)
    assert final["status"] == "completed"
    assert len(final["replans"]) == 1 and "failed" in final["replans"][0]["reason"]
    assert final["plan"]["source"] == "replan"
    assert any(e["action"] == "repair_hardware" for e in final["execution_log"])


def test_graph_ends_with_error_when_triage_fails(monkeypatch):
    monkeypatch.setattr(graph_mod, "run_triage", lambda t, n: FakeOutcome(None, "triage failed"))
    final, _ = graph_mod.run_incident("???", scenario_state("A"))
    assert final["errors"] == ["triage failed"]
    assert final.get("plan") is None


# --- operator ----------------------------------------------------------------
def test_execute_plan_counts_invalid_attempts_without_checks():
    from agents.operator import execute_plan
    from core.spec import goal_facts

    state = scenario_state("A")
    goals = goal_facts(state.to_facts())
    bad = [{"action": a, "args": args} for a, args in BAD_A]
    r = execute_plan(state, bad, goals, check_first=False, allow_replan=False)
    assert r.attempted_invalid == 1 and not r.goal_reached


def test_execute_plan_replans_blocked_steps_when_checking():
    from agents.operator import execute_plan
    from core.spec import goal_facts

    state = scenario_state("A")
    goals = goal_facts(state.to_facts())
    bad = [{"action": a, "args": args} for a, args in BAD_A]
    r = execute_plan(state, bad, goals)
    assert r.goal_reached and r.attempted_invalid == 0 and len(r.replans) == 1


# --- comms -------------------------------------------------------------------
def test_comms_rejects_long_or_internal_sms_and_falls_back():
    from agents.comms import CommsDraft, write_comms

    bad = CommsDraft(sms=[{"sector_id": "S3", "text": "Tower T3 is down"}], report_markdown="# r")
    facts = {
        "incident_text": "x",
        "faults": {"T3": "power"},
        "affected_sectors": ["S3"],
        "restored": True,
        "actions": [],
        "replans": [],
        "duration_s": 1,
        "status": "ok",
    }
    out = write_comms(facts, writer=scripted(bad, bad))
    sms = out["sms"][0]["text"]
    assert "T3" not in sms and "City Mall" in sms and len(sms) <= 160
    assert out["report"] == "# r"


def test_comms_template_when_llm_unavailable():
    from agents.comms import write_comms

    facts = {
        "incident_text": "x",
        "faults": {},
        "affected_sectors": ["S2"],
        "restored": False,
        "actions": ["a"],
        "replans": [],
        "duration_s": 3,
        "status": "failed",
    }
    out = write_comms(facts, writer=scripted(RuntimeError("down")))
    assert out["source"] == "template" and "Railway Station" in out["sms"][0]["text"]
    assert "## Root cause" in out["report"]


# --- free-tier changes: observed sectors, paired baseline ---------------------
def test_planner_restores_sectors_triage_missed():
    """M5-style miss: triage lists no sector, but S3 is visibly without service."""
    from agents.planner_agent import affected_sectors

    t = triage_for(sectors=())
    assert affected_sectors(scenario_state("A"), t) == ["S3"]
    r = plan_with_verification(t, scenario_state("A"), proposer=scripted(draft(GOOD_A)))
    assert r.valid and any(s["action"] == "restore_service" for s in r.steps)


def test_paired_baseline_is_the_first_unchecked_proposal():
    from agents.planner_agent import baseline_from_first_round

    seen = []

    def respond(messages):
        seen.append(1)
        return draft(BAD_A) if len(seen) == 1 else draft(GOOD_A)

    r = plan_with_verification(triage_for(), scenario_state("A"), proposer=RunnableLambda(respond))
    b = baseline_from_first_round(r)
    assert r.valid and not b.valid and b.source == "llm_only"
    assert b.steps == [{"action": a, "args": args} for a, args in BAD_A]


def test_paired_baseline_when_llm_unavailable():
    from agents.planner_agent import PlanResult, RoundLog, baseline_from_first_round

    r = PlanResult([], "planner_fallback", True, 0, [RoundLog(0, [], False, "no key")])
    b = baseline_from_first_round(r)
    assert not b.valid and b.message == "no key"
