import pytest

from core.planner import solve, validate
from core.problem_builder import UnknownIdError, build_problem
from core.scenarios import scenario_state
from core.simulator import apply_action, goals_met
from core.spec import goal_facts

EXPECTED = {
    ("A", "default"): [
        "diagnose(T3)",
        "dispatch_technician(T3)",
        "restore_power(T3)",
        "restart_node(T3)",
        "restore_service(S3, T3)",
        "notify_customers(S3)",
    ],
    ("B", "default"): [
        "diagnose(T3)",
        "dispatch_technician(T3)",
        "repair_hardware(T3)",
        "restart_node(T3)",
        "restore_service(S3, T3)",
        "notify_customers(S3)",
    ],
    ("C", "default"): ["reroute_traffic(S2, T1)", "notify_customers(S2)"],
    ("C", "full_repair"): [
        "diagnose(T2)",
        "dispatch_technician(T2)",
        "repair_link(T2)",
        "restore_service(S2, T2)",
        "notify_customers(S2)",
    ],
}


def fmt(step):
    return f"{step['action']}({', '.join(step['args'])})"


@pytest.mark.parametrize("key,mode", list(EXPECTED))
def test_scenario_solves_and_executes(key, mode):
    state = scenario_state(key)
    goals = goal_facts(state.to_facts(), mode)
    problem = build_problem(state, mode)
    result = solve(problem)
    assert result.status == "solved"
    assert [fmt(s) for s in result.plan] == EXPECTED[(key, mode)]
    assert validate(problem, result.plan).valid
    for step in result.plan:
        assert apply_action(state, step["action"], tuple(step["args"])).ok
    assert goals_met(state, goals) == []


def test_validator_reports_first_failing_step():
    problem = build_problem(scenario_state("B"))
    plan = [{"action": "diagnose", "args": ["T3"]}, {"action": "restart_node", "args": ["T3"]}]
    report = validate(problem, plan)
    assert not report.valid and report.failed_step == 2
    assert report.reason == "step 2: restart_node(T3) needs hw_ok(T3)"


def test_validator_reports_unmet_goals():
    problem = build_problem(scenario_state("A"))
    report = validate(problem, [{"action": "diagnose", "args": ["T3"]}])
    assert not report.valid and report.failed_step is None
    assert "sector_served(S3)" in report.unmet_goals


@pytest.mark.parametrize(
    "plan,fragment",
    [
        ([{"action": "reboot", "args": ["T3"]}], "unknown action"),
        ([{"action": "diagnose", "args": ["T9"]}], "not a known node"),
        ([{"action": "diagnose", "args": ["S3"]}], "not a known node"),
        ([{"action": "diagnose", "args": []}], "takes 1 argument"),
    ],
)
def test_validator_rejects_malformed_steps(plan, fragment):
    report = validate(build_problem(scenario_state("A")), plan)
    assert not report.valid and fragment in report.reason


def test_unknown_ids_raise_clear_error():
    with pytest.raises(UnknownIdError, match="unknown sector id 'S9'"):
        build_problem(scenario_state("A"), [("sector_served", "S9")])


def test_unsolvable_is_reported_cleanly():
    result = solve(build_problem(scenario_state("A"), [("can_backup", "T1", "S3")]))
    assert result.status == "unsolvable"
    assert result.plan == []
