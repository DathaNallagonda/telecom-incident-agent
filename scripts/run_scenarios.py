"""Solve scenarios A, B, C, print the plans and execute them on the simulator."""

from core.planner import solve, validate
from core.problem_builder import build_problem
from core.scenarios import SCENARIOS, scenario_state
from core.simulator import apply_action, goals_met
from core.spec import goal_facts

for key, sc in SCENARIOS.items():
    for mode in ("default", "full_repair"):
        state = scenario_state(key)
        goals = goal_facts(state.to_facts(), mode)
        problem = build_problem(state, mode)
        res = solve(problem)
        report = validate(problem, res.plan)
        print(
            f"\n{key} - {sc['title']} [{mode}] {res.status} "
            f"in {res.solve_ms:.0f} ms, valid={report.valid}"
        )
        for i, s in enumerate(res.plan, 1):
            r = apply_action(state, s["action"], tuple(s["args"]))
            print(f"  {i}. {s['action']}({', '.join(s['args'])}) -> {r.message}")
        print("  goals unmet after execution:", goals_met(state, goals) or "none")
