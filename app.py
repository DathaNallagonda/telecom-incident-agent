import streamlit as st

from core.planner import solve
from core.problem_builder import build_problem
from core.scenarios import SCENARIOS, scenario_state

st.title("Telecom Incident Agent")
st.caption("Day 1: the symbolic core. A planner finds a verified repair plan.")
key = st.selectbox(
    "Scenario", list(SCENARIOS), format_func=lambda k: f"{k} - {SCENARIOS[k]['title']}"
)
if st.button("Solve"):
    result = solve(build_problem(scenario_state(key)))
    st.write(f"**{result.status}** in {result.solve_ms:.0f} ms - {result.message}")
    st.table(
        [
            {"#": i, "action": s["action"], "args": ", ".join(s["args"])}
            for i, s in enumerate(result.plan, 1)
        ]
    )
