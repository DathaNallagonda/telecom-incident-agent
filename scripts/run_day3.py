"""Day 3 check: the full graph (triage -> plan -> execute -> communicate) on A, B, C,
plus A again with an injected restart failure. Risky steps are auto-approved."""

import warnings

warnings.filterwarnings("ignore")

from agents.graph import run_incident  # noqa: E402
from core.scenarios import scenario_state  # noqa: E402

CASES = [
    ("A", "customers near the mall say no signal since 9am", False),
    (
        "B",
        "Tower 3 alarm keeps flashing, remote reset didn't help, looks like a board failure",
        False,
    ),
    ("C", "phones in sector 2 show full bars but nothing loads, data totally dead", False),
    ("A", "customers near the mall say no signal since 9am", True),
]

for key, text, inject in CASES:
    final, _ = run_incident(text, scenario_state(key), inject_failure=inject)
    steps = " -> ".join(
        f"{e['action']}({', '.join(e['args'])}):{e['result']}"
        for e in final.get("execution_log", [])
    )
    print(f"\n=== {key}{' + injected failure' if inject else ''}: status {final.get('status')}")
    print(
        "  first plan:",
        (final.get("first_plan") or {}).get("source"),
        "| replans:",
        len(final.get("replans", [])),
        "| errors:",
        final.get("errors"),
    )
    print("  executed:", steps)
    comms = final.get("comms") or {}
    print("  comms:", comms.get("source"), [m["text"] for m in comms.get("sms", [])])
