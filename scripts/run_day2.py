"""Day 2 check: plain-English incidents -> triage -> verified plan, plus the baseline."""

import warnings

warnings.filterwarnings("ignore")

from agents.graph import run_incident  # noqa: E402
from agents.planner_agent import baseline_plan  # noqa: E402
from agents.schemas import TriageResult  # noqa: E402
from core.scenarios import scenario_state  # noqa: E402

INCIDENTS = {
    "A": "customers near the mall say no signal since 9am",
    "B": "Tower 3 alarm keeps flashing, remote reset didn't help, looks like a board failure",
    "C": "phones in sector 2 show full bars but nothing loads, data totally dead",
}

for key, text in INCIDENTS.items():
    state = scenario_state(key)
    final, _ = run_incident(text, state)
    print(f"\n=== {key}: {text}")
    if final.get("errors"):
        print("  errors:", final["errors"])
    t = final.get("triage")
    if t:
        faults = ", ".join(f"{f['node_id']}:{f['fault_type']}" for f in t["fault_types"])
        print(f"  triage: {faults} | sectors {t['affected_sectors']} | {t['priority']}")
        print("  tools:", " -> ".join(c["tool"] for c in final.get("triage_calls", [])))
    p = final.get("plan")
    if p:
        steps = [f"{s['action']}({', '.join(s['args'])})" for s in p["steps"]]
        print(f"  plan [{p['source']}, rounds {p['rounds_used']}]: {' -> '.join(steps)}")
        b = baseline_plan(TriageResult.model_validate(t), state)
        print(f"  baseline valid={b.valid} ({len(b.steps)} steps) {'' if b.valid else b.message}")
