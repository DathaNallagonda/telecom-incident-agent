"""Build eval/scenarios.json: 20 scenarios with expected triage and optimal plan length.

Expected triage comes from the injected failures; optimal plan length comes
from pyperplan-opt on the true network, so neither is hand-counted.

Run: python -m eval.make_scenarios
"""

from __future__ import annotations

import json
from pathlib import Path

from core.planner import solve
from core.problem_builder import build_problem
from core.simulator import inject_failure, sample_network

OUT = Path(__file__).with_name("scenarios.json")

# id, type, failures [(kind, node)], goal_mode, inject_failure_during_execution, noisy, text
SCENARIOS = [
    # --- power loss: restart only after power is restored (full repair) ---
    (
        "P1",
        "power",
        [("power", "T3")],
        "full_repair",
        True,
        True,
        "customers near the mall say no signal since 9am, my phone just shows SOS",
    ),
    (
        "P2",
        "power",
        [("power", "T1")],
        "full_repair",
        False,
        False,
        "Mains failure alarm at tower T1, site on battery then dropped. Restore the site.",
    ),
    (
        "P3",
        "power",
        [("power", "T4")],
        "full_repair",
        False,
        False,
        "T4 went dark after a storm; NOC sees no power telemetry. Bring the tower back.",
    ),
    (
        "P4",
        "power",
        [("power", "T6")],
        "full_repair",
        False,
        False,
        "Airport Road tower reports power loss alarm. Full restoration of the site required.",
    ),
    (
        "P5",
        "power",
        [("power", "T5")],
        "full_repair",
        False,
        False,
        "Rectifier alarm then total outage at T5. Please restore power and the tower.",
    ),
    # --- hardware fault: technician path, restart must not come first ---
    (
        "H1",
        "hardware",
        [("hardware", "T3")],
        "full_repair",
        False,
        False,
        "Tower 3 alarm keeps flashing, remote reset didn't help, looks like a board failure",
    ),
    (
        "H2",
        "hardware",
        [("hardware", "T2")],
        "full_repair",
        False,
        False,
        "Baseband unit fault reported on T2; the radio is down. Repair the site.",
    ),
    (
        "H3",
        "hardware",
        [("hardware", "T5")],
        "full_repair",
        True,
        False,
        "T5 hardware alarm: radio module failure. Tower is down, needs a fix.",
    ),
    (
        "H4",
        "hardware",
        [("hardware", "T6")],
        "full_repair",
        False,
        False,
        "Equipment failure at the Airport Road tower T6, unit not responding.",
    ),
    (
        "H5",
        "hardware",
        [("hardware", "T1")],
        "full_repair",
        False,
        False,
        "T1 shows a hardware fault after a power surge; the tower is offline.",
    ),
    # --- link cut: reroute to a backup tower ---
    (
        "L1",
        "link",
        [("link", "T2")],
        "default",
        False,
        True,
        "phones at the railway station show full bars but nothing loads, data totally dead",
    ),
    (
        "L2",
        "link",
        [("link", "T4")],
        "default",
        False,
        False,
        "Fibre cut reported on the T4 backhaul. Restore service to its customers.",
    ),
    (
        "L3",
        "link",
        [("link", "T6")],
        "default",
        False,
        False,
        "Backhaul down at T6 after roadworks. Keep customers in service.",
    ),
    (
        "L4",
        "link",
        [("link", "T1")],
        "default",
        False,
        False,
        "Microwave backhaul failure on T1; the site is up but isolated from the core.",
    ),
    (
        "L5",
        "link",
        [("link", "T3")],
        "default",
        False,
        False,
        "T3 lost its uplink to the core. Sector customers have no data.",
    ),
    # --- multi-fault cascades: ordering across several faults ---
    (
        "M1",
        "multi",
        [("power", "T3"), ("link", "T2")],
        "default",
        False,
        True,
        "lots of complaints: no signal at the mall AND the station has bars but no internet",
    ),
    (
        "M2",
        "multi",
        [("hardware", "T4"), ("power", "T5")],
        "default",
        True,
        False,
        "Two sites down: T4 hardware alarm and T5 power failure. Restore service everywhere.",
    ),
    (
        "M3",
        "multi",
        [("power", "T1"), ("hardware", "T2")],
        "default",
        False,
        False,
        "T1 lost power and T2 has a board failure at the same time; both sectors are out.",
    ),
    (
        "M4",
        "multi",
        [("link", "T5"), ("power", "T6")],
        "default",
        False,
        False,
        "T5 backhaul is cut and T6 lost power. Restore service in all affected sectors.",
    ),
    (
        "M5",
        "multi",
        [("link", "T1"), ("power", "T4"), ("hardware", "T6")],
        "full_repair",
        True,
        False,
        "Major incident: T1 uplink down, T4 power outage, T6 hardware failure. Full repair.",
    ),
]


def build_state(failures):
    state = sample_network()
    for kind, node in failures:
        inject_failure(state, kind, node)
    return state


def main() -> None:
    out = []
    for sid, kind, failures, goal_mode, inject, noisy, text in SCENARIOS:
        state = build_state(failures)
        result = solve(build_problem(state, goal_mode))
        assert result.status == "solved", (sid, result.message)
        out.append(
            {
                "id": sid,
                "type": kind,
                "incident_text": text,
                "noisy_text": noisy,
                "failures": [{"kind": k, "node": n} for k, n in failures],
                "goal_mode": goal_mode,
                "inject_failure_during_execution": inject,
                "expected_triage": {
                    "faults": {n: k for k, n in failures},
                    "affected_sectors": sorted(s.id for s in state.sectors.values() if s.affected),
                },
                "optimal_plan_length": len(result.plan),
                "optimal_plan": [f"{s['action']}({', '.join(s['args'])})" for s in result.plan],
            }
        )
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {len(out)} scenarios to {OUT}")
    for s in out:
        print(
            f"{s['id']:3} {s['goal_mode']:11} opt={s['optimal_plan_length']:2}  "
            f"{' -> '.join(s['optimal_plan'])}"
        )


if __name__ == "__main__":
    main()
