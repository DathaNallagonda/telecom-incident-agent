"""Evaluation harness: LLM-only baseline vs the full system on 20 scenarios.

For each scenario and run:
  1. Triage Agent reads the incident text (shared by both arms, scored vs expected).
  2. System: verify-and-fix loop, then the Operator executes with precondition
     checks and replanning (auto-approved).
  3. Baseline: one LLM plan from the same first prompt, executed as-is with no
     checks and no replanning; steps with unmet preconditions count as unsafe
     or invalid actions attempted.
Goal reached is checked on the TRUE network after execution.

The model is pinned to Groq gpt-oss-120b (one model, clean numbers), the cache
is off (each run is a fresh sample), and every finished run is appended to
eval/results.jsonl, so the harness resumes where it stopped. It stops cleanly
when Groq's daily token limit is hit.

Run:  python -m eval.run_eval [--runs 3] [--only P1,L1] [--pause 20]
Then: python -m eval.summarize
"""

from __future__ import annotations

import os

os.environ.setdefault("LLM_PROVIDERS", "groq")
os.environ.setdefault("LLM_CACHE", "0")
os.environ.setdefault("LLM_MAX_RETRIES", "4")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

warnings.filterwarnings("ignore")

from agents.operator import execute_plan  # noqa: E402
from agents.planner_agent import baseline_plan, plan_with_verification  # noqa: E402
from agents.triage import run_triage  # noqa: E402
from core.llm import GROQ_MODEL  # noqa: E402
from core.simulator import inject_failure, sample_network  # noqa: E402
from core.spec import goal_facts  # noqa: E402

HERE = Path(__file__).parent
SCENARIOS = HERE / "scenarios.json"
RESULTS = HERE / "results.jsonl"

RATE_LIMIT_MARKERS = ("rate_limit", "rate limit", "429", "RateLimitError", "Too Many Requests")
DAILY_MARKERS = ("per day", "TPD", "RPD", "tokens per day", "requests per day")


class RateLimited(Exception):
    def __init__(self, text: str):
        super().__init__(text)
        self.daily = any(m in text for m in DAILY_MARKERS)


def build_state(scenario):
    state = sample_network()
    for f in scenario["failures"]:
        inject_failure(state, f["kind"], f["node"])
    return state


def _check_rate_limited(*texts) -> None:
    for t in texts:
        if t and any(m in t for m in RATE_LIMIT_MARKERS):
            raise RateLimited(t)


def score_triage(triage, expected) -> dict:
    if triage is None:
        return {"ok": False, "exact": False, "fault_acc": 0.0, "sector_acc": 0.0}
    got = {f.node_id: f.fault_type for f in triage.fault_types}
    exp = expected["faults"]
    fault_hits = sum(1 for n, k in exp.items() if got.get(n) == k)
    fault_acc = fault_hits / len(exp) if exp else 1.0
    if set(got) - set(exp):  # extra nodes flagged lower accuracy
        fault_acc *= len(exp) / len(set(got) | set(exp))
    exp_s, got_s = set(expected["affected_sectors"]), set(triage.affected_sectors)
    sector_acc = len(exp_s & got_s) / len(exp_s | got_s) if exp_s | got_s else 1.0
    return {
        "ok": True,
        "exact": got == exp and got_s == exp_s,
        "fault_acc": round(fault_acc, 3),
        "sector_acc": round(sector_acc, 3),
    }


def execution_summary(r) -> dict:
    return {
        "status": r.status,
        "goal_reached": r.goal_reached,
        "goals_unmet": r.goals_unmet,
        "attempted_invalid": r.attempted_invalid,
        "blocked_caught": sum(1 for s in r.log if s.result == "blocked"),
        "replans": len(r.replans),
        "steps_executed": len(r.log),
    }


def run_one(scenario: dict, run: int) -> dict:
    goal_mode = scenario["goal_mode"]
    truth = build_state(scenario)
    goals = goal_facts(truth.to_facts(), goal_mode)
    inject = scenario["inject_failure_during_execution"]

    t0 = time.perf_counter()
    outcome = run_triage(scenario["incident_text"], truth)
    triage_s = time.perf_counter() - t0
    _check_rate_limited(outcome.error)
    record = {
        "scenario": scenario["id"],
        "type": scenario["type"],
        "run": run,
        "model": GROQ_MODEL,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "optimal_len": scenario["optimal_plan_length"],
        "triage": score_triage(outcome.result, scenario["expected_triage"])
        | {"error": outcome.error, "latency_s": round(triage_s, 2)},
    }
    if outcome.result is None:
        record["system"] = record["baseline"] = {
            "valid": False,
            "goal_reached": False,
            "skipped": "triage failed",
        }
        return record

    # --- full system
    t0 = time.perf_counter()
    plan = plan_with_verification(outcome.result, truth.copy(), goal_mode)
    plan_s = time.perf_counter() - t0
    _check_rate_limited(*(r.reason for r in plan.validation_log))
    sys_exec = execute_plan(
        truth.copy(),
        plan.steps,
        goals,
        goals=goal_mode,
        check_first=True,
        allow_replan=True,
        fail_first_restart=inject,
    )
    record["system"] = {
        "source": plan.source,
        "valid": plan.valid,
        "rounds": plan.rounds_used,
        "plan_len": len(plan.steps),
        "steps": [f"{s['action']}({', '.join(s['args'])})" for s in plan.steps],
        "round_reasons": [r.reason for r in plan.validation_log if not r.valid],
        "latency_s": round(triage_s + plan_s, 2),
    } | execution_summary(sys_exec)

    # --- LLM-only baseline
    t0 = time.perf_counter()
    base = baseline_plan(outcome.result, truth.copy(), goal_mode)
    base_s = time.perf_counter() - t0
    _check_rate_limited(base.message)
    base_exec = execute_plan(
        truth.copy(),
        base.steps,
        goals,
        goals=goal_mode,
        check_first=False,
        allow_replan=False,
        fail_first_restart=inject,
    )
    record["baseline"] = {
        "valid": base.valid,
        "plan_len": len(base.steps),
        "steps": [f"{s['action']}({', '.join(s['args'])})" for s in base.steps],
        "invalid_reason": None if base.valid else base.message,
        "latency_s": round(triage_s + base_s, 2),
    } | execution_summary(base_exec)
    return record


def done_runs() -> set[tuple[str, int]]:
    if not RESULTS.exists():
        return set()
    out = set()
    for line in RESULTS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out.add((r["scenario"], r["run"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--only", default="", help="comma-separated scenario ids")
    ap.add_argument("--pause", type=float, default=20, help="seconds between runs (TPM)")
    args = ap.parse_args()

    scenarios = json.loads(SCENARIOS.read_text(encoding="utf-8"))
    if args.only:
        keep = set(args.only.split(","))
        scenarios = [s for s in scenarios if s["id"] in keep]
    done = done_runs()
    todo = [(s, r) for r in range(1, args.runs + 1) for s in scenarios if (s["id"], r) not in done]
    print(f"{len(done)} runs already saved; {len(todo)} to go", flush=True)

    for n, (scenario, run) in enumerate(todo, 1):
        for attempt in range(3):
            try:
                rec = run_one(scenario, run)
                break
            except RateLimited as e:
                if e.daily:
                    print(
                        f"\nDaily rate limit reached: {str(e)[:160]}\n"
                        "Progress is saved; run again tomorrow to continue.",
                        flush=True,
                    )
                    return 2
                print(f"  rate limited, waiting 60 s (attempt {attempt + 1})", flush=True)
                time.sleep(60)
        else:
            print("  still rate limited; stopping. Progress is saved.", flush=True)
            return 2
        with RESULTS.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        s, b = rec["system"], rec["baseline"]
        print(
            f"[{n}/{len(todo)}] {scenario['id']} run {run}: triage "
            f"{'exact' if rec['triage']['exact'] else 'MISS'} | system {s.get('source')} "
            f"valid={s['valid']} goal={s['goal_reached']} | baseline valid={b['valid']} "
            f"goal={b['goal_reached']} invalid_attempts={b.get('attempted_invalid')}",
            flush=True,
        )
        time.sleep(args.pause)
    print("all runs done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
