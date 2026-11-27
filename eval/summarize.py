"""Summarise eval/results.jsonl into eval/results.json, eval/summary.md and two charts.

Run: python -m eval.summarize
"""

from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

HERE = Path(__file__).parent
ROOT = HERE.parent
RESULTS_JSONL = HERE / "results.jsonl"
RESULTS_JSON = HERE / "results.json"
SUMMARY_MD = HERE / "summary.md"
DOCS = ROOT / "docs"

TYPES = ["power", "hardware", "link", "multi"]
TYPE_LABEL = {
    "power": "Power loss",
    "hardware": "Hardware fault",
    "link": "Link cut",
    "multi": "Multi-fault",
    "all": "All scenarios",
}

# Palette: categorical slots 1-2 of the validated default palette (light surface).
SYSTEM_COLOR, BASELINE_COLOR = "#2a78d6", "#eb6834"
SURFACE, TEXT, TEXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def load() -> list[dict]:
    if not RESULTS_JSONL.exists():
        return []
    return [json.loads(x) for x in RESULTS_JSONL.read_text(encoding="utf-8").splitlines() if x]


def pct(values) -> float:
    values = list(values)
    return 100 * sum(1 for v in values if v) / len(values) if values else 0.0


def mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return round(st.mean(values), 2) if values else None


def injected_ids() -> set[str]:
    scenarios = json.loads((HERE / "scenarios.json").read_text(encoding="utf-8"))
    return {s["id"] for s in scenarios if s["inject_failure_during_execution"]}


def summarise(rows: list[dict]) -> dict:
    injected_scenarios = injected_ids()
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["type"]].append(r)
        groups["all"].append(r)
    out = {}
    for key in TYPES + ["all"]:
        g = groups.get(key, [])
        if not g:
            continue
        sys_ = [r["system"] for r in g]
        base = [r["baseline"] for r in g]
        planned = [(r["system"], r["optimal_len"]) for r in g if r["system"].get("plan_len")]
        bplanned = [(r["baseline"], r["optimal_len"]) for r in g if r["baseline"].get("plan_len")]
        injected = [r for r in g if r["scenario"] in injected_scenarios]
        clean = [r for r in g if r["scenario"] not in injected_scenarios]
        out[key] = {
            "runs": len(g),
            "scenarios": len({r["scenario"] for r in g}),
            "valid_plan_rate": {
                "baseline": pct(b["valid"] for b in base),
                "system": pct(s["valid"] for s in sys_),
            },
            "goal_reached_rate": {
                "baseline": pct(b["goal_reached"] for b in base),
                "system": pct(s["goal_reached"] for s in sys_),
            },
            "goal_reached_without_injected_failure": {
                "runs": len(clean),
                "baseline": pct(r["baseline"]["goal_reached"] for r in clean),
                "system": pct(r["system"]["goal_reached"] for r in clean),
            },
            "goal_reached_with_injected_failure": {
                "runs": len(injected),
                "baseline": pct(r["baseline"]["goal_reached"] for r in injected),
                "system": pct(r["system"]["goal_reached"] for r in injected),
            },
            "unsafe_or_invalid_attempted": {
                "baseline": sum(b.get("attempted_invalid", 0) for b in base),
                "system": sum(s.get("attempted_invalid", 0) for s in sys_),
            },
            "runs_with_unsafe_or_invalid": {
                "baseline": pct(b.get("attempted_invalid", 0) > 0 for b in base),
                "system": pct(s.get("attempted_invalid", 0) > 0 for s in sys_),
            },
            "plan_len_over_optimal": {
                "baseline": mean(b["plan_len"] / o for b, o in bplanned),
                "system": mean(s["plan_len"] / o for s, o in planned),
            },
            "triage_exact_rate": pct(r["triage"]["exact"] for r in g),
            "triage_fault_acc": mean(r["triage"]["fault_acc"] for r in g),
            "triage_sector_acc": mean(r["triage"]["sector_acc"] for r in g),
            "avg_llm_rounds": mean(s.get("rounds") for s in sys_),
            "fallback_rate": pct(s.get("source") == "planner_fallback" for s in sys_),
            "llm_fixed_after_feedback": pct(
                s.get("source") == "llm_verified" and (s.get("rounds") or 0) > 1 for s in sys_
            ),
            "replan_recovery_rate": (
                pct(r["system"]["goal_reached"] for r in injected) if injected else None
            ),
            "replan_runs": len(injected),
            "latency_s": {
                "baseline": mean(b.get("latency_s") for b in base),
                "system": mean(s.get("latency_s") for s in sys_),
            },
        }
    return out


def _style(ax, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT2, length=0, labelsize=10)
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.set_ylabel(ylabel, color=TEXT2, fontsize=10)


def grouped_bars(
    summary: dict, metric: str, title: str, ylabel: str, path: Path, percent: bool
) -> None:
    keys = [k for k in TYPES + ["all"] if k in summary]
    base = [summary[k][metric]["baseline"] for k in keys]
    syst = [summary[k][metric]["system"] for k in keys]
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    width, gap = 0.36, 0.03
    xs = range(len(keys))
    for offset, vals, color, label in [
        (-(width + gap) / 2, base, BASELINE_COLOR, "LLM-only baseline"),
        ((width + gap) / 2, syst, SYSTEM_COLOR, "Full system (verified)"),
    ]:
        bars = ax.bar([x + offset for x in xs], vals, width, color=color, label=label)
        for b, v in zip(bars, vals, strict=True):
            ax.annotate(
                f"{v:.0f}%" if percent else f"{v:g}",
                (b.get_x() + b.get_width() / 2, v),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                color=TEXT,
            )
    ax.set_xticks(list(xs), [TYPE_LABEL[k] for k in keys])
    # the "All scenarios" group is a total: set it apart with a hairline
    if "all" in keys and len(keys) > 1:
        ax.axvline(len(keys) - 1.5, color=GRID, linewidth=1)
    if percent:
        ax.set_ylim(0, 110)
        ax.set_yticks([0, 25, 50, 75, 100])
    else:  # counts: whole-number ticks only
        top = max(base + syst + [1])
        ax.set_ylim(0, top * 1.15 + 0.5)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    _style(ax, ylabel)
    ax.set_title(title, loc="left", color=TEXT, fontsize=12, pad=10)
    leg = ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.09),
        ncol=2,
        fontsize=9,
        labelcolor=TEXT2,
    )
    leg.set_in_layout(True)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def write_markdown(summary: dict, rows: list[dict]) -> str:
    a = summary.get("all", {})
    if not a:
        return "# Evaluation summary\n\nNo results yet.\n"
    model = rows[0]["model"] if rows else "?"
    lines = [
        "# Evaluation summary",
        "",
        f"{a['runs']} runs over {a['scenarios']} scenarios; model `{model}`; "
        "goal reached is checked on the true network after execution.",
        "",
        "**Headline:** incidents fully resolved "
        f"{a['goal_reached_rate']['system']:.0f}% (system) vs "
        f"{a['goal_reached_rate']['baseline']:.0f}% (LLM-only baseline); valid plans "
        f"{a['valid_plan_rate']['system']:.0f}% vs {a['valid_plan_rate']['baseline']:.0f}%; "
        f"unsafe or invalid actions attempted: {a['unsafe_or_invalid_attempted']['system']} vs "
        f"{a['unsafe_or_invalid_attempted']['baseline']}.",
        "",
        "| Metric | Baseline | System |",
        "| --- | --- | --- |",
        f"| Valid-plan rate | {a['valid_plan_rate']['baseline']:.0f}% | "
        f"{a['valid_plan_rate']['system']:.0f}% |",
        f"| Goal reached after execution | {a['goal_reached_rate']['baseline']:.0f}% | "
        f"{a['goal_reached_rate']['system']:.0f}% |",
        "| ...without injected failures "
        f"({a['goal_reached_without_injected_failure']['runs']} runs)"
        f" | {a['goal_reached_without_injected_failure']['baseline']:.0f}% | "
        f"{a['goal_reached_without_injected_failure']['system']:.0f}% |",
        f"| ...with an injected restart failure "
        f"({a['goal_reached_with_injected_failure']['runs']} runs) | "
        f"{a['goal_reached_with_injected_failure']['baseline']:.0f}% | "
        f"{a['goal_reached_with_injected_failure']['system']:.0f}% |",
        f"| Unsafe or invalid actions attempted (total) | "
        f"{a['unsafe_or_invalid_attempted']['baseline']} | "
        f"{a['unsafe_or_invalid_attempted']['system']} |",
        f"| Runs with any unsafe or invalid action | "
        f"{a['runs_with_unsafe_or_invalid']['baseline']:.0f}% | "
        f"{a['runs_with_unsafe_or_invalid']['system']:.0f}% |",
        f"| Plan length / optimal | {a['plan_len_over_optimal']['baseline']} | "
        f"{a['plan_len_over_optimal']['system']} |",
        f"| Latency per incident (s) | {a['latency_s']['baseline'] or '— (shares the first call)'}"
        f" | {a['latency_s']['system']} |",
        "",
        "| System-only metric | Value |",
        "| --- | --- |",
        f"| Triage exact match | {a['triage_exact_rate']:.0f}% |",
        f"| Triage fault accuracy / sector accuracy | {a['triage_fault_acc']} / "
        f"{a['triage_sector_acc']} |",
        f"| Average LLM plan rounds | {a['avg_llm_rounds']} |",
        f"| LLM plan fixed after verifier feedback | {a['llm_fixed_after_feedback']:.0f}% |",
        f"| Planner fallback rate | {a['fallback_rate']:.0f}% |",
        "| Replan recovery rate | "
        + (
            f"{a['replan_recovery_rate']:.0f}% of {a['replan_runs']} runs |"
            if a["replan_recovery_rate"] is not None
            else "n/a |"
        ),
        "",
        "## By scenario type",
        "",
        "| Type | Runs | Valid plans (base / sys) | Goal reached (base / sys) | "
        "Unsafe attempts (base / sys) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for k in TYPES:
        if k in summary:
            s = summary[k]
            lines.append(
                f"| {TYPE_LABEL[k]} | {s['runs']} | {s['valid_plan_rate']['baseline']:.0f}% / "
                f"{s['valid_plan_rate']['system']:.0f}% | {s['goal_reached_rate']['baseline']:.0f}%"
                f" / {s['goal_reached_rate']['system']:.0f}% | "
                f"{s['unsafe_or_invalid_attempted']['baseline']} / "
                f"{s['unsafe_or_invalid_attempted']['system']} |"
            )
    lines += [
        "",
        "## How to read this",
        "",
        "- The baseline has no verifier and no replanning, by design. In the 4 scenarios with "
        "an injected restart failure it cannot recover, so most of the goal-reached gap measures "
        "replanning, not plan quality.",
        "- The verifier's direct effect shows in valid plans and unsafe actions: an LLM plan that "
        "breaks a precondition is fixed before it runs (system) or runs as-is (baseline).",
        "- The system plans from what triage believes. When triage misses an affected sector, "
        "the first plan misses it too; only a replan from the live network recovers it.",
        "- One model (Groq gpt-oss-120b) for both arms; temperature 0; cache off so each run "
        "is a fresh sample.",
        "",
        "![Goal reached](../docs/goal_reached_rate.png)",
        "![Valid plans](../docs/valid_plan_rate.png)",
        "![Unsafe actions](../docs/unsafe_actions.png)",
        "",
    ]
    return "\n".join(lines)


MAIN_MODEL = "openai/gpt-oss-120b"
CHARTS = ["valid_plan_rate.png", "goal_reached_rate.png", "unsafe_actions.png"]


def model_comparison(by_model: dict[str, dict]) -> str:
    """Side-by-side table: does the verifier matter more for a weaker model?"""
    lines = [
        "## Model comparison",
        "",
        "| Model | Runs | Valid plans (base / sys) | Goal reached (base / sys) | "
        "Unsafe attempts (base / sys) | LLM fixed after feedback | Fallback |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for model, s in by_model.items():
        a = s["all"]
        lines.append(
            f"| `{model}` | {a['runs']} | {a['valid_plan_rate']['baseline']:.0f}% / "
            f"{a['valid_plan_rate']['system']:.0f}% | {a['goal_reached_rate']['baseline']:.0f}% / "
            f"{a['goal_reached_rate']['system']:.0f}% | "
            f"{a['unsafe_or_invalid_attempted']['baseline']} / "
            f"{a['unsafe_or_invalid_attempted']['system']} | "
            f"{a['llm_fixed_after_feedback']:.0f}% | {a['fallback_rate']:.0f}% |"
        )
    return "\n".join(lines) + "\n"


README = ROOT / "README.md"
START, END = "<!-- results:start -->", "<!-- results:end -->"


def update_readme(summary: dict, main_model: str | None, by_model: dict[str, dict]) -> None:
    """Rewrite the README's results block so its numbers always match the latest runs."""
    if not README.exists():
        return
    text = README.read_text(encoding="utf-8")
    if START not in text or END not in text:
        return
    a = summary.get("all")
    if not a:
        block = "_Benchmark in progress; results will appear here._"
    else:
        clean = a["goal_reached_without_injected_failure"]
        inj = a["goal_reached_with_injected_failure"]
        lines = [
            f"Model `{main_model}`, {a['runs']} runs over {a['scenarios']} scenarios.",
            "",
            "| Metric | LLM-only baseline | Verified system |",
            "| --- | --- | --- |",
            f"| Valid plans | {a['valid_plan_rate']['baseline']:.0f}% | "
            f"{a['valid_plan_rate']['system']:.0f}% |",
            f"| Unsafe or invalid actions attempted | "
            f"{a['unsafe_or_invalid_attempted']['baseline']} | "
            f"{a['unsafe_or_invalid_attempted']['system']} |",
            f"| Incident resolved, no injected failure ({clean['runs']} runs) | "
            f"{clean['baseline']:.0f}% | {clean['system']:.0f}% |",
            f"| Incident resolved, restart fails mid-run ({inj['runs']} runs) | "
            f"{inj['baseline']:.0f}% | {inj['system']:.0f}% |",
            f"| Plan length / optimal | {a['plan_len_over_optimal']['baseline']} | "
            f"{a['plan_len_over_optimal']['system']} |",
            "",
            f"Triage exact match {a['triage_exact_rate']:.0f}%; LLM plans fixed after "
            f"verifier feedback {a['llm_fixed_after_feedback']:.0f}%; pyperplan fallback "
            f"{a['fallback_rate']:.0f}%.",
        ]
        for model, s in by_model.items():
            if model != main_model and s.get("all"):
                o = s["all"]
                lines += [
                    "",
                    f"Weaker model `{model}` ({o['runs']} runs): valid plans "
                    f"{o['valid_plan_rate']['baseline']:.0f}% (baseline) vs "
                    f"{o['valid_plan_rate']['system']:.0f}% (system); unsafe or invalid "
                    f"actions {o['unsafe_or_invalid_attempted']['baseline']} vs "
                    f"{o['unsafe_or_invalid_attempted']['system']}.",
                ]
        lines += [
            "",
            "![Goal reached](docs/goal_reached_rate.png)",
            "",
            "| Valid plans | Unsafe or invalid actions |",
            "| --- | --- |",
            "| ![Valid plans](docs/valid_plan_rate.png) | "
            "![Unsafe actions](docs/unsafe_actions.png) |",
        ]
        block = "\n".join(lines)
    i, j = text.index(START) + len(START), text.index(END)
    README.write_text(text[:i] + "\n" + block + "\n" + text[j:], encoding="utf-8")


def main() -> None:
    all_rows = load()
    by_model_rows: dict[str, list[dict]] = defaultdict(list)
    for r in all_rows:
        by_model_rows[r["model"]].append(r)
    main_model = MAIN_MODEL if MAIN_MODEL in by_model_rows else next(iter(by_model_rows), None)
    rows = by_model_rows.get(main_model, [])
    summary = summarise(rows)
    by_model = {m: summarise(r) for m, r in by_model_rows.items()}
    RESULTS_JSON.write_text(
        json.dumps(
            {"main_model": main_model, "summary": summary, "by_model": by_model, "runs": all_rows},
            indent=1,
        ),
        encoding="utf-8",
    )
    text = write_markdown(summary, rows)
    if len(by_model) > 1:
        text += "\n" + model_comparison(by_model)
    SUMMARY_MD.write_text(text, encoding="utf-8")
    update_readme(summary, main_model, by_model)
    DOCS.mkdir(exist_ok=True)
    if not summary:  # no results: remove stale charts so the app shows none
        for name in CHARTS:
            (DOCS / name).unlink(missing_ok=True)
    else:
        grouped_bars(
            summary,
            "valid_plan_rate",
            "Valid plans: LLM-only vs verified system",
            "Runs with a valid plan (%)",
            DOCS / "valid_plan_rate.png",
            True,
        )
        grouped_bars(
            summary,
            "goal_reached_rate",
            "Incident fully resolved on the real network",
            "Runs reaching the goal (%)",
            DOCS / "goal_reached_rate.png",
            True,
        )
        grouped_bars(
            summary,
            "unsafe_or_invalid_attempted",
            "Unsafe or invalid actions attempted on the network",
            "Actions attempted (total)",
            DOCS / "unsafe_actions.png",
            False,
        )
    print(SUMMARY_MD.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
